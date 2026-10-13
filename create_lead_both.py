"""
create_lead_both.py

What this script does, in plain words:
1. Reads a business card photo ONCE
2. Sends it to a free vision model via OpenRouter ONCE to extract the fields
   (no point paying for/running extraction twice for two destinations)
3. Creates a Lead in ERPNext using that extracted data
4. Creates a Lead in Frappe CRM using that same extracted data
5. Prints a clear summary of what happened in EACH system separately -
   if one succeeds and the other fails (or is skipped as a duplicate),
   you'll see both outcomes clearly, not just one

This combines create_lead_erpnext.py and create_lead_crm.py, which you
already confirmed work correctly on their own. Nothing about the
extraction, field mapping, or duplicate-check logic has changed from
those two scripts - this just runs both destination steps back-to-back
using one shared extraction result, instead of running two separate
scripts (and re-extracting the same card twice).

NOTE: ERPNext's Lead and Frappe CRM's Lead are two independent systems
with two independent duplicate lists. A card already in ERPNext but not
yet in CRM (or vice versa) is NOT a bug - each system's duplicate check
only looks at its own existing leads.

CHANGES IN THIS VERSION:
- job_title: prompt now explains it must be the PERSON's designation, not
  the company's business type (e.g. "Tours and Travels"), plus a code
  safety net that clears it if it's just part of the company name.
- email: always converted to lowercase (also makes duplicate checks
  reliable, since John@Site.com and john@site.com now match).
- phone: country code (+91 / 91 / 0091) and leading 0 are removed, only
  the main 10 digits are stored.
- full_name: if a card lists more than one person, only ONE person is
  used (the first / most prominent), and their job title, phone and email
  are kept together. The prompt tells the model never to combine two
  people's names into one field.
"""

import os
import sys
import sys
from dotenv import load_dotenv
load_dotenv()
import json
import base64
import re
import requests
from openai import OpenAI

# ============================================================
# SETTINGS
# ============================================================

# --- OpenRouter (extraction) ---
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_NAME = "openrouter/free"

# --- ERPNext (storage) ---
ERPNEXT_SITE_URL = os.environ.get("ERPNEXT_SITE_URL", "https://erp.localhost")
ERPNEXT_API_KEY = os.environ["ERPNEXT_API_KEY"]
ERPNEXT_API_SECRET = os.environ["ERPNEXT_API_SECRET"]

# --- Frappe CRM (storage) ---
# Frappe CRM runs on the same site as ERPNext, so this is the same site
# URL and API key/secret.
CRM_SITE_URL = os.environ.get("CRM_SITE_URL", "https://erp.localhost")
CRM_API_KEY = os.environ["CRM_API_KEY"]
CRM_API_SECRET = os.environ["CRM_API_SECRET"]

# --- Card to process ---
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IMAGE_PATH", "")

# ============================================================
# STEP 1: Extract fields from the card photo (ONCE, shared by both)
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

instruction = """
You are reading a photo of a business card. Extract exactly these eight fields:
- full_name
- job_title
- company
- phone
- email
- website
- address
- city

Rules:
- If a field does not clearly appear on the card, return it as an empty string "".
- Never guess or invent a value that is not visibly printed on the card.
- "full_name" must be ONE single person's name only. If the card lists more
  than one person, use only the first or most prominent one (the name at the
  top or in the largest print) and ignore the others. NEVER combine two
  people's names into one field. The "job_title", "phone" and "email" you
  return must belong to that same person - do not mix one person's name
  with another person's details.
- "job_title" is the PERSON's own designation at the company, such as
  Manager, Director, Founder, Proprietor, Sales Executive. It is NOT the
  company's tagline, business type, or list of services (for example
  "Tours and Travels", "Software Solutions", "Interior Designers").
  If the card shows only a business description and no personal
  designation, return "" for job_title.
- "phone" must be ONE single phone number only, even if the card lists
  several (e.g. separate "Office" and "Mobile" numbers). If more than one
  number is printed, prefer the one labelled "Mobile" or "Cell". If there's
  no such label, use the first number printed on the card. Do not combine
  multiple numbers into one field, and do not use commas or slashes.
- "address" should be the full postal address as printed (street, area, etc).
- "city" should be just the city/town name on its own, pulled out separately
  from the full address.
- Return ONLY a JSON object with exactly these eight keys.
- Do not include any explanation, notes, or markdown formatting like ```json.
- Field values must be plain text only. Do not use Markdown link syntax
  like [text](url) for the website or email fields.
"""

print("Reading card and extracting fields...")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
)

# Defensive check: if OpenRouter couldn't fulfil the request (model
# unavailable, rate limited, etc.), it can come back with an empty/missing
# choices list instead of raising a normal error.
if not response.choices or response.choices[0].message.content is None:
    print("The model didn't return any text - it may be overloaded or unavailable.")
    print("Raw response from OpenRouter:")
    print(response)
    raise SystemExit(1)

raw_text = response.choices[0].message.content.strip()
raw_text = raw_text.replace("```json", "").replace("```", "").strip()

# Defensive check: even non-None content can come back empty, or not be
# valid JSON at all (e.g. a moderation classifier's output like
# "User Safety: safe" instead of an actual extraction attempt - seen once
# already with the openrouter/free auto-router).
if not raw_text:
    print("The model returned an empty response after cleanup.")
    print("Raw response from OpenRouter:")
    print(response)
    raise SystemExit(1)

try:
    extracted = json.loads(raw_text)
except json.JSONDecodeError:
    print("The model's response wasn't valid JSON - here's what it actually sent:")
    print(raw_text)
    raise SystemExit(1)

# Safety net: strip accidental Markdown link formatting from any field.
def strip_markdown_link(value):
    match = re.match(r"^\[(.+?)\]\((.+?)\)$", value)
    if match:
        return match.group(1)
    return value

# Safety net: even with the instruction above, the model can still return
# more than one phone number crammed into one field (e.g. an "Office" and
# a "Mobile" number separated by a comma or slash), or a number with a
# country code (+91 / 91 / 0091) or a leading 0. ERPNext and CRM both
# want ONE plain number, so this reduces whatever the model sent to just
# the main 10 digits.
def clean_phone_number(value):
    if not value:
        return value
    # 1. If several numbers are crammed together, keep only the first one.
    first_number = re.split(r"[,/;]| or ", value)[0].strip()
    # 2. Keep digits only (drops +, spaces, dashes, brackets, dots).
    digits = re.sub(r"\D", "", first_number)
    # 3. Remove country code / leading-zero prefixes. The "91" is only
    #    removed when the total length says it's a country code (12 digits),
    #    because a genuine 10-digit mobile can itself start with 91.
    if digits.startswith("00"):                              # 0091 98765 43210
        digits = digits[2:]
    if len(digits) == 13 and digits.startswith("910"):       # +91 (0) 98765 43210
        digits = digits[3:]
    elif len(digits) == 12 and digits.startswith("91"):      # +91 98765 43210
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):       # 0 98765 43210
        digits = digits[1:]
    return digits

# 1) Strip Markdown link formatting from every text field
for key in extracted:
    if isinstance(extracted[key], str):
        extracted[key] = strip_markdown_link(extracted[key])

# 2) Phone: single number, 10 digits, no country code
if "phone" in extracted:
    original_phone = extracted["phone"]
    extracted["phone"] = clean_phone_number(original_phone)
    if extracted["phone"] != original_phone:
        print(f"Note: phone cleaned: '{original_phone}' -> '{extracted['phone']}'")

# 3) Email: always lowercase, no stray spaces
if extracted.get("email"):
    extracted["email"] = extracted["email"].strip().lower()

# 4) Job title: if it's just part of the company name (e.g. company
#    "Sharma Tours and Travels", job title "Tours and Travels"), it's the
#    business type, not a real job title - clear it.
job_title_value = (extracted.get("job_title") or "").strip()
company_value = (extracted.get("company") or "").strip()
if job_title_value and company_value and job_title_value.lower() in company_value.lower():
    print(f"Note: job title '{job_title_value}' looks like part of the company name - clearing it.")
    extracted["job_title"] = ""

print("Extracted fields:")
print(json.dumps(extracted, indent=2))


# ============================================================
# PART A: Create the Lead in ERPNext
# ============================================================

def create_erpnext_lead(extracted):
    """
    Maps extracted fields to ERPNext's Lead fieldnames, checks for a
    duplicate by email, creates the Lead, and creates/links an Address
    if we have enough info to do so properly. Returns nothing - just
    prints its own outcome, same as the standalone script did.
    """
    print("\n--- ERPNext ---")

    lead_payload = {
        "lead_name": extracted.get("full_name", ""),
        "company_name": extracted.get("company", ""),
        "email_id": extracted.get("email", ""),
        "mobile_no": extracted.get("phone", ""),
        "website": extracted.get("website", ""),
        "job_title": extracted.get("job_title", ""),
        "city": extracted.get("city", ""),
        "country": "India",
    }
    lead_payload = {k: v for k, v in lead_payload.items() if v}

    url = f"{ERPNEXT_SITE_URL}/api/resource/Lead"
    headers = {
        "Authorization": f"token {ERPNEXT_API_KEY}:{ERPNEXT_API_SECRET}",
        "Content-Type": "application/json",
    }

    skip_creation = False
    email_to_check = extracted.get("email", "")

    if email_to_check:
        check_result = requests.get(
            url,
            headers=headers,
            params={"filters": json.dumps([["email_id", "=", email_to_check]])},
            verify=False,
        )
        if check_result.status_code == 200:
            existing = check_result.json().get("data", [])
            if existing:
                existing_id = existing[0]["name"]
                print(
                    f"Already exists as {existing_id} "
                    f"(matched on email {email_to_check}) - skipping."
                )
                skip_creation = True
        else:
            print(
                f"Warning: could not check for duplicates (status "
                f"{check_result.status_code}). Continuing anyway..."
            )
    else:
        print("No email was extracted, so skipping the duplicate check.")

    result = None
    if not skip_creation:
        result = requests.post(url, headers=headers, json=lead_payload, verify=False)
        if result.status_code == 200:
            new_lead = result.json()["data"]
            print(f"Success! Lead created: {new_lead['name']}")
        else:
            print(f"Something went wrong (status {result.status_code}):")
            print(result.text)

    # Address (only if Lead creation succeeded and we have both fields)
    if (
        result is not None
        and result.status_code == 200
        and extracted.get("address")
        and extracted.get("city")
    ):
        new_lead_name = result.json()["data"]["name"]
        address_payload = {
            "address_title": extracted.get("company") or extracted.get("full_name"),
            "address_type": "Office",
            "address_line1": extracted["address"],
            "city": extracted["city"],
            "country": "India",
            "links": [{"link_doctype": "Lead", "link_name": new_lead_name}],
        }
        address_url = f"{ERPNEXT_SITE_URL}/api/resource/Address"
        address_result = requests.post(
            address_url, headers=headers, json=address_payload, verify=False
        )
        if address_result.status_code == 200:
            print(f"Address created and linked to {new_lead_name}")
        else:
            print(f"Address creation failed (status {address_result.status_code}):")
            print(address_result.text)
    elif not skip_creation and extracted.get("address"):
        print("Note: an address was extracted but NOT saved - missing a city value:")
        print(f"  {extracted['address']}")


# ============================================================
# PART B: Create the Lead in Frappe CRM
# ============================================================

def create_crm_lead(extracted):
    """
    Maps extracted fields to CRM Lead's fieldnames, checks for a duplicate
    by email, and creates the Lead. CRM Lead has no address/city/company
    fields the way ERPNext's Lead does, so those are noted but not saved.
    """
    print("\n--- Frappe CRM ---")

    full_name = extracted.get("full_name", "").strip()
    if full_name:
        name_parts = full_name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
    else:
        first_name = ""
        last_name = ""

    lead_payload = {
        "first_name": first_name,
        "last_name": last_name,
        "lead_name": full_name,
        "organization": extracted.get("company", ""),
        "job_title": extracted.get("job_title", ""),
        "email": extracted.get("email", ""),
        "mobile_no": extracted.get("phone", ""),
        "website": extracted.get("website", ""),
    }
    lead_payload = {k: v for k, v in lead_payload.items() if v}

    url = f"{CRM_SITE_URL}/api/resource/CRM Lead"
    headers = {
        "Authorization": f"token {CRM_API_KEY}:{CRM_API_SECRET}",
        "Content-Type": "application/json",
    }

    skip_creation = False
    email_to_check = extracted.get("email", "")

    if email_to_check:
        check_result = requests.get(
            url,
            headers=headers,
            params={"filters": json.dumps([["email", "=", email_to_check]])},
            verify=False,
        )
        if check_result.status_code == 200:
            existing = check_result.json().get("data", [])
            if existing:
                existing_id = existing[0]["name"]
                print(
                    f"Already exists as {existing_id} "
                    f"(matched on email {email_to_check}) - skipping."
                )
                skip_creation = True
        else:
            print(
                f"Warning: could not check for duplicates (status "
                f"{check_result.status_code}). Continuing anyway..."
            )
    else:
        print("No email was extracted, so skipping the duplicate check.")

    if not skip_creation:
        result = requests.post(url, headers=headers, json=lead_payload, verify=False)
        if result.status_code == 200:
            new_lead = result.json()["data"]
            print(f"Success! CRM Lead created: {new_lead['name']}")
        else:
            print(f"Something went wrong (status {result.status_code}):")
            print(result.text)

        unsaved_notes = []
        if extracted.get("address"):
            unsaved_notes.append(f"Address: {extracted['address']}")
        if extracted.get("city"):
            unsaved_notes.append(f"City: {extracted['city']}")
        if unsaved_notes:
            print("Note: CRM Lead has no field for the following, not saved:")
            for note in unsaved_notes:
                print(f"  - {note}")


# ============================================================
# Run both, independently - one failing doesn't stop the other
# ============================================================

try:
    create_erpnext_lead(extracted)
except Exception as e:
    print(f"\n--- ERPNext ---\nUnexpected error, skipped: {e}")

try:
    create_crm_lead(extracted)
except Exception as e:
    print(f"\n--- Frappe CRM ---\nUnexpected error, skipped: {e}")

print("\nDone - check the outcomes for each system above.")
