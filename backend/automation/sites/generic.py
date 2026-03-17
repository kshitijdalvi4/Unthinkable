"""
Generic form filler for unknown job sites.
Uses label-text matching heuristics.
"""
from playwright.async_api import Page
import asyncio


# Map commonly used label keywords → candidate data keys
LABEL_FIELD_MAP = [
    (["full name", "your name", "name"],               "name"),
    (["email", "e-mail"],                              "email"),
    (["phone", "mobile", "contact"],                   "phone"),
    (["github", "portfolio", "website"],               "github"),
    (["salary", "ctc", "compensation", "expected"],    "salary"),
    (["why", "motivation", "interest", "cover"],       "why"),
    (["years of experience", "experience"],            "experience_years"),
    (["skills", "technical skills"],                   "skills"),
    (["linkedin"],                                     "linkedin"),
    (["visa", "sponsorship", "authorized"],            "visa"),
    (["notice", "availability"],                       "notice"),
    (["gender", "sex"],                                "gender"),
    (["race", "ethnicity", "hispanic"],                "race"),
    (["veteran", "military"],                          "veteran"),
    (["disability", "handicap"],                       "disability"),
    (["internshala", "profile link"],                  "linkedin"),
    (["years of experience", "total exp", "relevant exp"], "experience_years"),
    (["hometown", "current city", "current location"], "notice"), 
    (["cover letter", "how would you", "hired"],       "why"),
]


def _get_value(candidate: dict, key: str) -> str:
    kb = candidate.get("knowledge_base", {})
    ai = candidate.get("additional_info", {})
    
    # Priority: Knowledge Base -> Additional Info -> Top Level
    direct_map = {
        "name":             kb.get("name") or candidate.get("name", ""),
        "email":            kb.get("email") or candidate.get("email", ""),
        "experience_years": str(kb.get("experience_years") or candidate.get("experience_years", "0")),
        "skills":           kb.get("skills") or ", ".join(candidate.get("skills", [])[:6]),
        "phone":            kb.get("phone") or ai.get("phone", ""),
        "github":           ai.get("github", ""),
        "linkedin":         ai.get("linkedin", ""),
        "salary":           kb.get("expected_salary") or ai.get("salary", ""),
        "why":              ai.get("why", ""),
        "visa":             "No", # Default
        "notice":           kb.get("notice_period", ""),
        "gender":           "Decline to self-identify",
        "race":             "Decline to self-identify",
        "veteran":          "I am not a protected veteran",
        "disability":       "No, I don't have a disability",
    }
    return direct_map.get(key, "")


async def handle_generic(page: Page, job_url: str, candidate_data: dict) -> dict:
    filled_fields = []
    print(f"[Generic] Navigating to {job_url}")

    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception as e:
        print(f"[Generic] Navigation/Wait warning: {e}")

    await asyncio.sleep(2)

    # Handling for sites that have an initial "Apply" or "Apply Now" button before the form
    apply_btn = page.locator("button:has-text('Apply'), a:has-text('Apply Now'), .ashby-apply-button, .workday-apply-button").first
    if await apply_btn.is_visible(timeout=2000):
        print("[Generic] Initial Apply button found. Clicking...")
        await apply_btn.click()
        await asyncio.sleep(3)

    # Multi-step loop
    for step in range(10):
        print(f"[Generic] Processing step {step+1}...")
        
        # 1. Fill fields on current page
        step_filled = await fill_page_fields(page, candidate_data)
        filled_fields.extend(step_filled)
        
        # 2. Handle File Upload (Resume)
        resume_input = page.locator("input[type='file'][accept*='pdf'], input[type='file'][name*='resume']").first
        if await resume_input.count() > 0:
             print("[Generic] Resume upload field detected. (Requires manual upload for now)")
             filled_fields.append("Resume Upload (Detected)")

        # 3. Detect "Next" or "Continue" buttons
        # Workday/Ashby often use specific buttons
        next_btn = page.locator("button:has-text('Next'), button:has-text('Continue'), [data-automation-id='bottom-navigation-next-button']").first
        submit_btn = page.locator("button:has-text('Submit'), button:has-text('Finish'), [data-automation-id='bottom-navigation-submit-button']").first
        
        if await submit_btn.is_visible(timeout=500):
            print("[Generic] Submit button found. Stopping for user review.")
            break
        elif await next_btn.is_visible(timeout=500):
            print("[Generic] Clicking Next...")
            await next_btn.click()
            await asyncio.sleep(2)
        else:
            print("[Generic] No more navigation buttons found.")
            break

    return {
        "status": "partial" if filled_fields else "needs_manual",
        "message": f"Generic filler: processed {len(filled_fields)} interaction(s). Please review and submit.",
        "filled_fields": list(set(filled_fields)),
    }

async def fill_page_fields(page: Page, candidate: dict) -> list:
    filled = []
    
    # Get all potential form elements
    # Common containers for labels and inputs
    containers = await page.locator("div:has(label), .field, .form-group, .ashby-job-posting-form-field").all()
    
    for container in containers:
        try:
            # 1. Standard label lookup
            label_el = container.locator("label").first
            label_text = ""
            if await label_el.count() > 0:
                label_text = (await label_el.inner_text(timeout=500)).lower()
            
            # 2. Proximity fallback: Look for nearby text if label is missing or empty
            if not label_text:
                label_text = await container.evaluate("""(el) => {
                    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
                    let text = "";
                    let node;
                    while(node = walker.nextNode()) {
                        text += node.textContent + " ";
                    }
                    return text.toLowerCase().trim();
                }""")

            # 3. Deep Proximity: Check preceding sibling or parent header
            if not label_text or len(label_text) < 2:
                label_text = await container.evaluate("""(el) => {
                    const prev = el.previousElementSibling;
                    if (prev && prev.innerText) return prev.innerText.toLowerCase();
                    const parent = el.parentElement;
                    if (parent && parent.querySelector('h1, h2, h3, h4')) return parent.querySelector('h1, h2, h3, h4').innerText.toLowerCase();
                    return "";
                }""")

            # Match label to candidate data
            target_key = None
            for keywords, key in LABEL_FIELD_MAP:
                if any(kw in str(label_text) for kw in keywords):
                    target_key = key
                    break
            
            if not target_key: continue
            value = _get_value(candidate, str(target_key))
            if not value: continue
            
            # Find input in same container
            inp = container.locator("input:not([type='hidden']):not([type='submit']), textarea, select").first
            if await inp.count() == 0:
                # Fallback: find input in next sibling if current container is just a label
                inp = page.locator(f"xpath=//div[contains(text(), '{label_text[:20]}')]/following::input[1]")
                if await inp.count() == 0: continue
            
            tag = await inp.evaluate("el => el.tagName.toLowerCase()")
            itype = await inp.get_attribute("type") or "text"
            
            if tag == "input" and itype in ["text", "email", "tel", "number"]:
                curr = await inp.input_value()
                if not curr:
                    await inp.fill(value)
                    filled.append(str(label_text)[:20])
            elif tag == "textarea":
                curr = await inp.input_value()
                if not curr:
                    await inp.fill(value)
                    filled.append(str(label_text)[:20])
            elif tag == "select":
                # Basic select matching
                await inp.select_option(label={"contains": value})
                filled.append(str(label_text)[:20])
            elif itype == "radio":
                # Handle Yes/No radio pairs
                is_yes = any(w in value.lower() for w in ["yes", "true", "1"])
                radio_label = container.locator(f"label:has-text('{'Yes' if is_yes else 'No'}')").first
                if await radio_label.count() > 0:
                    await radio_label.click()
                    filled.append(str(label_text)[:20])

        except Exception as e:
            print(f"[Generic] Field fill error: {e}")
            continue
            
    return filled
