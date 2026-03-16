"""
LinkedIn-specific form filler.
Handles login and Easy Apply flow.
"""
from playwright.async_api import Page, Locator
from typing import Optional
import asyncio


LINKEDIN_LOGIN_URL = "https://www.linkedin.com/login"


async def robust_click(page: Page, locator: Locator, timeout=5000):
    """Unified clicking strategy that stops after the first successful attempt (including popups)."""
    try:
        if await locator.count() == 0: return
        
        initial_url = page.url
        context = page.context
        initial_page_count = len(context.pages)
        
        async def is_successful():
            print(f"[LinkedIn] DEBUG: is_successful check. URL: {page.url}, Pages: {len(context.pages)}")
            
            # New tab is the highest indicator of success for External Apply
            if len(context.pages) > initial_page_count:
                print(f"[LinkedIn] DEBUG: New tab detected. Count: {len(context.pages)}")
                return True

            # Modal is the best indicator of success for Easy Apply
            if await page.locator(".artdeco-modal, .jobs-easy-apply-modal, .artdeco-modal__content").is_visible(timeout=500):
                print("[LinkedIn] DEBUG: Modal detected.")
                return True
                
            # If URL changed, check if it's a "bad" redirect
            low_url = page.url.lower()
            if page.url != initial_url:
                print(f"[LinkedIn] DEBUG: URL changed to {page.url}")
                # If we are on a job page or feed/form, it's likely a success or progress
                if "jobs/view" in low_url or "feed" in low_url or "apply" in low_url or "login" in low_url:
                    return True
                if "/jobs/collections" in low_url and "view" not in low_url:
                    print("[LinkedIn] Undesirable diagnostic: redirected to collections.")
                    return False
                return True
                
            return False

        async def recover_if_needed():
            low_url = page.url.lower()
            if "/jobs/collections" in low_url and "view" not in low_url:
                print("[LinkedIn] Recovering from collections redirect...")
                await page.goto(initial_url, wait_until="domcontentloaded")
                await asyncio.sleep(2)
                return True
            return False

        # Strategy A: Focus and Enter
        print("[LinkedIn] Attempting Strategy A: Keyboard Enter...")
        await locator.scroll_into_view_if_needed()
        # Scroll a bit more to be safe
        await page.evaluate("(el) => { const rect = el.getBoundingClientRect(); window.scrollBy(0, rect.top - 100); }", await locator.element_handle())
        await locator.focus()
        await asyncio.sleep(0.5)
        await page.keyboard.press("Enter")
        await asyncio.sleep(2.0)
        
        if await is_successful():
            print("[LinkedIn] Strategy A succeeded.")
            return
        await recover_if_needed()

        # Strategy B: Physical Mouse Simulation
        print("[LinkedIn] Attempting Strategy B: Physical Mouse...")
        await locator.scroll_into_view_if_needed()
        box = await locator.bounding_box()
        if box:
            cx, cy = box['x'] + box['width']/2, box['y'] + box['height']/2
            # Move mouse slowly to simulate human interaction
            await page.mouse.move(cx, cy, steps=10)
            await asyncio.sleep(0.2)
            await page.mouse.click(cx, cy)
            await asyncio.sleep(2.0)

        if await is_successful():
            print("[LinkedIn] Strategy B succeeded.")
            return
        await recover_if_needed()

        # Strategy C: Standard Click
        print("[LinkedIn] Attempting Strategy C: Standard Click...")
        try:
            if await locator.is_visible():
                await locator.click(force=True, timeout=timeout)
                await asyncio.sleep(2.5)
                if await is_successful():
                    print("[LinkedIn] Strategy C succeeded.")
                    return
        except: pass
        await recover_if_needed()

        # Strategy D: JS Force Click
        print("[LinkedIn] Attempting Strategy D: JS Force Click...")
        try:
            await page.evaluate("(el) => el.click()", await locator.element_handle())
            await asyncio.sleep(2.5)
            if await is_successful():
                print("[LinkedIn] Strategy D succeeded.")
                return
        except: pass
        await recover_if_needed()

        # Strategy E: Direct Anchor Navigation (Fallback for stubborn external buttons)
        print("[LinkedIn] Attempting Strategy E: Direct Anchor Navigation...")
        try:
            info = await page.evaluate("""(el) => {
                const a = el.closest('a');
                return a ? { href: a.href, target: a.target } : null;
            }""", await locator.element_handle())
            
            if info and info['href'] and "linkedin.com" not in info['href'] or "/redir/" in info['href']:
                print(f"[LinkedIn] Force-opening external link: {info['href']}")
                new_tab = await context.new_page()
                await new_tab.goto(info['href'], wait_until="domcontentloaded")
                # Wait a bit for the redirect to settle
                await asyncio.sleep(3)
                if len(context.pages) > initial_page_count:
                    print("[LinkedIn] Strategy E succeeded via new tab.")
                    return
        except Exception as e:
            print(f"[LinkedIn] Strategy E failed: {e}")

    except Exception as e:
        print(f"[LinkedIn] robust_click major error: {e}")


async def _login(page: Page, credentials: dict) -> bool:
    """Log into LinkedIn. Returns True if successful."""
    print("[LinkedIn] Navigating to login page...")
    await page.goto(LINKEDIN_LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(2)
    
    # Check if we are already logged in
    if "feed" in page.url or await page.locator(".feed-identity-module").is_visible(timeout=2000):
        print("[LinkedIn] Already logged in.")
        return True
    
    try:
        # Check for 'Welcome back' screen (only password showing)
        if await page.locator("#password").is_visible(timeout=3000) and not await page.locator("#username").is_visible(timeout=1000):
            print("[LinkedIn] 'Welcome back' screen detected. Trying to click 'Sign in as another account' to get full form...")
            # Often there's a link to change account
            another_account = page.locator("button:has-text('Sign in as another account'), a:has-text('Sign in as another account')").first
            if await another_account.is_visible(timeout=2000):
                await another_account.click()
                await asyncio.sleep(2)

        # Wait for username/email field
        username_selectors = [
            "#username", 
            "input[name='session_key']", 
            "input[autocomplete='username']", 
            "input[type='email']",
            "input[aria-label*='Email']",
            "input[placeholder*='Email']",
            "input[id*=':r0:']",
            "input[id*=':r3:']"
        ]
        
        current_selector = None
        for sel in username_selectors:
            try:
                # Use visible and attached check
                loc = page.locator(sel).first
                if await loc.is_visible(timeout=2000):
                    current_selector = sel
                    break
            except:
                continue
        
        if not current_selector:
            # Maybe it's just 'input' with a label nearby? 
            # Look for the email field by its common name attribute first if ids fail
            email_field = page.locator("input[name='session_key']").first
            if await email_field.is_visible(timeout=1000):
                current_selector = "input[name='session_key']"

        if not current_selector:
            # Try broader 'Sign in' search if we are on a landing page instead of login page
            print("[LinkedIn] Username field not found. Checking for landing page 'Sign in' link...")
            signin_selectors = [
                "a.nav__button-secondary",
                "a.authwall-base-card__sign-in-btn",
                "a:has-text('Sign in')",
                "button:has-text('Sign in')",
                "[data-tracking-control-name='guest_homepage-basic_nav-header-signin']"
            ]
            
            for s_sel in signin_selectors:
                try:
                    signin_link = page.locator(s_sel).first
                    if await signin_link.is_visible(timeout=1000):
                        print(f"[LinkedIn] Clicking sign-in link: {s_sel}")
                        await signin_link.click()
                        await asyncio.sleep(3)
                        break
                except:
                    continue
            
            # Re-check for username after clicking sign-in
            for sel in username_selectors:
                try:
                    if await page.locator(sel).first.is_visible(timeout=3000):
                        current_selector = sel
                        break
                except:
                    continue

        if not current_selector or not credentials:
            print("[LinkedIn] CRITICAL: Could not find login fields. Page state: " + page.url)
            return False

        print("[LinkedIn] Entering credentials...")
        await page.locator(current_selector).first.fill(credentials.get("email", ""))
        
        # Selectors for password - use name since IDs are missing
        password_selectors = [
            "#password",
            "input[name='session_password']",
            "input[type='password']",
            "input[id*=':r1:']",
            "input[id*=':r4:']"
        ]
        current_pass_selector = None
        for p_sel in password_selectors:
            try:
                if await page.locator(p_sel).first.is_visible(timeout=2000):
                    current_pass_selector = p_sel
                    break
            except:
                continue
        
        if current_pass_selector:
            await page.locator(current_pass_selector).first.fill(credentials.get("password", ""))
        else:
            print("[LinkedIn] Could not find password field. Page state: " + page.url)
            return False
        
        # Strategy 1: Enter on password field
        await page.keyboard.press("Enter")
        await asyncio.sleep(3)
        
        # Strategy 2: Explicit 'Sign in' button click if still on login page
        if LINKEDIN_LOGIN_URL in page.url or "login" in page.url:
            print("[LinkedIn] Still on login page after Enter. Trying explicit button click...")
            submit_selectors = [
                "button[type='submit']",
                "button.btn__primary--large",
                "#login-submit",
                "button:has-text('Sign in')",
                ".login__form_action_container button",
                "[data-litms-control-id='login-submit']"
            ]
            for s_btn in submit_selectors:
                try:
                    btn = page.locator(s_btn).first
                    if await btn.is_visible(timeout=2000):
                        print(f"[LinkedIn] Clicking submit button: {s_btn}")
                        await btn.click()
                        await asyncio.sleep(5)
                        break
                except:
                    continue
        else:
            await asyncio.sleep(2)
        
        # Check for 2FA or successful login
        if "checkpoint" in page.url:
            print("[LinkedIn] 2FA/Checkpoint detected. Please complete it in the browser.")
            # Wait for user to bypass
            await page.wait_for_url("**/feed/**", timeout=120000)

        return "feed" in page.url or await page.locator(".feed-identity-module").is_visible(timeout=5000)
    except Exception as e:
        print(f"[LinkedIn] Login failed: {e}")
        return False


def _extract_job_id_from_url(url: str) -> Optional[str]:
    """
    Extracts a LinkedIn job ID from any LinkedIn job URL format.
    Handles /jobs/view/ID and /jobs/search/?currentJobId=ID patterns.
    """
    import re
    from urllib.parse import urlparse, parse_qs

    # Pattern 1: /jobs/view/TITLE-ID/ (most common direct link)
    m = re.search(r'/jobs/view/(?:[^/]+-)?(\d+)', url)
    if m:
        return m.group(1)

    # Pattern 2: /jobs/search/?currentJobId=ID
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if 'currentJobId' in qs:
        return qs['currentJobId'][0]

    return None


def _build_job_view_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs/view/{job_id}/"


async def handle_linkedin(page: Page, job_url: str, candidate_data: dict, credentials: Optional[dict] = None) -> dict:
    """Entry point for LinkedIn job application."""
    if credentials:
        logged_in = await _login(page, credentials)
        if not logged_in:
            return {
                "status": "error",
                "message": "LinkedIn login failed. Check credentials.",
                "filled_fields": [],
            }

    # --- Resolve canonical /jobs/view/ URL ---
    job_id = _extract_job_id_from_url(job_url)
    if job_id:
        canonical_url = _build_job_view_url(job_id)
        if canonical_url != job_url:
            print(f"[LinkedIn] Resolved canonical job URL: {canonical_url} (from {job_url})")
            job_url = canonical_url
    else:
        print(f"[LinkedIn] Could not extract job ID from URL. Using as-is: {job_url}")

    # --- Route Interception: Block aggressive collections redirect ---
    async def block_collections(route):
        url = route.request.url.lower()
        if "/jobs/collections" in url or "/jobs/search" in url:
            print(f"[LinkedIn] Intercepted and blocked redirect/fetch to: {url}")
            await route.abort("aborted")
        else:
            await route.continue_()
            
    try:
        await page.route("**/*", block_collections)
    except Exception as e:
        print(f"[LinkedIn] Warning: Could not set route interception: {e}")

    # --- Robust navigation with redirect recovery (up to 3 attempts) ---
    for attempt in range(3):
        print(f"[LinkedIn] Navigating to {job_url} (Attempt {attempt+1}/3)")
        await page.goto(job_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        
        current = page.url.lower()

        # Already on the right page
        if job_url.split('?')[0].rstrip('/').split('/')[-1] in current:
            print("[LinkedIn] Successfully on job view page.")
            break

        # Redirect to /jobs/search/?currentJobId= — force back to /jobs/view/
        if "/jobs/search/" in current and "currentjobid=" in current:
            extracted = _extract_job_id_from_url(page.url)
            if extracted:
                print(f"[LinkedIn] Search-page redirect detected! Forcing /jobs/view/{extracted}/")
                job_url = _build_job_view_url(extracted)
                await asyncio.sleep(1)
                continue
            else:
                print("[LinkedIn] Search-page redirect but couldn't extract ID. Retrying with original URL.")
                continue

        # Redirect to /jobs/collections/
        if "/jobs/collections" in current and "view" not in current:
            print("[LinkedIn] Collections redirect detected. Retrying...")
            continue

        # If we got somewhere reasonable, stop
        break

    # Final safety: if still on a search/collections page, try one raw goto
    current = page.url.lower()
    if "/jobs/search/" in current or ("/jobs/collections" in current and "view" not in current):
        print("[LinkedIn] ALERT: Still on wrong page after retries. Forcing one last goto...")
        await page.goto(job_url, wait_until="commit")
        await asyncio.sleep(4)

    print(f"[LinkedIn] Final URL after navigation: {page.url}")

    # 0. Check for "No longer accepting responses"
    closed_check = await page.evaluate("""
        () => {
            const forbidden = ["no longer accepting responses", "job is closed", "no longer accepting applications"];
            const text = document.body.innerText.toLowerCase();
            return forbidden.some(msg => text.includes(msg));
        }
    """)
    if closed_check:
        print("[LinkedIn] Job is closed or no longer accepting responses.")
        return {
            "status": "closed",
            "message": "This job is no longer accepting responses on LinkedIn.",
            "filled_fields": [],
        }

    # 1. Wait for the job detail panel to fully load (key: right-panel content)
    await asyncio.sleep(2)
    try:
        await page.wait_for_selector(
            ".jobs-unified-top-card, .jobs-details__main-content, .job-details-jobs-unified-top-card__job-title",
            timeout=8000
        )
        print("[LinkedIn] Job detail panel loaded.")
    except:
        print("[LinkedIn] Job detail panel not found, proceeding anyway...")

    # 2. Use JavaScript to find the Apply / Easy Apply button in the detail panel
    apply_logic_js = """
    (targets) => {
        const exclusions = ["next", "similar", "related", "back", "previous", "dismiss", "close", "save"];

        // Prioritise inside the job-details panel (right panel on search page)
        const panels = [
            document.querySelector('.jobs-details__main-content'),
            document.querySelector('.jobs-unified-top-card'),
            document.querySelector('.jobs-s-apply'),
            document.querySelector('[data-job-id]'),
            document.body
        ].filter(Boolean);

        for (const panel of panels) {
            const elements = Array.from(panel.querySelectorAll('button, a, [role="button"], #jobs-apply-button-id, .jobs-apply-button'));

            for (const target of targets) {
                const found = elements.find(el => {
                    const text = (el.innerText || "").toLowerCase();
                    const aria = (el.getAttribute('aria-label') || "").toLowerCase();
                    const id = (el.id || "").toLowerCase();
                    const cls = (el.className || "").toLowerCase();

                    if (exclusions.some(exc => text.includes(exc) || aria.includes(exc))) return false;

                    const matchesText = text.includes(target.toLowerCase());
                    const matchesAria = aria.includes(target.toLowerCase());
                    const matchesId = target.toLowerCase() === "apply" && id.includes("apply") && id.includes("button");
                    const matchesCls = cls.includes("jobs-apply-button");

                    return (matchesText || matchesAria || matchesId || matchesCls) &&
                           (el.offsetParent !== null || el.getClientRects().length > 0) &&
                           !el.disabled;
                });

                if (found) {
                    found.scrollIntoView({ block: "center" });
                    found.setAttribute('data-antigravity-target', 'true');
                    const isEasy = (found.innerText || "").toLowerCase().includes('easy') ||
                                   (found.getAttribute('aria-label') || "").toLowerCase().includes('easy') ||
                                   (found.className || "").toLowerCase().includes('easy');
                    return { found: true, text: found.innerText || target, isExternal: !isEasy };
                }
            }
        }
        return { found: false };
    }
    """

    detection = await page.evaluate(apply_logic_js, ["Easy Apply", "Apply"])

    if detection["found"]:
        apply_btn = page.locator("[data-antigravity-target='true']").first
        initial_page_count = len(page.context.pages)
        initial_url = page.url
        
        print(f"[LinkedIn] Click target found: {detection['text']}")
        
        # Unified robust click
        await robust_click(page, apply_btn)

        # Remove marker ASAP
        try:
            await page.evaluate("() => document.querySelector('[data-antigravity-target=\"true\"]')?.removeAttribute('data-antigravity-target')")
        except: pass

        # 1. Check for New Tab (External Apply Redirect)
        if len(page.context.pages) > initial_page_count:
            print("[LinkedIn] New tab detected. Delegating to handle_generic for external form.")
            new_page = [p for p in page.context.pages if p != page][-1]
            await new_page.wait_for_load_state("domcontentloaded")
            
            from automation.sites.generic import handle_generic
            result = await handle_generic(new_page, new_page.url, candidate_data)
            return {
                "status": "external_" + result.get("status", "partial"),
                "message": f"Redirected to external site: {new_page.url}. " + result.get("message", ""),
                "filled_fields": result.get("filled_fields", [])
            }

        # 2. Check for Redirect Recovery (Handled within robust_click now)
        if "/jobs/collections" in page.url.lower() and "view" not in page.url.lower():
             return {
                "status": "error",
                "message": "LinkedIn keeps redirecting away from the job page. Please apply manually.",
                "filled_fields": []
             }

        # 3. Check for Modal (Easy Apply)
        modal_open = await page.locator(".jobs-easy-apply-modal, .artdeco-modal").is_visible(timeout=3000)
        if modal_open:
            print("[LinkedIn] Easy Apply modal opened successfully.")
            return await _handle_easy_apply(page, candidate_data)
            
        # 4. Fallback: Check for "Already Applied" or Success Toast
        success_toast = await page.locator(".artdeco-inline-feedback--success").is_visible(timeout=1000)
        if success_toast:
            print("[LinkedIn] Success toast detected immediately after click.")
            return {"status": "completed", "message": "LinkedIn signals application already sent!", "filled_fields": []}

        # 5. Last Resort JS Click
        print("[LinkedIn] ALERT: No modal or tab detected. Trying last-resort JS click...")
        await page.evaluate("Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Apply'))?.click()")
        await asyncio.sleep(3)
        modal_open = await page.locator(".jobs-easy-apply-modal, .artdeco-modal").is_visible(timeout=2000)
        if modal_open:
            return await _handle_easy_apply(page, candidate_data)

        return {
            "status": "needs_manual",
            "message": "Clicked the button, but no modal or redirect occurred. Please check the browser.",
            "filled_fields": []
        }
    else:
        print("[LinkedIn] JS detection failed to find any Apply button.")
        return {
            "status": "needs_manual",
            "message": "No 'Apply' or 'Easy Apply' button found on this page.",
            "filled_fields": [],
        }

async def _handle_easy_apply(page: Page, candidate_data: dict) -> dict:
    """Internal helper to handle the multi-step Easy Apply modal."""
    
    # 1. Wait for the Easy Apply modal to appear
    try:
        modal_selector = ".jobs-easy-apply-modal, [role='dialog'].artdeco-modal"
        await page.wait_for_selector(modal_selector, timeout=15000)
        print("[LinkedIn] Modal detected. Waiting for content to mount...")
        await asyncio.sleep(1.5) 
    except Exception:
        print("[LinkedIn] Warning: Specific modal selector not found. Attempting broad search.")
    
    # Multi-step filling loop
    all_filled = []
    
    for step in range(25):
        print(f"\n[LinkedIn] Processing step {step+1}...")
        
        # 2. Deep Error Scanning
        errors = await page.evaluate("""
            () => {
                const errs = Array.from(document.querySelectorAll('.artdeco-inline-feedback--error, .fb-dash-form-element__error-field, [role="alert"], [aria-invalid="true"]'));
                return errs.map(e => e.innerText || "Validation error").filter(t => t.trim().length > 0);
            }
        """)
        if errors:
            print(f"[LinkedIn] CRITICAL: Form validation errors: {errors}")
        
        # 3. Fill fields on current page
        step_filled = await fill_form_on_page(page, candidate_data)
        all_filled.extend(step_filled)
        
        # 4. Identification of Navigation Buttons
        next_btn = page.locator("button[aria-label*='next'], button[aria-label*='Continue'], button:has-text('Next')").first
        review_btn = page.locator("button[aria-label*='Review'], button:has-text('Review')").first
        submit_btn = page.locator("button[aria-label*='Submit application'], button:has-text('Submit application')").first
        
        target_btn = None
        is_final_step = False
        
        if await submit_btn.is_visible(timeout=500):
            print("[LinkedIn] SUCCESS: Final 'Submit' screen reached. Stopping for user review.")
            is_final_step = True
        elif await review_btn.is_visible(timeout=500):
            target_btn = review_btn
            print("[LinkedIn] 'Review' button found. This is likely the penultimate step.")
        elif await next_btn.is_visible(timeout=500):
            target_btn = next_btn
            print("[LinkedIn] 'Next' button found.")
        
        if target_btn and not is_final_step:
            current_step_info = await page.evaluate("""() => document.querySelector('.jobs-easy-apply-modal')?.innerText.split('\\n').find(t => t.includes('Step')) || 'unknown'""")
            print(f"[LinkedIn] Current progress: {current_step_info}")
            
            print(f"[LinkedIn] Clicking navigation button...")
            await robust_click(page, target_btn)
            await asyncio.sleep(2)
        elif is_final_step:
            break
        else:
            print("[LinkedIn] No navigation buttons visible. Might be at the end or stuck.")
            break

    # 5. Success Monitoring
    print("[LinkedIn] >> MONITORING FOR SUCCESS SCREEN. Please review and click 'Submit' manually. <<")
    success_detected = False
    for attempt in range(60): 
        is_success = await page.evaluate("""
            () => {
                const text = document.body.innerText.toLowerCase();
                const successMsg = ["application sent", "successfully", "sent to", "done", "great!", "submitted"];
                const hasText = successMsg.some(m => text.includes(m));
                const hasFeedback = document.querySelector(".artdeco-inline-feedback--success, .jobs-post-apply-collection") !== null || 
                                   Array.from(document.querySelectorAll(".artdeco-modal__header")).some(h => h.innerText.includes("Application Sent"));
                return hasText || hasFeedback;
            }
        """)
        if is_success:
            print("\n" + "*"*50)
            print("[LinkedIn] VERIFIED SUCCESS: Application has been sent!")
            print("*"*50 + "\n")
            success_detected = True
            break
        await asyncio.sleep(5)

    status = "completed" if success_detected else "partial"
    message = "Verified Success!" if success_detected else "Form filled. Please check for errors and click Submit."
    
    if not all_filled:
        message = "Agent couldn't find many fields to fill. Please check the form manually."

    return {
        "status": status,
        "message": message,
        "filled_fields": list(set(all_filled)),
    }

async def fill_form_on_page(page: Page, candidate: dict) -> list:
    """Intelligent fill for text inputs, radios, and selects on the current page."""
    filled = []
    
    kb = candidate.get("knowledge_base", {})
    ai = candidate.get("additional_info", {})
    
    name = kb.get("name", "") or candidate.get("name", "")
    name_parts = name.split() if name else []
    
    field_map = {
        "first name": name_parts[0] if name_parts else "",
        "last name": name_parts[-1] if len(name_parts) > 1 else "",
        "first": name_parts[0] if name_parts else "",
        "last": name_parts[-1] if len(name_parts) > 1 else "",
        "name": name,
        "email": kb.get("email", "") or candidate.get("email", ""),
        "phone": kb.get("phone", ""),
        "mobile": kb.get("phone", ""),
        "salary": kb.get("expected_salary", ""),
        "ctc": kb.get("expected_salary", ""),
        "experience": str(kb.get("experience_years", "0")),
        "years": str(kb.get("experience_years", "0")),
        "notice": kb.get("notice_period", ""),
        "city": kb.get("location", ""),
        "location": kb.get("location", ""),
        "address": kb.get("location", ""),
        "linkedin": ai.get("linkedin", ""),
        "github": ai.get("github", ""),
        "portfolio": ai.get("website", ""),
        "website": ai.get("website", ""),
        "headline": kb.get("title", ""),
        "summary": kb.get("summary", ""),
        "objective": kb.get("summary", ""),
        "visa": "No", # Defaulting to 'No' for "Do you require sponsorship"
        "sponsorship": "No",
        "authorized": "Yes",
        "work authority": "Yes",
        "citizen": "Yes",
        "legally": "Yes",
    }
    
    form_elements = await page.locator(".fb-dash-form-element, .jobs-easy-apply-form-section__grouping, .artdeco-text-input--container").all()
    if not form_elements:
        form_elements = await page.locator("div:has(label):has(input, select, textarea)").all()
    
    for element in form_elements:
        try:
            label_el = element.locator("label").first
            if not await label_el.count():
                label_el = element.locator("[for], .fb-dash-form-element__label").first
            if not await label_el.count():
                continue
            
            label = await label_el.inner_text(timeout=500)
            label = label.lower().strip()
            
            target_value = None
            for key, val in field_map.items():
                if key in label and val:
                    target_value = str(val)
                    break
            
            if not target_value:
                for k, v in ai.items():
                    if k.lower() in label and v:
                        target_value = str(v)
                        break

            if not target_value:
                continue

            input_box = element.locator("input[type='text'], input[type='tel'], input[type='number'], input[type='email'], input:not([type]), textarea")
            if await input_box.count() > 0:
                input_el = input_box.first
                try:
                    current = await input_el.input_value(timeout=300)
                except:
                    current = ""
                if not current:
                    await input_el.clear()
                    await input_el.fill(target_value)
                    filled.append(label)
                    continue

            binary_inputs = await element.locator("input[type='radio']").all()
            if binary_inputs:
                is_positive = any(w in target_value.lower() for w in ["yes", "have", "did", "am", "currently", "true", "1", "authorized"])
                is_negative = any(w in target_value.lower() for w in ["no", "false", "0", "none", "require"])
                target_choice = "yes" if is_positive else ("no" if is_negative else None)
                
                if target_choice:
                    for bi in binary_inputs:
                        bi_id = await bi.get_attribute("id")
                        if not bi_id: continue
                        bi_label_el = page.locator(f"label[for='{bi_id}']")
                        if not await bi_label_el.count(): continue
                        bi_label_text = await bi_label_el.inner_text()
                        
                        # Handle "Yes/No" vs "I am/I am not"
                        low_lbl = bi_label_text.lower()
                        if (target_choice == "yes" and ("yes" in low_lbl or "i have" in low_lbl or "i am" in low_lbl or "authorized" in low_lbl)) or \
                           (target_choice == "no" and ("no" in low_lbl or "i don't" in low_lbl or "not" in low_lbl or "require" in low_lbl)):
                            if not await bi.is_checked():
                                await bi.click()
                            filled.append(label)
                            break
                continue

            checkbox_inputs = await element.locator("input[type='checkbox']").all()
            if checkbox_inputs and target_value.lower() in ["yes", "true", "1"]:
                for cb in checkbox_inputs:
                    if not await cb.is_checked():
                        await cb.click()
                filled.append(label)
                continue

            select = element.locator("select")
            if await select.count() > 0:
                options = await select.locator("option").all()
                matched = False
                for opt in options:
                    txt = await opt.inner_text()
                    if txt.strip().lower() in ["select", "please select", "choose", ""]:
                        continue
                    if target_value.lower() in txt.lower() or txt.lower() in target_value.lower():
                        val_attr = await opt.get_attribute("value")
                        if val_attr:
                            await select.select_option(value=val_attr)
                            filled.append(label)
                            matched = True
                            break
                if not matched and options:
                    for opt in options:
                        txt = await opt.inner_text()
                        if txt.strip().lower() not in ["select", "please select", "choose", ""]:
                            val_attr = await opt.get_attribute("value")
                            if val_attr:
                                await select.select_option(value=val_attr)
                                filled.append(f"{label} (auto-selected)")
                                break
                continue

        except Exception:
            continue
            
    return filled
