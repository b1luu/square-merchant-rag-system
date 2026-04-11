"""Mosa Employee Handbook.pdf → policy and POS procedure records (topic splits)."""

from __future__ import annotations

from mosa_rag.normalize import normalize_for_parse
from mosa_rag.pdf_text import PageText, join_pages
from mosa_rag.schema import Record


def _must_contain(blob: str, needle: str) -> None:
    if needle not in blob:
        raise ValueError(f"Expected handbook PDF text to contain: {needle!r}")


def extract_handbook_records(pages: list[PageText], source_file: str) -> list[Record]:
    blob = normalize_for_parse(join_pages(pages))
    for n in ["Expectations", "How to Use Square", "Hungry Panda", "Dress Code"]:
        _must_contain(blob, n)

    records: list[Record] = []

    def add(
        *,
        rtype: str,
        title: str,
        page: int,
        entity: str,
        rt: str,
        role_scope: list[str] | None = None,
        doc_type: str = "handbook",
        tags: list[str] | None = None,
        rules: list[str] | None = None,
        steps: list[str] | None = None,
        storage_life: str = "",
        ingredients: list[str] | None = None,
    ) -> None:
        records.append(
            Record(
                id="",
                type=rtype,
                title=title,
                entity_name=entity,
                doc_type=doc_type,
                role_scope=role_scope or ["all_staff"],
                rules=rules or [],
                steps=steps or [],
                tags=tags or [],
                ingredients=ingredients or [],
                storage_life=storage_life,
                source_file=source_file,
                source_page=page,
                retrieval_text=rt,
            )
        )

    # --- Expectations ---
    add(
        rtype="policy_rule",
        title="Front-of-house employee expectations",
        page=1,
        entity="FOH responsibilities",
        role_scope=["bar", "front_of_house"],
        tags=["expectations", "FOH"],
        rt=(
            "Front-of-house expectations at Mosa Tea: open and close the bar and front areas (see opening/closing checklists); "
            "keep the front and bathroom clean all day; take orders and prepare drinks; monitor stock and decide whether to prep more "
            "or mark items sold out for the day; support the kitchen and restock during downtime."
        ),
    )
    add(
        rtype="policy_rule",
        title="Kitchen employee expectations",
        page=1,
        entity="Kitchen responsibilities",
        role_scope=["kitchen"],
        tags=["expectations", "kitchen"],
        rt=(
            "Kitchen expectations at Mosa Tea: prepare daily ingredients (teas, cream foams, toppings); keep the workspace clean—wash equipment after use, wipe surfaces, "
            "clean the boba sink with Simple Green daily after use; verify ingredients are ready for the next opening shift (pistachio paste, roasted pistachios, Genmai, Buckwheat Barley, "
            "toppings like tea jelly and Hun Kue, and cut fruits including apple, orange, lemon, strawberry, mango); support the front during rushes."
        ),
    )
    add(
        rtype="policy_rule",
        title="Shared teamwork expectations across roles",
        page=2,
        entity="Shared responsibilities",
        tags=["teamwork", "communication"],
        rt=(
            "Employees share responsibilities with their shift partner: if you are on front, check in with kitchen (and vice versa) on how to help. "
            "Additional downtime guidance is referenced under “What to do if you have down time.”"
        ),
    )

    # --- First day / onboarding ---
    add(
        rtype="policy_rule",
        title="First day onboarding — uniform, locker, clock-in",
        page=2,
        entity="Onboarding basics",
        tags=["onboarding", "uniform", "timekeeping"],
        rules=[
            "Uniform is stored in the employee locker; stay in full uniform during the shift.",
            "Personal items may be stored in the locker; you may bring your own lock.",
            "Clock in using Square POS; you may clock in up to 5 minutes before the scheduled start.",
        ],
        rt=(
            "First day at Mosa: find your uniform in the employee locker and wear full uniform on shift. You may store personal items in lockers and may use your own lock. "
            "Clock in on the Square POS (may clock in up to five minutes early). Training is supported by the teammate working with you."
        ),
    )

    # --- Where to find items ---
    add(
        rtype="policy_rule",
        title="Where to find front supplies (front cabinets)",
        page=3,
        entity="Front cabinet inventory map",
        tags=["storage", "supplies", "front"],
        rt=(
            "Front cabinets contain receipt/ticket rolls; cups, sample cups, lids; bags; straws; merchandise, business cards, tape; napkins and hand towels. "
            "If supplies are almost out, notify Dennis or Anna."
        ),
    )
    add(
        rtype="policy_rule",
        title="Where to find back-storage supplies",
        page=4,
        entity="Back storage map",
        tags=["storage", "supplies", "back"],
        rt=(
            "Back storage guidance: boba, lychee jelly, brown sugar, creamer appear in Possimei white packages; fruit syrups, sugar, matcha powder, and grapefruit cans are label-driven; "
            "cups are in boxes marked U-600 PP; straws are 6mm (thin) vs 12mm (thick) per box labeling; cup sealer location per storage layout."
        ),
    )
    add(
        rtype="policy_rule",
        title="4-cup trays — unboxing and shelf stocking",
        page=5,
        entity="Tray boxes",
        tags=["storage", "trays"],
        rt=(
            "4-cup trays arrive in box cubes by the wall. After opening a box, remove remaining trays to the shelf. "
            "By end of closing, remove empty boxes and take them out with trash."
        ),
    )

    # --- Square POS (split procedures) ---
    add(
        rtype="pos_procedure",
        title="Square POS — find your 4-digit POS passcode",
        page=5,
        entity="Square passcode lookup",
        role_scope=["all_staff"],
        tags=["square", "POS", "passcode"],
        steps=[
            "Open Square Team app → Me (bottom).",
            "Open Account under My Info.",
            "Your POS passcode appears at the bottom.",
        ],
        rt=(
            "How to find your Square POS passcode: in the Square Team app, tap Me, then Account under My Info; the passcode is listed at the bottom."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square POS — clock in, clock out, and breaks",
        page=5,
        entity="Square time clock",
        tags=["square", "time_clock", "breaks"],
        steps=[
            "On the Square tablet, open View Time Clock.",
            "Enter your POS passcode.",
            "Choose clock in or clock out.",
            "For breaks: use Start break / End break when applicable.",
        ],
        rt=(
            "Square time clock procedure: on the Square tablet choose View Time Clock, enter your POS passcode, then clock in/out. "
            "Breaks can be tracked with start/end break actions."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square Team — view schedule and filter to my shifts",
        page=6,
        entity="Square schedule",
        tags=["square", "schedule"],
        steps=[
            "Open Square Team app → Schedule (bottom).",
            "Use the top-right menu → My shifts only to filter.",
        ],
        rt=(
            "Viewing shifts: Square Team app → Schedule shows who you work with. Use the menu’s “my shifts only” filter to see just your shifts."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square Team — request time off",
        page=6,
        entity="Time off requests",
        tags=["square", "time_off"],
        steps=[
            "Square Team → Me → Time Off.",
            "Enter days you cannot work if not already reflected in availability.",
        ],
        rules=["Time off requests are approved by Anna."],
        rt=(
            "Requesting time off in Square Team: Me → Time Off; enter unavailable days. Requests are ultimately approved by Anna."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square Team — update weekly availability",
        page=6,
        entity="Availability updates",
        tags=["square", "availability"],
        steps=["Square Team → Me → Availability; set each weekday."],
        rules=["Availability repeats weekly; update it whenever your ongoing schedule needs to change."],
        rt=(
            "Changing availability: Square Team → Me → Availability, set each day. This pattern repeats weekly—update it if your standing schedule changes."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square Team — shift cover and shift trade",
        page=6,
        entity="Shift swaps",
        tags=["square", "shift_swap"],
        steps=[
            "Under Schedule, open a shift → Request shift cover (open offer) OR Trade shift (specific teammate).",
        ],
        rt=(
            "Shift coverage: from Schedule, request shift cover to offer a shift broadly, or trade shifts with a specific coworker."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Square Team — correct clock-out time and add tips (retro edit)",
        page=6,
        entity="Hours corrections",
        tags=["square", "hours", "tips"],
        steps=[
            "Square Team → Me → Hours Worked.",
            "Select the day; adjust clock-out time and add tips if needed.",
            "Submit request changes with a reason (for example forgot to add tips).",
        ],
        rt=(
            "Retroactive time corrections: Square Team → Me → Hours Worked → pick day → adjust clock-out and tips → request changes with a reason."
        ),
    )

    # --- Third-party ordering ---
    add(
        rtype="pos_procedure",
        title="Hungry Panda — enter order naming and payment workflow",
        page=6,
        entity="Hungry Panda workflow",
        tags=["hungry_panda", "third_party", "POS"],
        steps=[
            "Accept the Hungry Panda order; receipt prints.",
            "Enter the same order on the POS named HP + last 4 digits of the receipt (example: HP 1234).",
            "Pay in cash so tickets print, then refund the order with Hungry Panda as the refund reason (note HP/Panda).",
            "Use printed tickets to make drinks.",
        ],
        rt=(
            "Hungry Panda procedure: accept the order, then mirror it on the POS as HP + last four digits of the receipt. "
            "Use cash pay to print tickets, refund with a Hungry Panda reason, then fulfill using printed tickets."
        ),
    )
    add(
        rtype="pos_procedure",
        title="Mark items sold out in Square (include end time); toppings end-of-day restock",
        page=7,
        entity="Sellouts in POS",
        tags=["sellout", "availability"],
        steps=[
            "POS → Availability → select item → Available → switch to Unavailable and set an end time (often end of day).",
            "For toppings like boba or cream foams, return availability at end of day when restocked.",
        ],
        rules=["Marketplace orders: also mark sold out separately on the Hungry Panda app when needed."],
        rt=(
            "Sold-out workflow: in Square Availability, set items unavailable with an end time. Toppings often need to be flipped back in stock at end of day. "
            "Hungry Panda may require separate sold-out marking on its app. DoorDash/Uber Eats/Grubhub/Square Online orders are handled differently—follow current third-party guidance in-store."
        ),
    )

    # --- Dress / hygiene / safety ---
    add(
        rtype="policy_rule",
        title="Dress code and personal presentation",
        page=7,
        entity="Dress code",
        tags=["dress_code", "uniform"],
        rules=[
            "Wear cap and apron at all times.",
            "Closed-toe shoes only (no sandals/flip-flops).",
            "Nails clean; no nail polish; acrylic/spa nails require gloves always.",
            "Long hair tied back.",
        ],
        rt=(
            "Dress code: cap and apron always; closed-toe shoes; hygiene maintained; no nail polish; acrylic/spa nails require gloves; long hair tied back."
        ),
    )
    add(
        rtype="policy_rule",
        title="Hygiene — handwashing and illness policy",
        page=7,
        entity="Hygiene policy",
        tags=["hygiene", "illness"],
        rules=[
            "Wash hands with soap and dry with paper towels before drink making; rewash after restroom use before returning to bar/kitchen.",
            "Use PPE (gloves) when needed.",
            "Employees with contagious illness must take leave and notify management if unwell.",
        ],
        rt=(
            "Hygiene expectations: proper handwashing before drink prep; after restroom visits wash again before returning to food areas; use gloves when needed; "
            "do not work sick with contagious illnesses—notify management."
        ),
    )
    add(
        rtype="policy_rule",
        title="Food safety — equipment use and restricted areas",
        page=8,
        entity="Food safety basics",
        tags=["food_safety", "health"],
        rules=[
            "Follow local health regulations and Mosa policies.",
            "Report unsafe conditions immediately (example: electrical hazard).",
            "Store equipment is only for drink preparation.",
            "Only active employees may be in kitchen, bar, and back storage areas.",
            "Outside food/drinks are not allowed in kitchen/bar; store personal items in back storage for breaks; do not eat in kitchen/bar areas.",
        ],
        rt=(
            "Food safety: comply with local health rules; report hazards; use tools only for drink prep; restrict kitchen/bar/back storage to active employees; "
            "keep outside food and eating out of kitchen/bar operational zones."
        ),
    )
    add(
        rtype="policy_rule",
        title="Equipment operation and liability reminder",
        page=8,
        entity="Equipment safety",
        tags=["equipment", "safety"],
        rt=(
            "Operate equipment per instructions. Mosa Tea disclaims liability for injuries from misuse, negligence, or misconduct."
        ),
    )

    # --- Customer service ---
    add(
        rtype="policy_rule",
        title="Customer service — remakes and escalation",
        page=8,
        entity="Remake policy",
        tags=["customer_service"],
        rules=[
            "If a customer requests a remake, ask politely for the reason and remake as appropriate.",
            "Report unusual or questionable situations to Anna.",
        ],
        rt=(
            "Remakes: ask politely why the guest wants a remake, then remake using good judgment; escalate odd situations to Anna."
        ),
    )
    add(
        rtype="policy_rule",
        title="Valet parking coordination (weekend nights)",
        page=8,
        entity="Valet policy",
        tags=["valet", "parking"],
        rules=[
            "Godfather Restaurant valet is usually present Friday and Saturday nights.",
            "You may move the cone as needed.",
            "For hesitant new valet staff, offer a milk tea and remind them customers may briefly move the cone to park and should replace it after pickup.",
        ],
        rt=(
            "Valet coordination: weekend nights often have valet; cones may be moved for flow; if valet staff hesitate, offer a milk tea and clarify the parking agreement and cone etiquette."
        ),
    )

    # --- Teamwork + general expectations ---
    add(
        rtype="policy_rule",
        title="Team communication — lateness, handoffs, group chat announcements",
        page=9,
        entity="Team communication norms",
        tags=["teamwork", "communication"],
        rules=[
            "Notify your shift partner if you will be more than 5 minutes late.",
            "Share locations, low stock, and issues before shift handoff.",
            "Read group chat announcements within the first 10 minutes of your shift.",
        ],
        rt=(
            "Team norms: notify partner if late >5 minutes; communicate stock/location/issues at handoffs; read group chat announcements early in the shift."
        ),
    )
    add(
        rtype="policy_rule",
        title="General expectations — punctuality, parking, scheduling discipline",
        page=9,
        entity="Scheduling expectations",
        tags=["scheduling", "punctuality"],
        rules=[
            "Arrive on time; park behind the plaza when possible.",
            "Update availability weekly by Sunday for the upcoming week.",
            "Work published shifts unless exception (emergency/doctor’s note).",
        ],
        rt=(
            "Scheduling: be on time; update availability weekly by Sunday; work assigned shifts or find coverage except for emergencies/documentation."
        ),
    )
    add(
        rtype="policy_rule",
        title="California break rules (summary) + Square break clocking",
        page=9,
        entity="Meal and rest breaks",
        tags=["labor_law", "breaks", "california"],
        rules=[
            "Shifts over 6 hours: one unpaid 30-minute meal break.",
            "Shorter shifts / every ~4 hours: one paid 10-minute rest break (per stated handbook framing).",
            "Clock breaks in Square; breaks are required.",
        ],
        rt=(
            "California break policy summary: employees get one unpaid 30-minute meal break for shifts over 6 hours and one paid 10-minute rest break for every ~4 hours worked. "
            "Employees must clock meal and rest breaks in Square, and breaks are required."
        ),
    )
    add(
        rtype="policy_rule",
        title="Timekeeping, lockers, employee drink benefits",
        page=10,
        entity="Benefits and belongings",
        tags=["benefits", "timekeeping"],
        rules=[
            "Clock in/out every shift.",
            "Lockers in back storage for personal belongings; bring your own lock if desired.",
            "One free drink per shift (same-day), plus 10 discounted drinks/month at 10% off.",
        ],
        rt=(
            "Timekeeping and perks: clock all shifts; store belongings in lockers; employee drink benefit is one free drink per shift (same day) and ten drinks/month at 10% off."
        ),
    )

    # --- Catering / large batch formula ---
    add(
        rtype="batch_prep_recipe",
        title="Catering — large-batch milk tea formula (pitcher + electric whisk)",
        page=10,
        entity="Catering milk tea scaling",
        role_scope=["bar", "kitchen"],
        tags=["catering", "scaling", "milk_tea"],
        rules=[
            "Each pitcher can make about 8–10 drinks (3.5L max).",
            "Catering formula: (300mL tea × N) + (2 scoops creamer × N) + (sugar amount × N), where N is drink count.",
            "100% ice assumption may require tea adjustments based on ice/toppings.",
        ],
        steps=[
            "Example: 8 cups of Four Seasons milk tea, 100% ice, 50% sugar → 2400mL tea + 16 scoops creamer + hit 100% sugar four times into pitcher; whisk and pour to cups.",
        ],
        storage_life="Prep timing relative to pickup; follow overnight tea rules in the catering section.",
        rt=(
            "Catering milk tea batching: scale with N drinks using 300mL tea per drink, 2 scoops creamer per drink, and scaled sugar; mix in pitchers with an electric whisk. "
            "Pitchers hold ~8–10 drinks (3.5L max). Example given for eight 4S milk teas at 100% ice and 50% sugar."
        ),
    )
    add(
        rtype="policy_rule",
        title="Catering — which teas may be overnight vs must be fresh",
        page=11,
        entity="Catering tea freshness rules",
        tags=["catering", "tea_freshness"],
        rules=[
            "Overnight teas may be used for milk teas and fruit teas, and for pure tea at 50%+ sugar.",
            "Pure tea under 50% sugar must use fresh tea made that day.",
        ],
        rt=(
            "Catering tea usage: overnight tea is OK for milk teas, fruit teas, and purer teas at 50% sugar or more; purer teas below 50% sugar require fresh tea made the same day."
        ),
    )
    add(
        rtype="policy_rule",
        title="Catering / events — packing timing and materials",
        page=11,
        entity="Packing checklist",
        tags=["catering", "packing"],
        steps=[
            "Start packing about 1 hour before pickup.",
            "Include straws.",
            "Prefer fruit syrup boxes, lids, and boba/lychee jelly boxes as drink carriers; open boxes if needed.",
        ],
        rt=(
            "Packing catered orders: begin ~1 hour before pickup; include straws; use sturdy boxes when possible."
        ),
    )
    add(
        rtype="policy_rule",
        title="Influencer visit procedure (free drinks)",
        page=10,
        entity="Influencer policy",
        tags=["marketing", "influencer"],
        rules=[
            "You will know the IG account and arrival time in advance.",
            "Provide 5–6 free menu drinks; influencer may choose or you may pick Mosa Signatures.",
        ],
        rt=(
            "Influencers: visits are pre-announced with account and time; offer 5–6 free on-menu drinks chosen by the influencer or by staff from Mosa Signatures."
        ),
    )

    # --- Foodielab / farmers market / Tracy ---
    add(
        rtype="policy_rule",
        title="Foodielab schedule overview (bentos + drink pickups)",
        page=11,
        entity="Foodielab cadence",
        tags=["foodielab", "events"],
        rt=(
            "Foodielab overview: Tuesdays–Wednesdays bentos drop off; Thursdays–Saturdays they pick up 10 drinks for farmers market; see detailed bento and pickup instructions elsewhere in the handbook."
        ),
    )
    add(
        rtype="policy_rule",
        title="Foodielab bento combo — in-store Tuesdays & Wednesdays",
        page=11,
        entity="Foodielab bento procedure",
        tags=["foodielab", "bento"],
        rules=[
            "Place Foodielab poster by the register.",
            "Foodielab delivers prepaid bentos and customer list by 12:30 PM.",
            "Verify customer name on pickup.",
            "Combo is one bento + one drink: black milk tea with boba OR orange green tea / Four Seasons tea (no substitutions).",
            "Extra bentos for sale: customers Zelle Foodielab via QR on poster; first come first served.",
            "Store bentos refrigerated (meat); microwave on request.",
        ],
        rt=(
            "Foodielab bento pickup days: poster placement, verify names from the list, combo drink options are fixed, extra sales via Zelle/QR, refrigerate meat bentos, microwave if requested."
        ),
    )
    add(
        rtype="policy_rule",
        title="Foodielab farmers market drink pickup (Thu–Sat)",
        page=12,
        entity="Farmers market drinks",
        tags=["foodielab", "farmers_market"],
        rules=[
            "Make 10 drinks/day at 100% sugar and 25% ice: 7 boba milk teas and 3 orange green / Four Seasons.",
            "Use leftover tea from prior day if available for these 10 drinks.",
            "Pickup times: Thu 1:00 PM, Fri 1:00 PM, Sat 10:30 AM; if boba is not ready by Sat 10:30, substitute lychee jelly or tea jelly.",
        ],
        rt=(
            "Farmers market drink batch for Foodielab: ten drinks with specified ratios and pickup times; Saturday early pickup allows jelly substitutions if boba is not ready."
        ),
    )
    add(
        rtype="policy_rule",
        title="Tracy (@tracysgarbage) monthly event setup",
        page=12,
        entity="Tracy event",
        tags=["events", "setup"],
        steps=[
            "First Wednesday monthly, 6–9 PM collaboration.",
            "Move two tables from storage (microwave-on-top tables).",
            "Move a gray ottoman/bench to the front for setup.",
        ],
        rt=(
            "Tracy collaboration nights: monthly first Wednesday 6–9pm; relocate specified tables and ottoman/bench for her portrait setup."
        ),
    )

    # --- Troubleshooting ---
    add(
        rtype="policy_rule",
        title="Troubleshooting — kiosk / boba machine power cycling basics",
        page=12,
        entity="Equipment reset basics",
        tags=["troubleshooting", "kiosk"],
        rules=[
            "If kiosk or boba machine is off: shut down first, then plug/unplug.",
            "Kiosk must be on the kiosk stand to work.",
        ],
        rt=(
            "Basic troubleshooting: for kiosk/boba machine power issues, power cycle safely; kiosk must remain on its stand to function."
        ),
    )
    add(
        rtype="policy_rule",
        title="Troubleshooting — sugar machine errors and deep clean cap",
        page=12,
        entity="Sugar machine",
        tags=["troubleshooting", "sugar_machine"],
        steps=[
            "Ensure full refill; toggle warmer/power; unplug briefly while on and retry.",
            "If stuck: remove inner silver cap with gloves, rinse hot, dry with clean paper towel, reinstall; repeat until normal.",
        ],
        rt=(
            "Sugar syrup machine issues: verify refill and power cycles; if jammed, clean the internal silver cap with gloves using hot water, avoiding contamination."
        ),
    )
    add(
        rtype="policy_rule",
        title="Troubleshooting — Square support phone number",
        page=13,
        entity="Square support",
        tags=["square", "support"],
        rules=["Square support: 1-855-700-6000 (as listed in handbook)."],
        rt="If Square hardware/software fails in ways you cannot resolve, call Square support at 1-855-700-6000.",
    )
    add(
        rtype="policy_rule",
        title="Pest sighting response (roach/rat) + notify Anna",
        page=13,
        entity="Pest protocol",
        tags=["pests", "safety"],
        rules=[
            "Kill the pest and bleach-clean the spot.",
            "Place pest control balls in a fabric bag in a corner.",
            "Notify Anna about sightings/infestations.",
        ],
        rt=(
            "Pest protocol: kill/clean with bleach; place pest balls per handbook; escalate to Anna."
        ),
    )
    add(
        rtype="policy_rule",
        title="Health inspector visit — call Anna and milk/cream temperature",
        page=13,
        entity="Health inspection",
        tags=["inspection", "food_safety"],
        rules=[
            "Call Anna immediately if inspectors arrive.",
            "Milk/heavy cream must pass temperature checks (handbook: target 2–4°C); cool in freezer if out of range.",
        ],
        rt=(
            "Health inspections are unannounced: call Anna immediately. Ensure milk/cream are in required cold range (handbook notes 2–4°C); use freezer to recover out-of-range product."
        ),
    )

    # --- Downtime + closing tips ---
    add(
        rtype="policy_rule",
        title="What to do during downtime (front bar)",
        page=13,
        entity="Downtime tasks",
        tags=["downtime", "FOH"],
        steps=[
            "Prep Hungry Panda bags/trays.",
            "Weekend/rush prep: make 3 matchas when appropriate.",
            "Cut fruit as needed (strawberries/mangos).",
            "Refill low syrups; restock sugar, creamer, ice, towels, soap, straws, cups.",
            "Prepare samples; help kitchen; do dishes.",
            "Check open Buckwheat Barley and Genmai in fridge daily: taste for quality; they last ~2–3 days and look/smell bad when spoiled.",
        ],
        rt=(
            "Downtime checklist for front: prep third-party materials, anticipate matcha on weekends, maintain fruit and syrups, restock disposables, support kitchen, monitor Genmai/Buckwheat quality in the fridge."
        ),
    )
    add(
        rtype="closing_checklist_item",
        title="Tips to close earlier — bathroom restock and cold storage",
        page=14,
        entity="Early-close tips",
        tags=["closing", "efficiency"],
        steps=[
            "Refill bathroom supplies during downtime.",
            "Refrigerate syrups and slower-moving items.",
            "Fix the sealer; stage dirty equipment to the back.",
            "Clean kitchen areas including boba sink, fruit cutter, and equipment washing.",
        ],
        rt=(
            "Closing efficiency tips: keep bathroom stocked during downtime; cold-store syrups; maintain sealer; stage washing; deep-clean key stations including boba sink and fruit cutter."
        ),
    )

    return records
