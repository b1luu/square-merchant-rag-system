"""Kitchen opening SOP: checklists, inventory minimums, prep threshold tables."""

from __future__ import annotations

from mosa_rag.normalize import normalize_for_parse
from mosa_rag.pdf_text import PageText, join_pages
from mosa_rag.schema import Record


def extract_kitchen_open_records(pages: list[PageText], source_file: str) -> list[Record]:
    records: list[Record] = []
    # Touch PDF text so we fail fast if the document changes materially
    _ = normalize_for_parse(join_pages(pages))

    # --- Deadline record ---
    deadline = (
        "Kitchen morning shift prep must be finished by weekday 12:00 PM or weekend 11:00 AM. "
        "This deadline applies to the listed morning prep tasks for Mosa Tea kitchen staff."
    )
    records.append(
        Record(
            id="",
            type="policy_rule",
            title="Kitchen morning prep completion deadline",
            entity_name="morning shift cutoff",
            doc_type="sop_kitchen_open",
            role_scope=["kitchen"],
            shift_scope=["morning"],
            day_scope=["weekday", "weekend"],
            time_scope=["by 12:00 weekdays", "by 11:00 weekends"],
            tags=["deadline", "opening", "kitchen"],
            source_file=source_file,
            source_page=1,
            retrieval_text=deadline,
        )
    )

    # --- First half checklist (numbered 1-11) ---
    first_half_items = [
        "Cook 1 batch of boba in the rice cooker.",
        "Brew 1 batch of black tea in the small pot (mix with frozen black tea from the freezer to cool).",
        "Brew 2 batches of TGY in the big pot.",
        "Brew 1 batch of Four Seasons in the small pot.",
        "Brew 1 batch of Green Tea in the small pot.",
        "Make 1 batch of Pistachio Cream Foam.",
        "Make 1 batch of Brown Sugar Cream Foam.",
        "Make 2 batches of Chestnut Cream.",
        "Make 2 batches of Matcha.",
        "Cut an orange, a lemon, and an apple into 2 of each (fruit prep).",
        "Make boba: 1 batch in the pot on weekdays; 2 batches for the weekend.",
    ]
    for i, sentence in enumerate(first_half_items, start=1):
        records.append(
            Record(
                id="",
                type="opening_checklist_item",
                title=f"Morning opening — first half ({i}/11)",
                entity_name=sentence.split(".")[0][:80],
                doc_type="sop_kitchen_open",
                role_scope=["kitchen"],
                shift_scope=["morning"],
                steps=[sentence],
                tags=["opening", "kitchen", "first_half", f"step_{i}"],
                source_file=source_file,
                source_page=1,
                retrieval_text=(
                    f"Kitchen morning opening checklist (first half, item {i} of 11). Applies to kitchen staff on opening shifts. "
                    f"Task: {sentence}"
                ),
            )
        )

    # --- Second half: cut prior-day jelly + inventory minimums intro ---
    records.append(
        Record(
            id="",
            type="opening_checklist_item",
            title="Morning opening — cut prior-day Hun Kue and Tea Jelly",
            entity_name="Hun Kue and Tea Jelly prep",
            doc_type="sop_kitchen_open",
            role_scope=["kitchen"],
            shift_scope=["morning"],
            steps=["Cut Hun Kue and Tea Jelly from the previous day."],
            tags=["opening", "kitchen", "second_half", "jelly"],
            source_file=source_file,
            source_page=1,
            retrieval_text=(
                "Kitchen morning opening checklist (second half): cut Hun Kue and Tea Jelly carried over from the previous day. "
                "Applies to kitchen staff during opening prep after the first-half tasks are completed."
            ),
        )
    )

    inv_lines = [
        ("Genmai pitchers", "Keep at least 2 pitchers of Genmai ready."),
        ("Buckwheat Barley pitchers", "Keep at least 2 pitchers of Buckwheat Barley ready."),
        ("Tea Jelly batches", "Keep at least 3 batches of Tea Jelly (use leftover brewed tea from the day before first)."),
        ("Hun Kue batches", "Keep at least 3 batches of Hun Kue."),
        ("Pistachio paste boxes", "Keep at least 2 boxes of pistachio paste."),
        ("Matcha concentrate bottles", "Keep at least 2 bottles of matcha concentrate."),
        ("Roasted pistachio stock", "Keep at least 1 full box of roasted pistachio."),
    ]
    for name, rule in inv_lines:
        records.append(
            Record(
                id="",
                type="inventory_minimum",
                title=f"Morning inventory minimum — {name}",
                entity_name=name,
                doc_type="sop_kitchen_open",
                role_scope=["kitchen"],
                shift_scope=["morning"],
                rules=[rule],
                tags=["inventory", "minimum", "opening", "kitchen"],
                source_file=source_file,
                source_page=1,
                retrieval_text=(
                    f"Kitchen morning prep inventory minimum for {name}. Applies to kitchen staff while checking inventory during the second half of opening prep. "
                    f"Rule: {rule}"
                ),
            )
        )

    closing_lines = [
        "2 pitchers of Genmai",
        "2 pitchers of Buckwheat Barley",
        "3 batches of Tea Jelly",
        "3 batches of Hun Kue",
        "2–3 boxes of Black Tea (stored in the freezer)",
        "2 bottles of Matcha Concentrate",
        "2 boxes of pistachio paste",
        "1 full box of roasted pistachio",
    ]
    for line in closing_lines:
        records.append(
            Record(
                id="",
                type="closing_checklist_item",
                title=f"Evening prep for next day — {line.split('(')[0].strip()}",
                entity_name=line,
                doc_type="sop_kitchen_open",
                role_scope=["kitchen"],
                shift_scope=["evening", "closing"],
                steps=[f"Before closing, ensure {line} is ready for the next opening shift."],
                tags=["closing", "prep_forward", "kitchen"],
                source_file=source_file,
                source_page=1,
                retrieval_text=(
                    "Kitchen evening/closing prep item: ensure the shop is stocked for tomorrow’s opening shift. "
                    f"Requirement: {line}."
                ),
            )
        )

    # --- Prep threshold tables ---
    # Weekday Mon–Thu
    weekday_rows = [
        ("Boba", "1/2 batch volume", "Make 1 batch", "Make 1/2 batch (last batch at 7:00 PM)"),
        ("Green / Four Seasons", "1/4 batch volume", "Make 1/2 batch", "Make 1/2 batch"),
        ("TGY", "1/4 batch volume", "Make 1 batch", "Make 1/2 batch"),
        ("Cream foams", "1/4 batch volume", "Make 1/2 batch", "Make 1/2 batch"),
    ]
    for item, low, before6, after6 in weekday_rows:
        records.append(
            Record(
                id="",
                type="prep_threshold_rule",
                title=f"Weekday (Mon–Thu) prep threshold — {item}",
                entity_name=item,
                doc_type="sop_kitchen_open",
                role_scope=["kitchen"],
                shift_scope=["mid", "evening"],
                day_scope=["monday", "tuesday", "wednesday", "thursday"],
                time_scope=["before 6:00 PM", "after 6:00 PM", "last batch cutoff 7:00 PM (boba)"],
                threshold=f"Low limit: {low}.",
                action=f"Before 6pm: {before6}. After 6pm: {after6}.",
                rules=[
                    "Only make when items are lower than the limit; amounts/times are rough guidance—adjust based on daily demand.",
                ],
                tags=["prep_threshold", "weekday", "kitchen"],
                source_file=source_file,
                source_page=1,
                retrieval_text=(
                    f"Weekday (Monday–Thursday) kitchen prep threshold rule for {item}. Applies to kitchen staff. "
                    f"When inventory falls to about {low}, prep as follows. Before 6:00 PM: {before6}. After 6:00 PM: {after6}. "
                    f"For boba, the last batch is scheduled at 7:00 PM. These thresholds are guidance—use judgment based on the day."
                ),
            )
        )

    weekend_rows = [
        ("Boba", "1/2 batch volume", "Make 1 batch", "Make 1/2 batch (last batch at 8:00 PM)"),
        ("Green / Four Seasons", "1/4 batch volume", "Make 1 batch", "Make 1/2 batch"),
        ("TGY", "1/4 batch volume", "Make 2 batches", "Make 1 batch"),
        ("Cream foams", "1/4 batch volume", "Make 1 batch", "Make 1/2 batch"),
    ]
    for item, low, before6, after6 in weekend_rows:
        records.append(
            Record(
                id="",
                type="prep_threshold_rule",
                title=f"Weekend (Fri–Sun) prep threshold — {item}",
                entity_name=item,
                doc_type="sop_kitchen_open",
                role_scope=["kitchen"],
                shift_scope=["mid", "evening"],
                day_scope=["friday", "saturday", "sunday"],
                time_scope=["before 6:00 PM", "after 6:00 PM", "last batch cutoff 8:00 PM (boba)"],
                threshold=f"Low limit: {low}.",
                action=f"Before 6pm: {before6}. After 6pm: {after6}.",
                rules=[
                    "Only make when items are lower than the limit; amounts/times are rough guidance—adjust based on daily demand.",
                ],
                tags=["prep_threshold", "weekend", "kitchen"],
                source_file=source_file,
                source_page=1,
                retrieval_text=(
                    f"Weekend (Friday–Sunday) kitchen prep threshold rule for {item}. Applies to kitchen staff. "
                    f"When inventory falls to about {low}, prep as follows. Before 6:00 PM: {before6}. After 6:00 PM: {after6}. "
                    f"For boba, the last batch is scheduled at 8:00 PM on weekends. These thresholds are guidance—use judgment based on the day."
                ),
            )
        )

    return records
