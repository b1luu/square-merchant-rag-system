"""SOP - Kitchen.pdf → batch prep recipes + key kitchen policy notes."""

from __future__ import annotations

from mosa_rag.normalize import normalize_for_parse
from mosa_rag.pdf_text import PageText, join_pages
from mosa_rag.schema import Record


def _must_contain(pdf_blob: str, needle: str) -> None:
    if needle not in pdf_blob:
        raise ValueError(f"Expected PDF text to contain: {needle!r}")


def extract_kitchen_batch_records(pages: list[PageText], source_file: str) -> list[Record]:
    blob = normalize_for_parse(join_pages(pages))
    records: list[Record] = []

    # --- Policy / equipment notes (page 3–4) ---
    policy_chunks: list[tuple[str, str, int, list[str], str]] = [
        (
            "policy_rule",
            "Aluminum pots cannot be heated on the stove",
            3,
            ["aluminum", "equipment", "tea_storage"],
            "Aluminum pots must not be heated on the stove; use them only to store tea. If cooking more than one batch at a time, use the large stainless-steel pot. Applies to kitchen staff preparing brewed tea.",
        ),
        (
            "policy_rule",
            "Label containers with date and time; move date marker when refilling",
            4,
            ["labeling", "FIFO", "food_safety"],
            "Put the date and time on containers when storing prepared ingredients. When refilling a dispenser from a container, transfer the date/time marker to the dispenser. Applies to kitchen prep and storage.",
        ),
        (
            "policy_rule",
            "Cooling Black Tea, Genmai, Barley, and Buckwheat after brewing",
            4,
            ["cooldown", "tea", "temperature"],
            "Cool-down method for Black Tea, Genmai, Barley, and Buckwheat: place a metal container filled with ice and water on top of the tea to help it cool. Follow this when room-temperature cooling is required after brewing.",
        ),
        (
            "policy_rule",
            "Crushed pistachio preparation (food processor)",
            4,
            ["pistachio", "food_processor", "topping"],
            "Crushed pistachio topping: use the food processor on Chop mode for 6–8 passes. Applies to kitchen staff preparing drink toppings.",
        ),
    ]
    for rtype, title, page, tags, rt in policy_chunks:
        records.append(
            Record(
                id="",
                type=rtype,
                title=title,
                entity_name=title,
                doc_type="sop_kitchen",
                role_scope=["kitchen"],
                tags=tags,
                source_file=source_file,
                source_page=page,
                retrieval_text=rt,
            )
        )

    # --- Batch / tea brewing specs: ingredients + steps + shelf life ---
    # Each tuple: title, page, ingredients, steps, storage_life, extra tags, retrieval summary sentences
    batches: list[tuple] = [
        (
            "Boba — rice cooker (1 batch)",
            1,
            ["Boba 600g", "Water to 20-cup line in rice cooker", "Sugar syrup 120g", "Hot water (boil)", "250ml hot water for finishing", "120g sugar syrup for finishing mix"],
            [
                "Pour hot water into the rice cooker.",
                "Bring water to a boil (about 8 minutes). When boiling, steam will come from the vent hole.",
                "Add boba; stir to prevent sticking to the bottom.",
                "Cook 30 minutes with heat on.",
                "Remove from stove; rest covered 15 minutes.",
                "Drain and rinse thoroughly under cold water to remove starchy liquid.",
                "Add 120g sugar syrup and 250ml hot water; mix well.",
            ],
            "Use within 4–6 hours (max).",
            ["boba", "rice_cooker", "topping"],
            "Kitchen batch recipe: cook boba in the rice cooker for drinks. After cooking, rinse, sweeten, and hold for service. Shelf life after prep: use within about 4–6 hours.",
        ),
        (
            "Boba — pot method (1 batch)",
            1,
            ["Boba 600g", "Water 5000ml", "Sugar syrup 120g"],
            [
                "Pour hot water into the pot and bring to a boil.",
                "Follow the same cook/rest/rinse/sweeten sequence as the rice cooker method (steps 3–7 of the rice cooker procedure).",
            ],
            "Use within 4–6 hours (max).",
            ["boba", "pot", "topping"],
            "Kitchen batch recipe: alternate boba cooking method using a pot with measured water. Finish with syrup mix consistent with the rice cooker procedure. Hold time: use within about 4–6 hours.",
        ),
        (
            "TGY Tea Jelly (1 batch)",
            1,
            ["TGY tea 3000ml", "Jelly powder 200g", "Sugar syrup 140g", "Powdered sugar 100g"],
            [
                "Boil the tea.",
                "Whisk together tea, jelly powder, sugar syrup, and powdered sugar until uniform.",
                "When boiling, pour into the metal topping container and refrigerate at least 2 hours to set.",
            ],
            "Refrigerate; use within 3 days (max).",
            ["tea_jelly", "topping", "TGY"],
            "Kitchen batch recipe: TGY tea jelly for toppings. Must cool/set in the refrigerator for at least two hours. Stored jelly: use within about three days.",
        ),
        (
            "Matcha Jelly (1 batch)",
            1,
            [
                "SENCHA matcha powder 60g",
                "Water 3000ml total (600g at 167°F for matcha liquid; 2400ml boiled for jelly liquid)",
                "Powdered sugar 450g",
                "Jelly powder 180g",
            ],
            [
                "Matcha liquid: fully mix matcha powder with 600g of 167°F water from the kettle.",
                "Jelly liquid: boil 2400ml water; mix with jelly powder and powdered sugar; must return to a full boil.",
                "Off heat, fully combine matcha liquid and jelly liquid.",
                "Pour into the metal topping container and refrigerate at least 2 hours to set.",
            ],
            "Refrigerate; use within 3 days (max).",
            ["matcha", "jelly", "topping"],
            "Kitchen batch recipe: matcha jelly. Critical detail: the jelly liquid must reboil after adding powders/sugar. Chill until set (≥2 hours). Use within about three days.",
        ),
        (
            "Hun Kue (1 batch)",
            2,
            ["Water 3000ml", "Cake jelly (HK) powder 300g", "Brown sugar syrup 120g", "Powdered sugar 120g"],
            [
                "Bring water to a boil.",
                "With heat on, whisk in HK powder and powdered sugar until dissolved.",
                "Add syrup; continue heating until the mixture boils again.",
                "Pour into the metal topping container; refrigerate at least 2 hours to set.",
            ],
            "Refrigerate; use within 3 days (max).",
            ["hun_kue", "topping"],
            "Kitchen batch recipe: Hun Kue jelly topping. Must reach a second boil after adding syrup. Chill until set. Use within about three days.",
        ),
        (
            "Brown Sugar Cream foam (1 batch)",
            2,
            ["Milk 500g", "Cream 125g", "Cream powder 125g", "Brown sugar syrup 37g", "Powdered sugar 37g"],
            [
                "Mix milk, cream, and powder: start low speed level 2 for 2 minutes, then high speed level 10 for 8 minutes until milk foam forms (~10 minutes total).",
                "Add brown sugar syrup; mix until fully combined.",
                "Wrap blender with plastic wrap (about 1.5–2 feet) to prevent splashes.",
            ],
            "Use within 1 day (max).",
            ["cream_foam", "brown_sugar"],
            "Kitchen batch recipe: brown sugar cream foam. Mixer speeds/timings are specified to build stable foam. Shelf life: use the same day (about one day max).",
        ),
        (
            "Pistachio Cream foam (1 batch)",
            2,
            ["Milk 550g", "Cream 120g", "Cream powder 125g", "Powdered sugar 75g", "Pistachio paste 125g"],
            [
                "Mix milk, cream, and powder: start low speed level 2 for 2 minutes, then high speed level 10 for 8 minutes until milk foam forms (~10 minutes total).",
                "Add pistachio paste; mix until fully combined.",
                "Wrap blender with plastic wrap (about 1.5–2 feet) to prevent splashes.",
            ],
            "Use within 1 day (max).",
            ["cream_foam", "pistachio"],
            "Kitchen batch recipe: pistachio cream foam. Follow the specified mixing speeds/times, then incorporate pistachio paste. Shelf life: use the same day (about one day max).",
        ),
        (
            "Roasted pistachio (prep)",
            2,
            ["One layer of pistachios on a baking tray"],
            ["Roast in the oven at 350°F for 15 minutes; preheat the oven for 5 minutes first."],
            "Store and use per daily prep needs (roast as needed).",
            ["pistachio", "roasting"],
            "Kitchen prep: roast pistachios for toppings—single layer on a tray, 15 minutes at 350°F, with a 5-minute preheat.",
        ),
        (
            "Pistachio paste (batch)",
            2,
            ["Roasted pistachio 500g", "Raw pistachio 500g"],
            ["Combine roasted and raw pistachios in a food processor; blend until paste consistency (about 10 processing cycles)."],
            "Store covered; use for daily drink prep.",
            ["pistachio", "paste"],
            "Kitchen batch prep: pistachio paste from equal parts roasted and raw pistachio processed to a smooth paste.",
        ),
        (
            "Chestnut Cream (1 batch)",
            2,
            ["Cream 250g", "Milk 50g", "Chestnut puree 60g"],
            [
                "Add all ingredients to a mixing bowl.",
                "Blend with a hand blender ~2 minutes to soft peaks; check every ~30 seconds.",
            ],
            "Use within 1 day (max).",
            ["chestnut", "cream"],
            "Kitchen batch recipe: chestnut cream topping. Blend carefully to soft peaks. Use within about one day.",
        ),
        (
            "Black Tea (brew batch)",
            3,
            ["Tea 140g", "Hot water 6000ml"],
            [
                "Heat water from the water heater and boil on the stove.",
                "Remove from heat; add tea leaves; steep 11 minutes covered.",
                "Strain; uncover and cool to room temperature (no ice; do not over-cool).",
            ],
            "Hold/use window after brewing: about 4–6 hours.",
            ["black_tea", "brew"],
            "Kitchen tea brewing: black tea batch with specified leaf weight and steep time. Cool naturally without icing. Intended use window after brewing is about 4–6 hours.",
        ),
        (
            "Green Tea (brew batch)",
            3,
            ["Tea 160g", "Hot water 4200ml", "Ice 2800g"],
            [
                "Use water at 85°C (185°F). If hotter, cool; if cooler, heat to target on the stove.",
                "Steep 3 minutes 10 seconds covered.",
                "Strain and mix thoroughly with ice.",
            ],
            "Hold/use window after brewing: about 4–6 hours.",
            ["green_tea", "brew", "temperature_sensitive"],
            "Kitchen tea brewing: green tea requires ~185°F water and a precise steep time, then ice-quench mixing. Use within about 4–6 hours after brewing.",
        ),
        (
            "Four Seasons Tea (brew batch)",
            3,
            ["Tea 160g", "Hot water 4200ml", "Ice 2800g"],
            [
                "Boil water from the water heater on the stove.",
                "Remove from heat; steep 10 minutes covered.",
                "Strain and mix thoroughly with ice.",
            ],
            "Hold/use window after brewing: about 4–6 hours.",
            ["four_seasons", "brew"],
            "Kitchen tea brewing: Four Seasons tea batch with 10-minute steep and ice mixing. Use within about 4–6 hours.",
        ),
        (
            "TGY Oolong Tea (brew batch)",
            3,
            ["Tea 160g", "Hot water 4200ml", "Ice 2800g"],
            [
                "Boil water from the water heater on the stove.",
                "Remove from heat; steep 6 minutes covered.",
                "Strain and mix thoroughly with ice.",
            ],
            "Hold/use window after brewing: about 4–6 hours.",
            ["TGY", "oolong", "brew"],
            "Kitchen tea brewing: TGY oolong batch with 6-minute steep and ice mixing. Use within about 4–6 hours.",
        ),
        (
            "Genmai (brew batch)",
            3,
            ["Genmai 120g", "Hot water 6000ml"],
            [
                "Boil water on the stove.",
                "Remove from heat; add Genmai; steep 40 minutes covered.",
                "Strain; uncover and cool naturally (no ice).",
            ],
            "Refrigerated hold: about 3 days.",
            ["genmai", "brew"],
            "Kitchen tea brewing: Genmai batch with long steep and room-temperature cooling. Refrigerated shelf life is about three days; taste daily for quality.",
        ),
        (
            "Barley tea (brew batch)",
            3,
            ["Barley 240g", "Hot water 6000ml"],
            [
                "Boil water on the stove.",
                "Remove from heat; steep barley 90 minutes covered.",
                "Strain; uncover and cool naturally (no ice).",
            ],
            "Refrigerated hold: about 4–5 days.",
            ["barley", "brew"],
            "Kitchen tea brewing: barley tea with a 90-minute steep and room-temperature cooling. Refrigerated shelf life is about 4–5 days.",
        ),
        (
            "Buckwheat tea (brew batch)",
            3,
            ["Buckwheat 120g", "Hot water 6000ml"],
            [
                "Boil water on the stove.",
                "Cook buckwheat 5 minutes covered.",
                "Remove from heat; steep 35 minutes covered.",
                "Strain; uncover and cool naturally (no ice).",
            ],
            "Refrigerated hold: about 3 days.",
            ["buckwheat", "brew"],
            "Kitchen tea brewing: buckwheat tea uses a cook+steep sequence then cooling. Refrigerated shelf life is about three days.",
        ),
    ]

    needles = [
        "Boba",
        "TGY Tea Jelly",
        "Matcha Jelly",
        "Hun Kue",
        "Brown Sugar Cream",
        "Pistachio Cream",
        "Black Tea",
        "Green Tea",
        "Four Seasons",
        "Genmai",
        "Buckwheat",
    ]
    for n in needles:
        _must_contain(blob, n)

    for title, page, ingredients, steps, storage, tags, rt_summary in batches:
        rt = (
            f"{title}. Type: batch_prep_recipe. Applies to kitchen staff making drink ingredients. "
            f"{rt_summary} Key ingredients: {', '.join(ingredients[:6])}. "
            f"Storage/shelf-life note: {storage}"
        )
        records.append(
            Record(
                id="",
                type="batch_prep_recipe",
                title=title,
                entity_name=title.split("—")[0].strip(),
                doc_type="sop_kitchen",
                role_scope=["kitchen"],
                ingredients=ingredients,
                steps=steps,
                storage_life=storage,
                tags=tags,
                source_file=source_file,
                source_page=page,
                retrieval_text=rt,
            )
        )

    return records
