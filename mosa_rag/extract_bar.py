"""SOP - Bar (1).pdf → drink recipes and bar-side prep (matcha concentrate)."""

from __future__ import annotations

from mosa_rag.normalize import normalize_for_parse
from mosa_rag.pdf_text import PageText, join_pages
from mosa_rag.schema import Record


def _must_contain(blob: str, needle: str) -> None:
    if needle not in blob:
        raise ValueError(f"Expected bar PDF text to contain: {needle!r}")


def extract_bar_records(pages: list[PageText], source_file: str) -> list[Record]:
    blob = normalize_for_parse(join_pages(pages))
    # normalize_for_parse collapses PDF double-spaces; match normalized text.
    needles = [
        "Brew Tea",
        "Milk Tea",
        "Au Lait",
        "Mosa Signature",
        "Matcha Concentrate",
        "Grapefruit Bloom",
    ]
    for n in needles:
        _must_contain(blob, n)

    records: list[Record] = []

    def add_drink(
        title: str,
        page: int,
        *,
        entity: str,
        steps: list[str],
        ingredients: list[str] | None = None,
        tags: list[str] | None = None,
        extra_rt: str = "",
    ) -> None:
        ing = ingredients or []
        tagl = tags or []
        step_blob = " ".join([f"{i + 1}) {s}" for i, s in enumerate(steps)])
        rt = (
            f"{title}. Type: drink_recipe. Applies to bar staff making drinks for customers. "
            f"{extra_rt} Steps: {step_blob}"
        )
        if ing:
            rt += f" Key ingredients/components: {', '.join(ing)}."
        records.append(
            Record(
                id="",
                type="drink_recipe",
                title=title,
                entity_name=entity,
                doc_type="sop_bar",
                role_scope=["bar"],
                shift_scope=["opening", "mid", "closing"],
                ingredients=ing,
                steps=steps,
                tags=tagl + ["bar", "drink"],
                source_file=source_file,
                source_page=page,
                retrieval_text=rt,
            )
        )

    def add_batch_bar(
        title: str,
        page: int,
        *,
        entity: str,
        ingredients: list[str],
        steps: list[str],
        storage: str,
        tags: list[str] | None = None,
    ) -> None:
        tagl = tags or []
        step_blob = " ".join([f"{i + 1}) {s}" for i, s in enumerate(steps)])
        rt = (
            f"{title}. Type: batch_prep_recipe. Applies to bar/kitchen-adjacent prep for service. "
            f"Storage/shelf-life: {storage} Steps: {step_blob}"
        )
        records.append(
            Record(
                id="",
                type="batch_prep_recipe",
                title=title,
                entity_name=entity,
                doc_type="sop_bar",
                role_scope=["bar", "kitchen"],
                ingredients=ingredients,
                steps=steps,
                storage_life=storage,
                tags=tagl,
                source_file=source_file,
                source_page=page,
                retrieval_text=rt,
            )
        )

    # --- Brew tea family ---
    add_drink(
        "Brewed iced tea — Black tea base (shaken)",
        1,
        entity="Black brewed tea",
        ingredients=["300ml brewed tea", "1 scoop ice in shaker", "sugar to spec", "3–5 ice cubes in cup"],
        steps=[
            "Add 300ml tea, 1 scoop of ice, and sugar to the shaker.",
            "Add 3–5 ice cubes to the serving cup.",
            "Shake well; strain into the serving cup.",
            "Top off with additional brewed tea if needed to fill the cup.",
        ],
        tags=["brew_tea", "iced", "shaken"],
    )
    add_drink(
        "Brewed iced tea — Four Seasons + Buckwheat Barley split (variant)",
        1,
        entity="Four Seasons with Buckwheat/Barley",
        steps=[
            "Start from the standard brewed-tea build, but for the first step use 75ml Buckwheat Barley plus 225ml Four Seasons (instead of all one tea).",
            "Continue shaking/straining/topping per the standard brewed-tea procedure.",
        ],
        tags=["brew_tea", "variant", "blend"],
        extra_rt="Variant for brewed tea when the guest orders Four Seasons with Buckwheat Barley blending.",
    )
    add_drink(
        "Brewed iced tea — Genmai + Green split (variant)",
        1,
        entity="Genmai + Green blend",
        steps=[
            "Start from the standard brewed-tea build, but for the first step add 150ml Genmai tea and 150ml Green tea.",
            "Continue shaking/straining/topping per the standard brewed-tea procedure.",
        ],
        tags=["brew_tea", "variant", "blend"],
    )
    add_drink(
        "Brewed iced tea — TGY Oolong with Osmanthus (variant)",
        1,
        entity="TGY Oolong + Osmanthus syrup",
        steps=[
            "Start from the standard brewed-tea build, but add one shot of Osmanthus syrup to the first step.",
            "Continue shaking/straining/topping per the standard brewed-tea procedure.",
        ],
        tags=["brew_tea", "variant", "osmanthus"],
    )

    # --- Milk tea (NDC) ---
    add_drink(
        "Milk tea (non-dairy creamer) — Black tea base (shaken)",
        1,
        entity="NDC milk tea (black)",
        ingredients=["300ml tea", "2 flat spoons creamer", "1 scoop ice in shaker", "sugar", "3–5 ice cubes in cup"],
        steps=[
            "Add 300ml tea, 2 flat spoons creamer, 1 scoop ice, and sugar to the shaker.",
            "Add 3–5 ice cubes to the serving cup.",
            "Shake well; strain into the serving cup.",
            "Top off with brewed tea if needed to fill the cup.",
        ],
        tags=["milk_tea", "NDC", "shaken"],
    )
    add_drink(
        "Milk tea (non-dairy creamer) — Genmai + Green split (variant)",
        1,
        entity="NDC milk tea (Genmai + Green)",
        steps=[
            "Start from the NDC milk tea build, but for the first step add 150ml Genmai and 150ml Green tea.",
            "Continue with shaker/cup steps as standard for NDC milk tea.",
        ],
        tags=["milk_tea", "NDC", "variant"],
    )
    add_drink(
        "Milk tea (non-dairy creamer) — TGY Oolong with Osmanthus (variant)",
        1,
        entity="NDC milk tea (TGY + Osmanthus)",
        steps=[
            "Start from the NDC milk tea build, but add one shot of Osmanthus syrup to the first step.",
            "Continue with shaker/cup steps as standard for NDC milk tea.",
        ],
        tags=["milk_tea", "NDC", "variant"],
    )

    # --- Au lait ---
    add_drink(
        "Au lait (milk) — Black tea layered pour",
        1,
        entity="Au lait black tea",
        ingredients=["150ml milk", "sugar", "ice", "brewed tea for layering"],
        steps=[
            "Mix 150ml milk and sugar in the Pyrex; stir well.",
            "Add milk and ice to the serving cup up to the top of the right mountain logo (if no topping, do not exceed 200ml milk).",
            "Slowly pour tea along the back of a spoon to create layers.",
        ],
        tags=["au_lait", "layered"],
    )
    add_drink(
        "Au lait (milk) — Genmai + Green split to layering step (variant)",
        1,
        entity="Au lait (Genmai + Green)",
        steps=[
            "Prepare milk/sugar and cup fill to the logo line as usual.",
            "For the tea pour step, mix 100ml Genmai tea and 100ml Green tea into the layering step (third step).",
        ],
        tags=["au_lait", "variant"],
    )
    add_drink(
        "Au lait (milk) — Osmanthus shot (variant)",
        1,
        entity="Au lait (Osmanthus)",
        steps=[
            "Add one shot of Osmanthus syrup to the first step (milk/sugar mix step).",
            "Continue with ice fill and layered tea pour per Au lait procedure.",
        ],
        tags=["au_lait", "variant", "osmanthus"],
    )

    # --- Signatures / fruit teas ---
    add_drink(
        "Mosa Signature — Grapefruit Bloom",
        2,
        entity="Grapefruit Bloom",
        ingredients=[
            "grapefruit spoons + tea jelly in cup",
            "225ml Four Seasons tea",
            "75ml Buckwheat Barley",
            "small shot Grapefruit syrup",
            "ice + sugar in shaker",
        ],
        steps=[
            "Add 2 spoons grapefruit and 1 spoon tea jelly to the serving cup.",
            "Add 225ml Four Seasons tea, 75ml Buckwheat Barley, 1 small shot Grapefruit syrup, 3–5 ice cubes, and sugar to the shaker.",
            "Shake well; strain into the serving cup.",
            "Top off with brewed tea if needed to fill the cup.",
        ],
        tags=["signature", "grapefruit"],
    )
    add_drink(
        "Fresh Fruit Tea (grapefruit apple citrus build)",
        2,
        entity="Fresh Fruit Tea",
        ingredients=[
            "2 flat spoons grapefruit",
            "2 apple slices",
            "1 orange slice",
            "1 lemon slice",
            "300ml tea + fruit tea base sugar",
            "2T apple juice",
            "2T lemon juice",
            "1T orange juice",
        ],
        steps=[
            "Add fruit to the serving cup per fruit tea spec.",
            "Add 300ml tea, fruit tea base sugar, juices, 3–5 ice cubes, and sugar to the shaker.",
            "Shake well; strain into the serving cup.",
            "Top off with brewed tea if needed.",
        ],
        tags=["fruit_tea", "fresh_fruit"],
    )
    add_drink(
        "Mosa Signature — Pistachio Mist (build)",
        2,
        entity="Pistachio Mist",
        ingredients=["150ml Genmai", "150ml Green", "Pistachio Cream Foam", "⅓ spoon crushed pistachio"],
        steps=[
            "Build Genmai (150ml) + Green (150ml) tea base.",
            "Top with Pistachio Cream Foam.",
            "Finish with ⅓ spoon crushed pistachio.",
        ],
        tags=["signature", "pistachio"],
    )
    add_drink(
        "Mosa Signature — Brown Sugar Mist (build)",
        2,
        entity="Brown Sugar Mist",
        ingredients=["300ml TGY Oolong tea", "1 shot Osmanthus syrup", "Brown Sugar cream foam"],
        steps=[
            "Prepare 300ml TGY Oolong tea with 1 shot Osmanthus syrup.",
            "Top with Brown Sugar cream foam.",
        ],
        tags=["signature", "brown_sugar"],
    )
    add_drink(
        "Mosa Signature — Taiwanese Retro (build)",
        2,
        entity="Taiwanese Retro",
        ingredients=["Signature Black Tea Au Lait (milk)", "Boba topping"],
        steps=[
            "Make the Signature Black Tea Au Lait (milk) base.",
            "Add boba.",
        ],
        tags=["signature", "milk_tea", "boba"],
    )

    # --- Tea with fruit ---
    add_drink(
        "Tea with Fruit — Apple",
        2,
        entity="Apple fruit tea",
        steps=[
            "Add 2 slices of fruit to the serving cup.",
            "Add 350ml tea, 1 small shot fruit syrup, 2T juice, 3–5 ice cubes, and sugar syrup to the shaker; shake and finish.",
        ],
        tags=["fruit_tea", "apple"],
    )
    add_drink(
        "Tea with Fruit — Lemon / Orange / Strawberry",
        2,
        entity="Lemon/Orange/Strawberry fruit tea",
        steps=[
            "Add 1 spoon fruit to the serving cup.",
            "Add 350ml tea, 1 small shot fruit syrup, 3–5 ice cubes, and sugar syrup to the shaker; shake and finish.",
        ],
        tags=["fruit_tea", "citrus", "berry"],
    )
    add_drink(
        "Tea with Fruit — Mango",
        2,
        entity="Mango fruit tea",
        steps=[
            "Add fruit to the serving cup per mango spec.",
            "Add 350ml tea, 1 small shot fruit syrup, 3–5 ice cubes, and sugar syrup to the shaker; shake and finish.",
        ],
        tags=["fruit_tea", "mango"],
    )
    add_drink(
        "Tea with Fruit — Grapefruit",
        2,
        entity="Grapefruit fruit tea",
        steps=[
            "Add 2 flat spoons grapefruit to the serving cup.",
            "Add 350ml tea, 1 small shot fruit syrup, 3–5 ice cubes, and sugar syrup to the shaker; shake and finish.",
        ],
        tags=["fruit_tea", "grapefruit"],
    )

    # --- Matcha concentrate (prep) ---
    add_batch_bar(
        "Matcha concentrate (bar prep)",
        3,
        entity="Matcha concentrate",
        ingredients=["167°F water", "800ml water", "35g matcha powder"],
        steps=[
            "Heat water to 167°F using the kettle.",
            "Blend 800ml of 167°F water with 35g matcha in the juice blender for 30–60 seconds.",
            "Pour into a syrup bottle and cool with an ice water bath.",
        ],
        storage="Cool down after blending; store in syrup bottle for service (follow date/time labeling in kitchen policy).",
        tags=["matcha", "concentrate", "prep"],
    )

    # --- Matcha series drinks ---
    add_drink(
        "Matcha Series — Chestnut Forest (fixed ice)",
        3,
        entity="Chestnut Forest",
        ingredients=["150ml milk", "sugar", "ice", "200ml matcha", "chestnut cream"],
        steps=[
            "Mix 150ml milk and sugar in Pyrex (if customer requests “no cream”, use 200ml milk instead).",
            "Add ice to the serving cup.",
            "Slowly pour 200ml matcha along a spoon to layer.",
            "Add ice until liquid is above the 490 line on the serving cup.",
            "Top with chestnut cream.",
        ],
        tags=["matcha", "chestnut", "fixed_ice"],
    )
    add_drink(
        "Matcha Series — Genmai Matcha with Matcha Jelly",
        3,
        entity="Genmai Matcha + Matcha Jelly",
        ingredients=["Matcha Jelly", "Genmai", "Green", "Matcha", "ice", "sugar"],
        steps=[
            "Add 1 scoop Matcha Jelly to the cup.",
            "Add 200ml Genmai (or 100ml Green + 100ml Genmai), 150ml Matcha, 1 scoop ice, and sugar to the shaker (extra matcha: adjust to 175ml Genmai / split Genmai+Green and 175ml Matcha).",
            "Add ice to the serving cup; shake and strain.",
            "Top off with extra tea if needed (about half Genmai and half Matcha).",
        ],
        tags=["matcha", "genmai", "jelly"],
    )
    add_drink(
        "Matcha Series — Strawberry / Mango Matcha Latte (fixed ice)",
        3,
        entity="Strawberry/Mango Matcha Latte",
        steps=[
            "Add half spoon fresh strawberry or mango in the cup.",
            "Mix 150ml milk with half shot strawberry syrup (or 1 shot mango syrup) and sugar in Pyrex.",
            "Add ice to the cup to the top of the right mountain logo.",
            "Layer in 200ml matcha with a spoon back.",
            "Add more ice if needed to fill.",
        ],
        tags=["matcha", "latte", "fruit"],
    )
    add_drink(
        "Hot drink — Apple Spice Tea Cider / Mango Spice Tea Cider",
        3,
        entity="Spice tea cider (hot)",
        ingredients=["2 shots fruit syrup (Apple or Mango)", "1 shot cinnamon syrup", "black tea", "microwave"],
        steps=[
            "If a topping is ordered, add it first.",
            "Add 2 shots fruit syrup, 1 shot cinnamon syrup, and black tea to a hot drink cup.",
            "Microwave 1 minute 30 seconds to heat thoroughly.",
            "Stir well; secure the lid.",
        ],
        tags=["hot", "cider", "spice"],
    )

    return records
