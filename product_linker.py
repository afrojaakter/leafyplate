"""
product_linker.py — Maps ingredients to grocery product affiliate links.

Phase 1: Uses search URL fallback (link to store search results).
Phase 2: Will use curated product catalog with exact product URLs.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

from config import PRODUCTS_DIR, AMAZON_AFFILIATE_TAG, WALMART_AFFILIATE_ID


def load_affiliate_config() -> dict:
    """Load affiliate store configuration."""
    config_path = PRODUCTS_DIR / "affiliate_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_product_catalog() -> list[dict]:
    """Load curated product catalog (Phase 2)."""
    catalog_path = PRODUCTS_DIR / "catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("products", [])


def clean_ingredient_for_search(item: str, prep: str = "") -> str:
    """Clean ingredient name for a grocery store search query.

    Removes preparation instructions and normalizes the name.
    Example: 'canned chickpeas' -> 'chickpeas canned'
    """
    # Remove common prep words that aren't useful for searching
    remove_words = [
        "fresh", "optional", "to taste", "for serving",
        "for garnish", "as needed", "divided",
    ]
    cleaned = item.lower().strip()
    for word in remove_words:
        cleaned = cleaned.replace(word, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_search_url(store_config: dict, query: str, affiliate_id: str = "") -> str:
    """Build a store search URL with affiliate parameters."""
    url = store_config["search_url_template"].format(query=quote_plus(query))
    if affiliate_id:
        separator = "&" if "?" in url else "?"
        url += f"{separator}{store_config['affiliate_param']}={affiliate_id}"
    return url


def get_affiliate_id(store_config: dict) -> str:
    """Get the affiliate ID for a store from environment or config."""
    env_var = store_config.get("affiliate_id_env", "")
    return os.environ.get(env_var, "")


def link_ingredient(ingredient: dict, affiliate_config: dict,
                    preferred_store: str = "") -> dict:
    """Map a single ingredient to product links across stores.

    Args:
        ingredient: Dict with 'item', 'amount', 'unit', 'prep' keys.
        affiliate_config: Full affiliate configuration.
        preferred_store: Subscriber's preferred store (shown first).

    Returns:
        The ingredient dict with an added 'products' list.
    """
    stores = affiliate_config["stores"]
    default_stores = affiliate_config.get("default_stores", ["walmart", "amazon_fresh"])

    # Determine which stores to show
    store_order = list(default_stores)
    if preferred_store and preferred_store in stores:
        store_order = [preferred_store] + [s for s in store_order if s != preferred_store]

    search_query = clean_ingredient_for_search(ingredient["item"], ingredient.get("prep", ""))

    products = []
    for store_id in store_order:
        if store_id not in stores:
            continue
        store = stores[store_id]
        affiliate_id = get_affiliate_id(store)
        url = build_search_url(store, search_query, affiliate_id)
        products.append({
            "store": store_id,
            "store_name": store["name"],
            "badge_label": store["badge_label"],
            "color": store["color"],
            "url": url,
        })

    result = dict(ingredient)
    result["products"] = products
    return result


def link_all_ingredients(ingredients: list[dict],
                         preferred_store: str = "") -> list[dict]:
    """Map all ingredients to product links.

    Args:
        ingredients: List of ingredient dicts from a recipe/meal plan.
        preferred_store: Subscriber's preferred grocery store.

    Returns:
        List of ingredients with 'products' added to each.
    """
    config = load_affiliate_config()
    return [link_ingredient(ing, config, preferred_store) for ing in ingredients]


def render_shopping_links_html(linked_ingredients: list[dict]) -> str:
    """Render ingredient shopping links as HTML for templates."""
    lines = []
    for ing in linked_ingredients:
        amount = ing.get("amount", "")
        unit = ing.get("unit", "")
        item = ing.get("item", "")
        prep = ing.get("prep", "")

        name_parts = [amount, unit, item]
        if prep:
            name_parts.append(f"({prep})")
        name = " ".join(p for p in name_parts if p)

        store_links = []
        for product in ing.get("products", []):
            badge_class = f"store-badge-{product['store'].replace('_', '-')}"
            store_links.append(
                f'<a href="{product["url"]}" class="store-badge {badge_class}" '
                f'target="_blank" rel="noopener">{product["badge_label"]}</a>'
            )

        links_html = "\n            ".join(store_links)
        lines.append(
            f'<div class="shopping-item">\n'
            f'    <span class="shopping-item-name">{name}</span>\n'
            f'    <div class="store-links">\n'
            f'        {links_html}\n'
            f'    </div>\n'
            f'</div>'
        )

    return "\n".join(lines)


# ─── CLI Test Mode ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_ingredients = [
            {"item": "chicken thighs, boneless", "amount": "2", "unit": "lbs", "prep": "", "group": "protein"},
            {"item": "canned chickpeas", "amount": "2", "unit": "cans", "prep": "drained", "group": "pantry"},
            {"item": "baby spinach", "amount": "5", "unit": "oz", "prep": "", "group": "produce"},
            {"item": "Greek yogurt", "amount": "32", "unit": "oz", "prep": "plain", "group": "dairy"},
        ]

        linked = link_all_ingredients(test_ingredients, preferred_store="walmart")
        for ing in linked:
            print(f"\n{ing['amount']} {ing['unit']} {ing['item']}:")
            for p in ing["products"]:
                print(f"  [{p['badge_label']}] {p['url']}")

        print("\n--- HTML Output ---")
        print(render_shopping_links_html(linked))
    else:
        print("Usage: python product_linker.py --test")
