"""
daily_batch_update.py — LeafyPlate content generation engine.

Fetches RSS feeds, identifies topics via Gemini AI, generates original content
(recipes, articles, myth busters, meal plans), and builds the static site.

Usage:
    python daily_batch_update.py              # Full daily batch
    python daily_batch_update.py --rebuild    # Rebuild site from existing data
    python daily_batch_update.py --test       # Generate one test recipe
"""

import os
import re
import sys
import json
import hashlib
import logging
import datetime
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote_plus
from email.utils import format_datetime

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    SITE_URL, SITE_TITLE, SITE_TAGLINE,
    ARTICLES_PER_PAGE, DAILY_RECIPE_COUNT, DAILY_ARTICLE_COUNT,
    MYTH_BUSTER_FREQUENCY_DAYS, URL_PREFIXES, FEATURED_TAGS,
    MONTH_TO_SEASON, GEMINI_API_KEY, GEMINI_MODEL,
    GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS,
    DATA_DIR, OUTPUT_DIR, TEMPLATES_DIR, PROMPTS_DIR, STATIC_DIR,
    PROCESSED_URLS_PATH, BASE_DIR,
)
from product_linker import link_all_ingredients, render_shopping_links_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("leafyplate")


# ═══════════════════════════════════════════════════════════════
# Section 1: Gemini API Client
# ═══════════════════════════════════════════════════════════════

def init_gemini():
    """Initialize the Gemini client."""
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set. Cannot generate content.")
        sys.exit(1)
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def call_gemini(client, prompt: str, max_retries: int = 2) -> str:
    """Call Gemini API and return the response text.

    Retries on failure up to max_retries times.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": GEMINI_TEMPERATURE,
                    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
            return response.text
        except Exception as e:
            log.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise
    return ""


def parse_gemini_json(response_text: str) -> dict | list:
    """Parse JSON from Gemini response, handling potential formatting issues."""
    text = response_text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Fix common Gemini JSON issues
    text = re.sub(r",\s*([}\]])", r"\1", text)          # trailing commas
    text = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', text)  # single-quoted keys
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)    # single-quoted values
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error at char {e.pos}: {e.msg}")
        # Log a snippet around the error position for debugging
        start = max(0, e.pos - 80)
        end = min(len(text), e.pos + 80)
        log.error(f"Context: ...{text[start:end]}...")
        raise


def generate_recipe_image(client, recipe: dict) -> str | None:
    """Generate a hero image for a recipe using Gemini image generation.

    Returns the relative image path (e.g. 'static/images/recipes/slug.png')
    or None if generation fails.
    """
    title = recipe.get("title", "")
    subtitle = recipe.get("subtitle", "")
    ingredients_summary = ", ".join(
        ing.get("item", "") for ing in recipe.get("ingredients", [])[:8]
    )

    prompt = (
        f"A beautiful overhead food photography shot of: {title}. "
        f"{subtitle}. Key ingredients: {ingredients_summary}. "
        "Professional food blog style, natural lighting, on a rustic wooden table, "
        "garnished and plated beautifully. Photorealistic, appetizing, warm tones."
    )

    log.info(f"Generating image for: {title}")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config={
                "response_modalities": ["IMAGE", "TEXT"],
            },
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Save the image
                slug = recipe.get("slug", "recipe")
                images_dir = STATIC_DIR / "images" / "recipes"
                images_dir.mkdir(parents=True, exist_ok=True)
                image_path = images_dir / f"{slug}.png"

                image_bytes = part.inline_data.data
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                relative_path = f"static/images/recipes/{slug}.png"
                log.info(f"Saved recipe image: {relative_path}")
                return relative_path

        log.warning("No image data in Gemini response")
        return None
    except Exception as e:
        log.warning(f"Image generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Section 2: RSS Feed Fetching
# ═══════════════════════════════════════════════════════════════

def load_sources() -> dict:
    """Load RSS feed sources from sources.json."""
    sources_path = BASE_DIR / "sources.json"
    with open(sources_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_all_feeds() -> list[dict]:
    """Parse all RSS feeds and return a flat list of feed items."""
    sources = load_sources()
    all_items = []

    for feed_def in sources.get("feeds", []):
        feed_id = feed_def["id"]
        feed_url = feed_def["url"]
        feed_name = feed_def["name"]
        category = feed_def.get("category", "general")

        log.info(f"Fetching feed: {feed_name}")
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:  # Limit to 10 most recent per feed
                all_items.append({
                    "source_id": feed_id,
                    "source_name": feed_name,
                    "category": category,
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "summary": entry.get("summary", "")[:500],
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            log.warning(f"Failed to fetch {feed_name}: {e}")

    log.info(f"Fetched {len(all_items)} items from {len(sources.get('feeds', []))} feeds")
    return all_items


# ═══════════════════════════════════════════════════════════════
# Section 3: Source Content Extraction
# ═══════════════════════════════════════════════════════════════

def extract_source_content(url: str, max_chars: int = 3000) -> str:
    """Fetch a source URL and extract main text content."""
    try:
        headers = {"User-Agent": "LeafyPlate/1.0 (content research)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        return text[:max_chars]
    except Exception as e:
        log.warning(f"Failed to extract content from {url}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════
# Section 4: Deduplication
# ═══════════════════════════════════════════════════════════════

def load_processed_urls() -> dict:
    """Load the set of already-processed URLs."""
    if PROCESSED_URLS_PATH.exists():
        with open(PROCESSED_URLS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_urls(processed: dict):
    """Save the processed URLs set."""
    PROCESSED_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_URLS_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)


def url_hash(url: str) -> str:
    """Generate MD5 hash of a URL for deduplication."""
    return hashlib.md5(url.encode()).hexdigest()


def is_duplicate(url: str, processed: dict) -> bool:
    """Check if a URL has already been processed."""
    return url_hash(url) in processed


def mark_processed(url: str, article_id: str, processed: dict):
    """Mark a URL as processed."""
    processed[url_hash(url)] = {
        "url": url,
        "article_id": article_id,
        "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# Section 5: Topic Identification via AI
# ═══════════════════════════════════════════════════════════════

def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = PROMPTS_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template: str, **kwargs) -> str:
    """Safely substitute {key} placeholders without breaking JSON braces.

    Uses string.Template with $-style substitution under the hood to avoid
    conflicts with JSON curly braces in prompt files.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def get_recent_articles(count: int = 20) -> list[dict]:
    """Load the most recent articles for context."""
    articles = load_all_articles()
    return articles[:count]


def identify_topics(client, feed_items: list[dict],
                    batch_size: int = 5) -> list[dict]:
    """Send feed items to Gemini to identify the best topics."""
    prompt_template = load_prompt("identify_food_topics.txt")
    recent = get_recent_articles(20)
    recent_summary = "\n".join(
        f"- {a['title']} ({a['type']}, {a['date']})"
        for a in recent
    ) or "No recent articles."

    # Format feed entries for the prompt
    entries_text = "\n".join(
        f"[{i+1}] {item['source_name']}: {item['title']}\n"
        f"    URL: {item['url']}\n"
        f"    Summary: {item['summary'][:200]}"
        for i, item in enumerate(feed_items[:50])
    )

    prompt = render_prompt(prompt_template,
        recent_articles=recent_summary,
        batch_size=batch_size,
        feed_entries=entries_text,
    )

    log.info(f"Identifying top {batch_size} topics from {len(feed_items)} items...")
    response = call_gemini(client, prompt)
    topics = parse_gemini_json(response)

    if isinstance(topics, list):
        log.info(f"Identified {len(topics)} topics")
        return topics
    return []


# ═══════════════════════════════════════════════════════════════
# Section 6: Content Generation
# ═══════════════════════════════════════════════════════════════

def get_current_season() -> str:
    """Return current season based on month (Northern Hemisphere)."""
    month = datetime.date.today().month
    return MONTH_TO_SEASON.get(month, "spring")


def generate_recipe(client, topic: dict, source_content: str) -> dict | None:
    """Generate a recipe post via Gemini."""
    prompt_template = load_prompt("generate_recipe.txt")
    prompt = render_prompt(prompt_template,
        source_content=source_content,
        season=get_current_season(),
        trending="seasonal vegetables, whole grains, legumes",
    )

    log.info(f"Generating recipe from: {topic.get('source_title', 'unknown')}")
    response = call_gemini(client, prompt)
    recipe = parse_gemini_json(response)

    if not isinstance(recipe, dict):
        return None

    # Validate minimum quality
    if len(recipe.get("ingredients", [])) < 3:
        log.warning("Recipe rejected: too few ingredients")
        return None
    if len(recipe.get("instructions", [])) < 2:
        log.warning("Recipe rejected: too few instructions")
        return None

    # Add metadata
    today = datetime.date.today()
    slug = recipe.get("title", "untitled").lower()
    slug = "-".join(slug.split()[:8])
    slug = "".join(c for c in slug if c.isalnum() or c == "-")

    recipe["id"] = f"{today.strftime('%Y%m%d')}-{slug}"
    recipe["type"] = "recipe"
    recipe["slug"] = slug
    recipe["date"] = today.isoformat()
    recipe["date_formatted"] = today.strftime("%B %d, %Y")
    recipe["author"] = "LeafyPlate Kitchen"
    recipe["total_time_minutes"] = (
        recipe.get("prep_time_minutes", 0) + recipe.get("cook_time_minutes", 0)
    )
    recipe["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Generate hero image
    image_path = generate_recipe_image(client, recipe)
    if image_path:
        recipe["image"] = image_path

    return recipe


def generate_article(client, topic: dict, source_content: str) -> dict | None:
    """Generate a nutrition article via Gemini."""
    prompt_template = load_prompt("generate_article.txt")
    prompt = render_prompt(prompt_template, source_content=source_content)

    log.info(f"Generating article from: {topic.get('source_title', 'unknown')}")
    response = call_gemini(client, prompt)
    article = parse_gemini_json(response)

    if not isinstance(article, dict):
        return None

    today = datetime.date.today()
    slug = article.get("title", "untitled").lower()
    slug = "-".join(slug.split()[:8])
    slug = "".join(c for c in slug if c.isalnum() or c == "-")

    article["id"] = f"{today.strftime('%Y%m%d')}-{slug}"
    article["type"] = "article"
    article["slug"] = slug
    article["date"] = today.isoformat()
    article["date_formatted"] = today.strftime("%B %d, %Y")
    article["author"] = "LeafyPlate"
    article["source_articles"] = [{
        "title": topic.get("source_title", ""),
        "url": topic.get("source_url", ""),
        "source": topic.get("source_name", ""),
    }]
    article["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return article


def generate_myth_buster(client, topic: dict, source_content: str) -> dict | None:
    """Generate a myth-busting article via Gemini."""
    prompt_template = load_prompt("generate_myth_buster.txt")
    prompt = render_prompt(prompt_template,
        topic=topic.get("angle", topic.get("source_title", "")),
        source_content=source_content,
    )

    log.info(f"Generating myth buster from: {topic.get('source_title', 'unknown')}")
    response = call_gemini(client, prompt)
    article = parse_gemini_json(response)

    if not isinstance(article, dict):
        return None

    today = datetime.date.today()
    slug = article.get("title", "untitled").lower()
    slug = "-".join(slug.split()[:8])
    slug = "".join(c for c in slug if c.isalnum() or c == "-")

    article["id"] = f"{today.strftime('%Y%m%d')}-{slug}"
    article["type"] = "myth_buster"
    article["slug"] = slug
    article["date"] = today.isoformat()
    article["date_formatted"] = today.strftime("%B %d, %Y")
    article["author"] = "LeafyPlate"
    article["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Ensure myth_buster tags
    tags = article.get("tags", [])
    if "myth-buster" not in tags:
        tags.insert(0, "myth-buster")
    article["tags"] = tags

    return article


# ═══════════════════════════════════════════════════════════════
# Section 7: JSON Data I/O
# ═══════════════════════════════════════════════════════════════

def save_article_json(article: dict):
    """Save article metadata as JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{article['id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(article, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {path.name}")


def load_all_articles() -> list[dict]:
    """Load all article JSON files, sorted by date descending."""
    articles = []
    if DATA_DIR.exists():
        for path in DATA_DIR.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                articles.append(json.load(f))
    articles.sort(key=lambda a: a.get("date", ""), reverse=True)
    return articles


# ═══════════════════════════════════════════════════════════════
# Section 8: Template Rendering
# ═══════════════════════════════════════════════════════════════

def load_template(name: str) -> str:
    """Load an HTML template."""
    path = TEMPLATES_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_template(template: str, **kwargs) -> str:
    """Replace {placeholder} tokens in template with kwargs.

    Uses a defaultdict so missing keys produce empty strings
    instead of raising KeyError.
    """
    safe_kwargs = defaultdict(str, kwargs)
    # Use format_map to handle missing keys gracefully
    try:
        return template.format_map(safe_kwargs)
    except (KeyError, ValueError):
        # Fallback: manual replacement
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


def get_base_path() -> str:
    """Get the base path for URLs (empty for root, or /LeafyPlate for GH Pages)."""
    from urllib.parse import urlparse
    parsed = urlparse(SITE_URL)
    return parsed.path.rstrip("/")


def render_full_page(content_html: str, **kwargs) -> str:
    """Wrap content in the base template."""
    base = load_template("base.html")
    base_path = get_base_path()
    year = datetime.date.today().year
    # Set defaults, then let kwargs override
    params = {
        "content": content_html,
        "base_path": base_path,
        "year": str(year),
        "og_type": "article",
        "extra_css": "",
        "page_title": "",
        "page_description": "",
        "page_url": "",
    }
    params.update(kwargs)
    return render_template(base, **params)


# ═══════════════════════════════════════════════════════════════
# Section 9: Page Renderers
# ═══════════════════════════════════════════════════════════════

def get_type_badge_info(article_type: str) -> tuple[str, str, str]:
    """Return (label, css_class, badge_class) for an article type."""
    mapping = {
        "recipe": ("Recipe", "recipe", "badge-recipe"),
        "article": ("Article", "article", "badge-article"),
        "meal_plan": ("Meal Plan", "meal-plan", "badge-meal-plan"),
        "myth_buster": ("Myth Buster", "myth-buster", "badge-myth-buster"),
    }
    return mapping.get(article_type, ("Post", "article", "badge-article"))


def render_tag_links(tags: list[str]) -> str:
    """Render a list of tag links."""
    base_path = get_base_path()
    return "\n".join(
        f'<a href="{base_path}/tags/{tag}/" class="tag">{tag}</a>'
        for tag in tags
    )


def render_article_card(article: dict) -> str:
    """Render an article preview card for index pages."""
    template = load_template("article_card.html")
    label, type_class, _ = get_type_badge_info(article.get("type", "article"))
    url_prefix = URL_PREFIXES.get(article.get("type", "article"), "article")
    base_path = get_base_path()

    # Build image HTML if available
    image = article.get("image", "")
    if image:
        card_image = (
            f'<div class="card-image">'
            f'<img src="{base_path}/{image}" alt="{article.get("title", "")}" loading="lazy">'
            f'</div>'
        )
    else:
        card_image = ""

    return render_template(
        template,
        type_label=label,
        type_class=type_class,
        date=article.get("date", ""),
        date_formatted=article.get("date_formatted", ""),
        title=article.get("title", "Untitled"),
        subtitle=article.get("subtitle", ""),
        summary=article.get("summary", ""),
        slug=article.get("slug", ""),
        url_prefix=url_prefix,
        tag_links=render_tag_links(article.get("tags", [])),
        card_image=card_image,
        base_path=base_path,
    )


def render_recipe_page(article: dict) -> str:
    """Render a complete recipe page."""
    template = load_template("recipe.html")
    base_path = get_base_path()

    # Build ingredients HTML grouped
    groups = defaultdict(list)
    for ing in article.get("ingredients", []):
        groups[ing.get("group", "main")].append(ing)

    ingredients_html = ""
    for group_name, items in groups.items():
        if len(groups) > 1:
            ingredients_html += f'<p class="ingredient-group-label">{group_name}</p>\n'
        ingredients_html += "<ul>\n"
        for ing in items:
            amount = ing.get("amount", "")
            unit = ing.get("unit", "")
            item = ing.get("item", "")
            prep = ing.get("prep", "")
            line = f'<span class="ingredient-amount">{amount} {unit}</span> {item}'
            if prep:
                line += f", {prep}"
            ingredients_html += f"  <li>{line}</li>\n"
        ingredients_html += "</ul>\n"

    # Build instructions HTML
    instructions_html = "\n".join(
        f"<li>{step['text']}</li>"
        for step in article.get("instructions", [])
    )

    # Build tips HTML
    tips_html = "\n".join(
        f"<li>{tip}</li>"
        for tip in article.get("tips", [])
    )

    # Diet label badges
    diet_label_badges = "\n".join(
        f'<span class="diet-label">{label}</span>'
        for label in article.get("diet_labels", [])
    )

    # Nutrition
    nutrition = article.get("nutrition_per_serving", {})

    # Shopping links
    linked_ingredients = link_all_ingredients(article.get("ingredients", []))
    shopping_links_html = render_shopping_links_html(linked_ingredients)

    # Encode for share URLs
    title = article.get("title", "")
    article_url = f"{SITE_URL}/recipe/{article.get('slug', '')}/"
    encoded_title = quote_plus(title)
    encoded_url = quote_plus(article_url)

    # Recipe hero image
    image = article.get("image", "")
    if image:
        recipe_image = (
            f'<div class="recipe-hero-image">'
            f'<img src="{base_path}/{image}" alt="{title}" loading="lazy">'
            f'</div>'
        )
    else:
        recipe_image = ""

    content = render_template(
        template,
        date=article.get("date", ""),
        date_formatted=article.get("date_formatted", ""),
        title=title,
        subtitle=article.get("subtitle", ""),
        total_time_minutes=str(article.get("total_time_minutes", 0)),
        servings=str(article.get("servings", 4)),
        difficulty=article.get("difficulty", "easy"),
        diet_label_badges=diet_label_badges,
        intro_html=article.get("intro_html", ""),
        ingredients_html=ingredients_html,
        instructions_html=instructions_html,
        tips_html=tips_html,
        calories=str(nutrition.get("calories", 0)),
        protein_g=str(nutrition.get("protein_g", 0)),
        carbs_g=str(nutrition.get("carbs_g", 0)),
        fat_g=str(nutrition.get("fat_g", 0)),
        fiber_g=str(nutrition.get("fiber_g", 0)),
        sodium_mg=str(nutrition.get("sodium_mg", 0)),
        shopping_links_html=shopping_links_html,
        encoded_title=encoded_title,
        encoded_url=encoded_url,
        recipe_image=recipe_image,
        tag_links=render_tag_links(article.get("tags", [])),
        base_path=base_path,
    )

    extra_css = f'<link rel="stylesheet" href="{base_path}/static/css/recipe-card.css">\n'
    extra_css += f'<link rel="stylesheet" href="{base_path}/static/css/print.css">'

    return render_full_page(
        content,
        page_title=title,
        page_description=article.get("summary", ""),
        page_url=article_url,
        extra_css=extra_css,
    )


def render_article_page(article: dict) -> str:
    """Render a complete article/myth-buster page."""
    template = load_template("article.html")
    base_path = get_base_path()

    article_type = article.get("type", "article")
    label, _, badge_class = get_type_badge_info(article_type)

    # Key takeaways
    takeaways = article.get("key_takeaways", [])
    takeaways_html = "\n".join(f"<li>{t}</li>" for t in takeaways)

    title = article.get("title", "")
    url_prefix = URL_PREFIXES.get(article_type, "article")
    article_url = f"{SITE_URL}/{url_prefix}/{article.get('slug', '')}/"
    encoded_title = quote_plus(title)
    encoded_url = quote_plus(article_url)

    content = render_template(
        template,
        type_label=label,
        badge_class=badge_class,
        date=article.get("date", ""),
        date_formatted=article.get("date_formatted", ""),
        title=title,
        subtitle=article.get("subtitle", ""),
        content_html=article.get("content_html", ""),
        takeaways_html=takeaways_html,
        encoded_title=encoded_title,
        encoded_url=encoded_url,
        tag_links=render_tag_links(article.get("tags", [])),
        base_path=base_path,
    )

    return render_full_page(
        content,
        page_title=title,
        page_description=article.get("summary", ""),
        page_url=article_url,
    )


def render_meal_plan_page(article: dict) -> str:
    """Render a complete meal plan page."""
    template = load_template("meal_plan.html")
    base_path = get_base_path()

    # Day cards
    days = article.get("days", {})
    day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day_cards_html = ""
    for day_name in day_order:
        day = days.get(day_name, {})
        if not day:
            continue
        meals_html = ""
        for meal_type in ["breakfast", "lunch", "dinner", "snack"]:
            meal = day.get(meal_type, {})
            if meal:
                name = meal.get("name", "")
                cals = meal.get("calories", "")
                meals_html += (
                    f'<div class="meal-row">\n'
                    f'  <span class="meal-label">{meal_type}</span>\n'
                    f'  <span class="meal-name">{name}</span>\n'
                    f'  <span class="meal-cals">{cals} cal</span>\n'
                    f'</div>\n'
                )
        day_cards_html += (
            f'<div class="day-card">\n'
            f'  <h3>{day_name}</h3>\n'
            f'  {meals_html}\n'
            f'</div>\n'
        )

    # Shopping list
    shopping = article.get("shopping_list", {})
    shopping_list_html = ""
    for section, items in shopping.items():
        items_html = "\n".join(f"<li>{item}</li>" for item in items)
        shopping_list_html += (
            f'<div class="shopping-group">\n'
            f'  <h3>{section.title()}</h3>\n'
            f'  <ul>{items_html}</ul>\n'
            f'</div>\n'
        )

    # Prep tips
    prep_tips = article.get("prep_tips", [])
    prep_tips_html = "\n".join(f"<li>{tip}</li>" for tip in prep_tips)

    title = article.get("title", "")
    article_url = f"{SITE_URL}/meal-plan/{article.get('slug', '')}/"
    encoded_title = quote_plus(title)
    encoded_url = quote_plus(article_url)

    content = render_template(
        template,
        title=title,
        overview=article.get("overview", ""),
        total_daily_calories_avg=str(article.get("total_daily_calories_avg", 0)),
        estimated_weekly_cost=article.get("estimated_weekly_cost", ""),
        day_cards_html=day_cards_html,
        shopping_list_html=shopping_list_html,
        prep_tips_html=prep_tips_html,
        encoded_title=encoded_title,
        encoded_url=encoded_url,
        tag_links=render_tag_links(article.get("tags", [])),
        base_path=base_path,
    )

    extra_css = f'<link rel="stylesheet" href="{base_path}/static/css/meal-plan.css">\n'
    extra_css += f'<link rel="stylesheet" href="{base_path}/static/css/print.css">'

    return render_full_page(
        content,
        page_title=title,
        page_description=article.get("summary", article.get("overview", "")),
        page_url=article_url,
        extra_css=extra_css,
    )


# ═══════════════════════════════════════════════════════════════
# Section 10: Site Builder
# ═══════════════════════════════════════════════════════════════

def write_output(path: str, content: str):
    """Write content to an output file, creating directories as needed."""
    full_path = OUTPUT_DIR / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_article_pages(articles: list[dict]):
    """Generate individual article pages."""
    for article in articles:
        article_type = article.get("type", "article")
        slug = article.get("slug", "")
        url_prefix = URL_PREFIXES.get(article_type, "article")

        if article_type == "recipe":
            html = render_recipe_page(article)
        elif article_type == "meal_plan":
            html = render_meal_plan_page(article)
        else:
            html = render_article_page(article)

        write_output(f"{url_prefix}/{slug}/index.html", html)
        log.info(f"Built page: {url_prefix}/{slug}/")


def build_index_pages(articles: list[dict]):
    """Generate paginated index pages."""
    template = load_template("index.html")
    total_pages = max(1, (len(articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE)

    for page_num in range(1, total_pages + 1):
        start = (page_num - 1) * ARTICLES_PER_PAGE
        end = start + ARTICLES_PER_PAGE
        page_articles = articles[start:end]

        cards_html = "\n".join(render_article_card(a) for a in page_articles)

        # Build pagination
        base_path = get_base_path()
        pagination_html = ""
        if total_pages > 1:
            for p in range(1, total_pages + 1):
                if p == page_num:
                    pagination_html += f'<span class="current">{p}</span>\n'
                elif p == 1:
                    pagination_html += f'<a href="{base_path}/">{p}</a>\n'
                else:
                    pagination_html += f'<a href="{base_path}/page/{p}/">{p}</a>\n'

        content = render_template(
            template,
            page_heading="Latest" if page_num == 1 else f"Page {page_num}",
            article_cards=cards_html,
            pagination=pagination_html,
        )

        page_html = render_full_page(
            content,
            page_title="LeafyPlate" if page_num == 1 else f"Page {page_num}",
            page_description=SITE_TAGLINE,
        )

        if page_num == 1:
            write_output("index.html", page_html)
        else:
            write_output(f"page/{page_num}/index.html", page_html)

    log.info(f"Built {total_pages} index pages")


def build_tag_pages(articles: list[dict]):
    """Generate tag archive pages."""
    tag_map: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        for tag in article.get("tags", []):
            tag_map[tag].append(article)

    # Tag index page
    tag_index_template = load_template("tags_index.html")
    tag_cards_html = ""
    for tag_name in sorted(tag_map.keys()):
        base_path = get_base_path()
        count = len(tag_map[tag_name])
        tag_cards_html += (
            f'<a href="{base_path}/tags/{tag_name}/" class="tag-card">\n'
            f'  <div class="tag-card-name">{tag_name}</div>\n'
            f'  <div class="tag-card-count">{count} posts</div>\n'
            f'</a>\n'
        )

    tag_index_content = render_template(tag_index_template, tag_cards=tag_cards_html)
    tag_index_html = render_full_page(
        tag_index_content,
        page_title="All Tags",
        page_description="Browse LeafyPlate content by topic",
    )
    write_output("tags/index.html", tag_index_html)

    # Individual tag pages
    tag_template = load_template("tag.html")
    for tag_name, tag_articles in tag_map.items():
        cards_html = "\n".join(render_article_card(a) for a in tag_articles)
        content = render_template(
            tag_template,
            tag_name=tag_name,
            article_count=str(len(tag_articles)),
            article_cards=cards_html,
            pagination="",
        )
        html = render_full_page(
            content,
            page_title=f"Tagged: {tag_name}",
            page_description=f"LeafyPlate posts tagged '{tag_name}'",
        )
        write_output(f"tags/{tag_name}/index.html", html)

    log.info(f"Built {len(tag_map)} tag pages")


def build_archive_pages(articles: list[dict]):
    """Generate monthly archive pages."""
    month_map: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        date_str = article.get("date", "")
        if len(date_str) >= 7:
            month_key = date_str[:7]  # YYYY-MM
            month_map[month_key].append(article)

    archive_template = load_template("archive.html")
    base_path = get_base_path()

    # Archive index
    archive_index_html = "<h1>Archive</h1>\n<ul>\n"
    for month_key in sorted(month_map.keys(), reverse=True):
        count = len(month_map[month_key])
        year, month = month_key.split("-")
        month_name = datetime.date(int(year), int(month), 1).strftime("%B %Y")
        archive_index_html += (
            f'<li><a href="{base_path}/archive/{year}/{month}/">'
            f'{month_name}</a> ({count} posts)</li>\n'
        )
    archive_index_html += "</ul>"

    write_output(
        "archive/index.html",
        render_full_page(
            archive_index_html,
            page_title="Archive",
            page_description="LeafyPlate monthly archive",
        ),
    )

    # Monthly pages
    for month_key, month_articles in month_map.items():
        year, month = month_key.split("-")
        month_name = datetime.date(int(year), int(month), 1).strftime("%B %Y")
        cards_html = "\n".join(render_article_card(a) for a in month_articles)
        content = render_template(
            archive_template,
            archive_title=month_name,
            archive_description=f"{len(month_articles)} posts",
            article_cards=cards_html,
            pagination="",
        )
        html = render_full_page(
            content,
            page_title=month_name,
            page_description=f"LeafyPlate posts from {month_name}",
        )
        write_output(f"archive/{year}/{month}/index.html", html)

    log.info(f"Built {len(month_map)} archive pages")


def build_static_pages():
    """Generate static pages (about, 404, signup)."""
    base_path = get_base_path()

    for page_name in ["about", "404", "signup"]:
        template = load_template(f"{page_name}.html")
        content = render_template(template, base_path=base_path)
        html = render_full_page(
            content,
            page_title=page_name.replace("_", " ").title() if page_name != "404" else "Page Not Found",
            page_description="",
        )
        if page_name == "404":
            write_output("404.html", html)
        else:
            write_output(f"{page_name}/index.html", html)

    log.info("Built static pages (about, 404, signup)")


def build_rss_feed(articles: list[dict], count: int = 20):
    """Generate RSS feed XML."""
    feed_template = load_template("rss_feed.xml")
    item_template = load_template("rss_item.xml")

    items_xml = ""
    for article in articles[:count]:
        url_prefix = URL_PREFIXES.get(article.get("type", "article"), "article")
        article_url = f"{SITE_URL}/{url_prefix}/{article.get('slug', '')}/"

        categories = "\n".join(
            f"      <category>{tag}</category>"
            for tag in article.get("tags", [])
        )

        # Format pub date
        date_str = article.get("date", "")
        try:
            dt = datetime.datetime.fromisoformat(date_str)
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            pub_date = format_datetime(dt)
        except (ValueError, TypeError):
            pub_date = ""

        items_xml += render_template(
            item_template,
            title=article.get("title", ""),
            article_url=article_url,
            summary=article.get("summary", ""),
            pub_date=pub_date,
            categories=categories,
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    feed_xml = render_template(
        feed_template,
        site_url=SITE_URL,
        build_date=format_datetime(now),
        items=items_xml,
    )

    write_output("feed.xml", feed_xml)
    log.info(f"Built RSS feed with {min(count, len(articles))} items")


def build_sitemap(articles: list[dict]):
    """Generate sitemap.xml."""
    template = load_template("sitemap.xml")

    article_urls = ""
    for article in articles:
        url_prefix = URL_PREFIXES.get(article.get("type", "article"), "article")
        article_url = f"{SITE_URL}/{url_prefix}/{article.get('slug', '')}/"
        article_urls += (
            f"  <url>\n"
            f"    <loc>{article_url}</loc>\n"
            f"    <lastmod>{article.get('date', '')}</lastmod>\n"
            f"    <changefreq>monthly</changefreq>\n"
            f"    <priority>0.8</priority>\n"
            f"  </url>\n"
        )

    sitemap_xml = render_template(
        template,
        site_url=SITE_URL,
        article_urls=article_urls,
    )
    write_output("sitemap.xml", sitemap_xml)
    log.info(f"Built sitemap with {len(articles)} URLs")


def copy_static_assets():
    """Copy static files to output directory."""
    import shutil
    dest = OUTPUT_DIR / "static"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(STATIC_DIR, dest)
    log.info("Copied static assets")


def build_site():
    """Build the complete static site from article data."""
    log.info("Building static site...")
    articles = load_all_articles()
    log.info(f"Loaded {len(articles)} articles")

    build_article_pages(articles)
    build_index_pages(articles)
    build_tag_pages(articles)
    build_archive_pages(articles)
    build_static_pages()
    build_rss_feed(articles)
    build_sitemap(articles)
    copy_static_assets()

    log.info(f"Site built successfully in {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════
# Section 11: Main Orchestration
# ═══════════════════════════════════════════════════════════════

def should_generate_myth_buster() -> bool:
    """Check if today is a myth-buster generation day."""
    today = datetime.date.today()
    return today.toordinal() % MYTH_BUSTER_FREQUENCY_DAYS == 0


def daily_batch():
    """Run the full daily content generation pipeline."""
    log.info("=" * 60)
    log.info("LeafyPlate — Daily Batch Generation")
    log.info("=" * 60)

    client = init_gemini()
    processed = load_processed_urls()

    # Step 1: Fetch RSS feeds
    feed_items = fetch_all_feeds()
    if not feed_items:
        log.warning("No feed items fetched. Rebuilding site from existing data.")
        build_site()
        return

    # Step 2: Filter out already-processed URLs
    new_items = [
        item for item in feed_items
        if not is_duplicate(item["url"], processed)
    ]
    log.info(f"{len(new_items)} new items (out of {len(feed_items)} total)")

    if not new_items:
        log.info("No new items. Rebuilding site from existing data.")
        build_site()
        return

    # Step 3: Identify best topics
    batch_size = DAILY_RECIPE_COUNT + DAILY_ARTICLE_COUNT + (1 if should_generate_myth_buster() else 0)
    topics = identify_topics(client, new_items, batch_size=batch_size)

    # Step 4: Generate content
    generated_count = 0
    recipe_count = 0
    article_count = 0

    for topic in topics:
        source_url = topic.get("source_url", "")
        suggested_type = topic.get("suggested_type", "recipe")

        # Extract source content
        source_content = extract_source_content(source_url) if source_url else ""

        article = None
        if suggested_type == "recipe" and recipe_count < DAILY_RECIPE_COUNT:
            article = generate_recipe(client, topic, source_content)
            if article:
                recipe_count += 1
        elif suggested_type == "myth_buster" and should_generate_myth_buster():
            article = generate_myth_buster(client, topic, source_content)
        elif suggested_type == "article" and article_count < DAILY_ARTICLE_COUNT:
            article = generate_article(client, topic, source_content)
            if article:
                article_count += 1
        else:
            # Default to recipe if we haven't hit the limit
            if recipe_count < DAILY_RECIPE_COUNT:
                article = generate_recipe(client, topic, source_content)
                if article:
                    recipe_count += 1
            elif article_count < DAILY_ARTICLE_COUNT:
                article = generate_article(client, topic, source_content)
                if article:
                    article_count += 1

        if article:
            save_article_json(article)
            mark_processed(source_url, article["id"], processed)
            generated_count += 1

    save_processed_urls(processed)
    log.info(f"Generated {generated_count} articles ({recipe_count} recipes, {article_count} articles)")

    # Step 5: Rebuild the entire site
    build_site()

    log.info("Daily batch complete!")


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LeafyPlate content generation engine")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild site from existing data only")
    parser.add_argument("--test", action="store_true", help="Generate one test recipe and build")
    args = parser.parse_args()

    if args.rebuild:
        build_site()
    elif args.test:
        log.info("Test mode: generating one recipe...")
        client = init_gemini()
        topic = {
            "source_title": "Test: Quick Healthy Dinner Ideas",
            "source_url": "",
            "source_name": "Test",
            "suggested_type": "recipe",
            "angle": "A quick and healthy weeknight dinner",
        }
        recipe = generate_recipe(client, topic, "Create a simple, healthy dinner recipe.")
        if recipe:
            save_article_json(recipe)
            log.info(f"Test recipe saved: {recipe['title']}")
        build_site()
    else:
        daily_batch()


if __name__ == "__main__":
    main()
