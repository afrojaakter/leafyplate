"""LeafyPlate — Site-wide configuration constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Site Identity ──────────────────────────────────────────────
SITE_URL = os.environ.get("SITE_URL", "https://yourusername.github.io/LeafyPlate")
SITE_TITLE = "LeafyPlate"
SITE_TAGLINE = "Fresh ideas. Real food. Zero nonsense."
SITE_DESCRIPTION = (
    "An AI-powered healthy eating platform featuring recipes, nutrition articles, "
    "weekly meal plans, and shoppable shopping lists."
)
SITE_AUTHOR = "LeafyPlate"
SITE_LANGUAGE = "en"
SITE_LOGO_EMOJI = "\U0001f96c"  # 🥬

# ─── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "articles"
PERSONAL_PLANS_DIR = BASE_DIR / "data" / "personal_plans"
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATES_DIR = BASE_DIR / "templates"
PROMPTS_DIR = BASE_DIR / "prompts"
STATIC_DIR = BASE_DIR / "static"
PRODUCTS_DIR = BASE_DIR / "products"
SUBSCRIBERS_DIR = BASE_DIR / "subscribers" / "profiles"
PROCESSED_URLS_PATH = BASE_DIR / "data" / "processed_urls.json"

# ─── Content Generation ────────────────────────────────────────
ARTICLES_PER_PAGE = 15
DAILY_RECIPE_COUNT = 2
DAILY_ARTICLE_COUNT = 1
MYTH_BUSTER_FREQUENCY_DAYS = 3

# ─── Content Type URL Prefixes ─────────────────────────────────
URL_PREFIXES = {
    "recipe": "recipe",
    "article": "article",
    "meal_plan": "meal-plan",
    "myth_buster": "article",
}

# ─── Featured Tags ─────────────────────────────────────────────
FEATURED_TAGS = [
    "vegetarian", "vegan", "meal-prep", "under-30-min",
    "high-protein", "budget-friendly", "gluten-free",
    "nutrition-science", "myth-buster", "meal-plan",
]

# ─── Season Detection (Northern Hemisphere) ────────────────────
MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

# ─── Gemini AI ─────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.8
GEMINI_MAX_OUTPUT_TOKENS = 8192

# ─── External APIs ─────────────────────────────────────────────
SPOONACULAR_API_KEY = os.environ.get("SPOONACULAR_API_KEY", "")
USDA_API_KEY = os.environ.get("USDA_API_KEY", "")

# ─── Email (Brevo) ─────────────────────────────────────────────
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
BREVO_LIST_ID = int(os.environ.get("BREVO_LIST_ID") or "1")
BREVO_SENDER_NAME = "LeafyPlate"
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "digest@leafyplate.com")

# ─── Affiliate Programs ───────────────────────────────────────
AMAZON_AFFILIATE_TAG = os.environ.get("AMAZON_AFFILIATE_TAG", "leafyplate-20")
WALMART_AFFILIATE_ID = os.environ.get("WALMART_AFFILIATE_ID", "")
