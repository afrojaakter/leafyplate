"""
send_digest.py — Send weekly blog digest email via Brevo.

Usage:
    python send_digest.py             # Send digest to all subscribers
    python send_digest.py --dry-run   # Preview without sending
"""

import json
import datetime
import argparse
import logging
from pathlib import Path

import requests

from config import (
    SITE_URL, BREVO_API_KEY, BREVO_LIST_ID,
    BREVO_SENDER_NAME, BREVO_SENDER_EMAIL,
    TEMPLATES_DIR, URL_PREFIXES,
)
from daily_batch_update import load_all_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("send_digest")

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
BREVO_CAMPAIGN_URL = "https://api.brevo.com/v3/emailCampaigns"


def get_this_weeks_articles() -> list[dict]:
    """Get articles published in the last 7 days."""
    articles = load_all_articles()
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)

    return [
        a for a in articles
        if a.get("date", "") >= week_ago.isoformat()
    ]


def render_digest_email(articles: list[dict]) -> str:
    """Render the weekly digest email HTML."""
    template_path = TEMPLATES_DIR / "newsletter_email.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Separate by type
    recipes = [a for a in articles if a.get("type") == "recipe"]
    non_recipes = [a for a in articles if a.get("type") != "recipe"]
    meal_plans = [a for a in articles if a.get("type") == "meal_plan"]

    # Recipe cards HTML
    recipe_cards = ""
    for recipe in recipes[:3]:
        url_prefix = URL_PREFIXES.get("recipe", "recipe")
        url = f"{SITE_URL}/{url_prefix}/{recipe.get('slug', '')}/"
        recipe_cards += (
            f'<div style="margin-bottom:16px;padding:16px;background:#F0EDE6;border-radius:8px;">\n'
            f'  <p style="color:#E07A3A;font-size:12px;font-weight:600;text-transform:uppercase;margin:0 0 4px;">Recipe</p>\n'
            f'  <p style="color:#2D2A24;font-size:16px;font-weight:600;margin:0 0 4px;">\n'
            f'    <a href="{url}" style="color:#2D2A24;text-decoration:none;">{recipe.get("title", "")}</a>\n'
            f'  </p>\n'
            f'  <p style="color:#7A7468;font-size:13px;margin:0;">{recipe.get("summary", "")}</p>\n'
            f'</div>\n'
        )

    # Featured article
    featured_html = ""
    if non_recipes:
        article = non_recipes[0]
        url_prefix = URL_PREFIXES.get(article.get("type", "article"), "article")
        url = f"{SITE_URL}/{url_prefix}/{article.get('slug', '')}/"
        featured_html = (
            f'<p style="color:#2D2A24;font-size:16px;font-weight:600;margin:0 0 4px;">\n'
            f'  <a href="{url}" style="color:#2D2A24;text-decoration:none;">{article.get("title", "")}</a>\n'
            f'</p>\n'
            f'<p style="color:#7A7468;font-size:13px;margin:0;">{article.get("summary", "")}</p>\n'
        )

    # Meal plan teaser
    meal_plan_teaser = ""
    meal_plan_url = SITE_URL
    if meal_plans:
        mp = meal_plans[0]
        meal_plan_teaser = mp.get("overview", mp.get("summary", ""))
        url_prefix = URL_PREFIXES.get("meal_plan", "meal-plan")
        meal_plan_url = f"{SITE_URL}/{url_prefix}/{mp.get('slug', '')}/"

    replacements = {
        "{recipe_cards_html}": recipe_cards or "<p>No new recipes this week.</p>",
        "{featured_article_html}": featured_html or "<p>No featured article this week.</p>",
        "{meal_plan_teaser}": meal_plan_teaser or "Check our latest meal plan on the site!",
        "{meal_plan_url}": meal_plan_url,
        "{year}": str(datetime.date.today().year),
        "{unsubscribe_url}": "#",
    }

    html = template
    for key, value in replacements.items():
        html = html.replace(key, str(value))

    return html


def send_campaign(subject: str, html_content: str, dry_run: bool = False):
    """Send digest as a Brevo campaign to the subscriber list."""
    if dry_run:
        preview_path = Path("data/digest_preview.html")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preview_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        log.info(f"[DRY RUN] Preview saved: {preview_path}")
        return

    if not BREVO_API_KEY:
        log.error("BREVO_API_KEY not set")
        return

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": BREVO_API_KEY,
    }

    # Create and send campaign
    payload = {
        "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
        "name": f"Weekly Digest — {datetime.date.today().isoformat()}",
        "subject": subject,
        "htmlContent": html_content,
        "recipients": {"listIds": [BREVO_LIST_ID]},
    }

    try:
        resp = requests.post(BREVO_CAMPAIGN_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        campaign = resp.json()
        campaign_id = campaign.get("id")
        log.info(f"Campaign created: {campaign_id}")

        # Send immediately
        send_url = f"{BREVO_CAMPAIGN_URL}/{campaign_id}/sendNow"
        resp = requests.post(send_url, headers=headers, timeout=30)
        resp.raise_for_status()
        log.info("Digest sent!")
    except Exception as e:
        log.error(f"Failed to send digest: {e}")


def main():
    parser = argparse.ArgumentParser(description="Send weekly blog digest")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    args = parser.parse_args()

    articles = get_this_weeks_articles()
    log.info(f"Found {len(articles)} articles from this week")

    if not articles:
        log.info("No articles this week. Skipping digest.")
        return

    html = render_digest_email(articles)
    subject = f"LeafyPlate Weekly Digest — {datetime.date.today().strftime('%B %d')}"

    send_campaign(subject, html, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
