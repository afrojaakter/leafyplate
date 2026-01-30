"""
send_meal_plans.py — Send personalized meal plan emails via Brevo.

Usage:
    python send_meal_plans.py                 # Send to all active subscribers
    python send_meal_plans.py --dry-run       # Preview emails without sending
    python send_meal_plans.py --profile afroj # Send to one subscriber
"""

import json
import datetime
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import requests

from config import (
    BREVO_API_KEY, BREVO_SENDER_NAME, BREVO_SENDER_EMAIL,
    PERSONAL_PLANS_DIR, SUBSCRIBERS_DIR, TEMPLATES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("send_meal_plans")

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


def load_subscriber(name: str) -> dict:
    """Load a subscriber profile."""
    path = SUBSCRIBERS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_plan(subscriber_name: str) -> dict | None:
    """Load the most recent meal plan for a subscriber."""
    plan_dir = PERSONAL_PLANS_DIR / subscriber_name
    if not plan_dir.exists():
        return None

    plans = sorted(plan_dir.glob("*.json"), reverse=True)
    if not plans:
        return None

    with open(plans[0], "r", encoding="utf-8") as f:
        return json.load(f)


def render_plan_email(plan: dict, profile: dict) -> str:
    """Render the personalized meal plan email HTML."""
    template_path = TEMPLATES_DIR / "personal_plan_email.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    today = datetime.date.today()
    # Find Monday of current week
    monday = today - datetime.timedelta(days=today.weekday())
    week_date = monday.strftime("%B %d, %Y")

    # Tonight's dinner
    day_name = today.strftime("%A").lower()
    days = plan.get("days", {})
    today_meals = days.get(day_name, {})
    tonight = today_meals.get("dinner", {})

    # Build days HTML
    day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    days_html = ""
    for dn in day_order:
        day = days.get(dn, {})
        if not day:
            continue
        days_html += f'<div style="margin-bottom:16px;padding:12px;background:#F0EDE6;border-radius:8px;">\n'
        days_html += f'<p style="color:#E07A3A;font-weight:600;font-size:13px;text-transform:capitalize;margin:0 0 8px;">{dn}</p>\n'
        for meal_type in ["breakfast", "lunch", "dinner", "snack"]:
            meal = day.get(meal_type, {})
            if meal:
                name = meal.get("name", "")
                cals = meal.get("calories", "")
                days_html += (
                    f'<p style="margin:2px 0;font-size:13px;">'
                    f'<span style="color:#7A7468;text-transform:uppercase;font-size:10px;font-weight:600;">{meal_type}</span> '
                    f'{name} <span style="color:#A39E94;">({cals} cal)</span></p>\n'
                )
        days_html += '</div>\n'

    # Shopping list HTML
    shopping = plan.get("shopping_list", {})
    shopping_html = ""
    for section, items in shopping.items():
        shopping_html += f'<p style="color:#E07A3A;font-weight:600;font-size:12px;text-transform:uppercase;margin:12px 0 4px;">{section}</p>\n'
        for item in items:
            shopping_html += f'<p style="margin:2px 0;font-size:13px;">☐ {item}</p>\n'

    # Prep tips
    prep_tips = plan.get("prep_tips", [])
    prep_tips_html = "\n".join(f"<li>{tip}</li>" for tip in prep_tips)

    # Replace placeholders
    replacements = {
        "{subscriber_name}": profile.get("name", ""),
        "{week_date}": week_date,
        "{personalization_notes}": plan.get("personalization_notes", ""),
        "{tonight_dinner_name}": tonight.get("name", "Check your plan"),
        "{tonight_dinner_time}": str(tonight.get("time_minutes", "")),
        "{tonight_dinner_calories}": str(tonight.get("calories", "")),
        "{days_html}": days_html,
        "{shopping_list_html}": shopping_html,
        "{prep_tips_html}": prep_tips_html,
        "{total_daily_calories_avg}": str(plan.get("total_daily_calories_avg", "")),
        "{estimated_weekly_cost}": plan.get("estimated_weekly_cost", ""),
        "{year}": str(today.year),
        "{unsubscribe_url}": "#",
        "{preferences_url}": "#",
    }

    html = template
    for key, value in replacements.items():
        html = html.replace(key, str(value))

    return html


def send_email(to_email: str, to_name: str, subject: str, html_content: str) -> bool:
    """Send an email via Brevo API."""
    if not BREVO_API_KEY:
        log.error("BREVO_API_KEY not set")
        return False

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": BREVO_API_KEY,
    }

    payload = {
        "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
        "to": [{"email": to_email, "name": to_name}],
        "subject": subject,
        "htmlContent": html_content,
    }

    try:
        resp = requests.post(BREVO_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        log.info(f"Email sent to {to_name} ({to_email})")
        return True
    except Exception as e:
        log.error(f"Failed to send email to {to_email}: {e}")
        return False


def run(profile_name: str | None = None, dry_run: bool = False):
    """Send meal plan emails to subscriber(s)."""
    if profile_name:
        names = [profile_name]
    else:
        names = [
            p.stem for p in SUBSCRIBERS_DIR.glob("*.json")
        ]

    for name in names:
        try:
            profile = load_subscriber(name)
        except Exception as e:
            log.error(f"Failed to load subscriber profile '{name}': {e}")
            continue

        if not profile.get("active", True):
            log.info(f"Skipping inactive subscriber: {name}")
            continue

        try:
            plan = load_latest_plan(name)
        except Exception as e:
            log.error(f"Failed to load plan for '{name}': {e}")
            continue

        if not plan:
            log.warning(f"No plan found for {name}")
            continue

        try:
            html = render_plan_email(plan, profile)
        except Exception as e:
            log.error(f"Failed to render email for '{name}': {e}")
            continue

        subject = f"Your Meal Plan — Week of {datetime.date.today().strftime('%B %d')}"

        if dry_run:
            log.info(f"[DRY RUN] Would send to {profile.get('name')} ({profile.get('email')})")
            # Save preview HTML
            preview_path = PERSONAL_PLANS_DIR / name / "email_preview.html"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write(html)
            log.info(f"Preview saved: {preview_path}")
        else:
            send_email(
                to_email=profile.get("email", ""),
                to_name=profile.get("name", ""),
                subject=subject,
                html_content=html,
            )


def main():
    parser = argparse.ArgumentParser(description="Send personalized meal plan emails")
    parser.add_argument("--profile", type=str, help="Send to a specific subscriber")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    args = parser.parse_args()

    run(profile_name=args.profile, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
