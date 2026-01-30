"""
meal_planner.py — Personalized meal plan generator.

Generates weekly meal plans tailored to subscriber profiles using Gemini AI.

Usage:
    python meal_planner.py                    # Generate for all active subscribers
    python meal_planner.py --profile afroj    # Generate for one subscriber
    python meal_planner.py --dry-run          # Preview without saving
"""

import json
import datetime
import argparse
import logging
from pathlib import Path

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, SUBSCRIBERS_DIR,
    PERSONAL_PLANS_DIR, PROMPTS_DIR, MONTH_TO_SEASON,
)
from daily_batch_update import (
    init_gemini, call_gemini, parse_gemini_json,
    load_all_articles, load_prompt, render_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("meal_planner")


def load_subscriber(name: str) -> dict:
    """Load a subscriber profile by name."""
    path = SUBSCRIBERS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_subscribers() -> list[dict]:
    """Load all active subscriber profiles."""
    subscribers = []
    if SUBSCRIBERS_DIR.exists():
        for path in SUBSCRIBERS_DIR.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                profile["_filename"] = path.stem
                if profile.get("active", True):
                    subscribers.append(profile)
    return subscribers


def get_recent_plans(subscriber_name: str, weeks: int = 2) -> str:
    """Load recent meal plans for a subscriber to avoid repetition."""
    plans_dir = PERSONAL_PLANS_DIR / subscriber_name
    if not plans_dir.exists():
        return "No previous plans."

    plans = sorted(plans_dir.glob("*.json"), reverse=True)[:weeks]
    summaries = []
    for plan_path in plans:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        title = plan.get("title", "Unknown")
        summaries.append(f"- {title}")
    return "\n".join(summaries) if summaries else "No previous plans."


def generate_personal_plan(client, profile: dict) -> dict | None:
    """Generate a personalized weekly meal plan for a subscriber."""
    prompt_template = load_prompt("generate_personal_plan.txt")

    name = profile.get("name", "Subscriber")
    prefs = profile.get("preferences", {})
    nutrition = profile.get("nutrition_targets", {})
    schedule = profile.get("schedule", {})

    today = datetime.date.today()
    week_number = today.isocalendar()[1]
    season = MONTH_TO_SEASON.get(today.month, "spring")
    subscriber_name = profile.get("_filename", name.lower())

    prompt = render_prompt(prompt_template,
        subscriber_profile=json.dumps(profile, indent=2),
        daily_calories=nutrition.get("daily_calories", 1800),
        busy_nights=", ".join(schedule.get("busy_nights", [])) or "none",
        meal_prep_day=schedule.get("meal_prep_day", "sunday"),
        household_size=profile.get("household_size", 1),
        cooking_skill=prefs.get("cooking_skill", "intermediate"),
        budget=profile.get("budget", "moderate"),
        max_cooking_time=prefs.get("max_cooking_time_minutes", 45),
        season=season,
        week_number=week_number,
        subscriber_name=name,
        recent_plans_summary=get_recent_plans(subscriber_name),
    )

    log.info(f"Generating meal plan for {name} (week {week_number})...")
    try:
        response = call_gemini(client, prompt)
        plan = parse_gemini_json(response)
    except Exception as e:
        log.error(f"Gemini API error generating plan for {name}: {e}")
        return None

    if not isinstance(plan, dict):
        log.error(f"Invalid plan response for {name}")
        return None

    # Validate: check all 7 days present
    days = plan.get("days", {})
    expected_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    missing = expected_days - set(days.keys())
    if missing:
        log.warning(f"Plan for {name} missing days: {missing}")

    # Validate: check no disliked ingredients in meal names
    disliked = set(ing.lower() for ing in prefs.get("disliked_ingredients", []))
    if disliked:
        for day_name, day_meals in days.items():
            for meal_type, meal in day_meals.items():
                meal_name = meal.get("name", "").lower()
                for ingredient in disliked:
                    if ingredient in meal_name:
                        log.warning(
                            f"Plan for {name}: disliked ingredient '{ingredient}' "
                            f"found in {day_name} {meal_type}: {meal_name}"
                        )

    # Add metadata
    plan["subscriber"] = name
    plan["week_number"] = week_number
    plan["year"] = today.year
    plan["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return plan


def save_personal_plan(plan: dict, subscriber_name: str):
    """Save a personal plan to disk."""
    year = plan.get("year", datetime.date.today().year)
    week = plan.get("week_number", 1)

    plan_dir = PERSONAL_PLANS_DIR / subscriber_name
    plan_dir.mkdir(parents=True, exist_ok=True)

    path = plan_dir / f"{year}-W{week:02d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    log.info(f"Saved plan: {path}")


def run(profile_name: str | None = None, dry_run: bool = False):
    """Generate meal plans for subscriber(s)."""
    client = init_gemini()

    if profile_name:
        subscribers = [load_subscriber(profile_name)]
        subscribers[0]["_filename"] = profile_name
    else:
        subscribers = load_all_subscribers()

    log.info(f"Generating plans for {len(subscribers)} subscriber(s)")

    for profile in subscribers:
        name = profile.get("_filename", profile.get("name", "unknown").lower())
        plan = generate_personal_plan(client, profile)

        if plan:
            if dry_run:
                log.info(f"[DRY RUN] Plan for {name}:")
                print(json.dumps(plan, indent=2))
            else:
                save_personal_plan(plan, name)
        else:
            log.error(f"Failed to generate plan for {name}")


def main():
    parser = argparse.ArgumentParser(description="Personalized meal plan generator")
    parser.add_argument("--profile", type=str, help="Generate for a specific subscriber")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    run(profile_name=args.profile, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
