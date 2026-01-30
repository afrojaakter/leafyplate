# Dailyblog — Claude Code Guidelines

## Project Overview

Dailyblog is an AI-powered healthy eating platform with two components:
1. **Public blog** — Static HTML/CSS site with auto-generated recipes, articles, meal plans
2. **Personalized email service** — Weekly meal plans + shopping lists tailored per subscriber

Tech stack: Python 3.12, Google Gemini API, feedparser, BeautifulSoup, Brevo (email), GitHub Actions, GitHub Pages.

## Architecture

```
daily_batch_update.py  → Main blog content engine (RSS → AI → HTML)
meal_planner.py        → Personalized meal plan generator
product_linker.py      → Ingredient → grocery store affiliate links
send_meal_plans.py     → Personal plan email delivery (Brevo)
send_digest.py         → Weekly blog digest email (Brevo)
config.py              → All constants and environment variables
```

### Data Flow
1. `daily_batch_update.py` fetches RSS → identifies topics via Gemini → generates content → saves JSON to `data/articles/` → builds static HTML in `output/`
2. `meal_planner.py` reads subscriber profiles → generates personalized plans via Gemini → saves to `data/personal_plans/`
3. `send_meal_plans.py` reads plans → renders email HTML → sends via Brevo
4. `product_linker.py` maps ingredients to grocery store search URLs with affiliate tags

### Key Directories
- `data/articles/` — JSON metadata for each blog post (one file per post)
- `data/personal_plans/` — JSON meal plans per subscriber per week
- `output/` — Generated static site (deployed to GitHub Pages)
- `templates/` — HTML templates using `{placeholder}` string substitution
- `prompts/` — Gemini AI prompt templates
- `subscribers/profiles/` — JSON preference files per subscriber
- `products/` — Affiliate config and product catalog
- `static/css/` — Stylesheets (style.css, recipe-card.css, meal-plan.css, print.css)

## Coding Standards

### Python
- **Python 3.12+** — Use modern syntax (type hints with `|` union, `list[dict]` not `List[Dict]`)
- **Logging** — Use the `logging` module, not `print()`, for all operational output. Use `log.info()` for normal flow, `log.warning()` for recoverable issues, `log.error()` for failures
- **Error handling** — Wrap external calls (RSS feeds, Gemini API, Brevo API, HTTP requests) in try/except. Log the error and continue gracefully — never let one failed feed or API call crash the entire batch
- **JSON I/O** — Always use `encoding="utf-8"` and `ensure_ascii=False` when writing JSON
- **Paths** — Use `pathlib.Path` for all file paths. Reference directories from `config.py` constants, not hardcoded strings
- **Environment variables** — All secrets and configurable values go through `config.py` via `os.environ.get()`. Never hardcode API keys or credentials
- **Functions** — Keep functions focused. Each should do one thing. Prefer pure functions where possible
- **No classes** — This project uses a functional style (like PenguinPulse). Don't introduce classes unless truly necessary

### Templates
- Templates use `{placeholder}` syntax (Python `str.format_map()`)
- Use `render_template()` from `daily_batch_update.py` which handles missing keys gracefully
- All HTML must be valid, semantic, and accessible (use `<main>`, `<nav>`, `<article>`, `aria-label`, etc.)
- Wrap all page content with `render_full_page()` which injects the base template

### CSS
- Use CSS custom properties (variables) defined in `:root` in `style.css`
- Mobile-first responsive design with breakpoints at 768px and 480px
- Support `prefers-color-scheme: dark` via CSS variable overrides
- No JavaScript unless absolutely necessary (the blog is static HTML/CSS only)

### Content / AI
- **Voice** — Warm, encouraging, slightly irreverent. Not preachy or clinical
- **Anti-diet-culture** — Never use: "guilt-free", "skinny", "cheat meal", "clean eating", "detox"
- **Nutrition** — Always label as "estimated". Never present AI-generated nutrition as medical advice
- **Quality gates** — Reject AI-generated content that fails: < 3 ingredients, < 2 instruction steps, missing title
- **Structured output** — All Gemini calls use `response_mime_type="application/json"` and validate the result

### Affiliate Links
- `product_linker.py` handles all affiliate URL construction
- Phase 1: Search URL fallback (link to store search results page)
- Never hardcode affiliate IDs — read from environment variables via `config.py`
- Supported stores: Walmart, Amazon Fresh, Instacart, Thrive Market

## Testing

### Local Build Test
```bash
python daily_batch_update.py --rebuild    # Build site from existing data
python product_linker.py --test           # Test ingredient-to-product mapping
python meal_planner.py --profile afroj --dry-run  # Preview a personal plan
python send_meal_plans.py --dry-run       # Preview email without sending
python send_digest.py --dry-run           # Preview digest without sending
```

### Full Pipeline Test (requires GEMINI_API_KEY)
```bash
export GEMINI_API_KEY=your_key_here
python daily_batch_update.py --test       # Generate one test recipe + rebuild site
python meal_planner.py --profile afroj    # Generate a real personal plan
```

### Verify Output
- Open `output/index.html` in a browser — check layout, dark mode, mobile responsive
- Open a recipe page — check ingredients, instructions, nutrition card, shopping links, print view
- Validate `output/feed.xml` with an RSS validator
- Check `output/sitemap.xml` has correct URLs

## Common Tasks

### Add a new RSS feed source
Edit `sources.json` → add entry to `feeds` array with `id`, `name`, `url`, `category`, `priority`

### Add a new subscriber
Create a JSON file in `subscribers/profiles/` matching the schema in `subscribers/schema.json`

### Add a new content type
1. Add a new prompt in `prompts/`
2. Add a generate function in `daily_batch_update.py`
3. Add a template in `templates/`
4. Add URL prefix in `config.py` `URL_PREFIXES`
5. Add badge styles in `style.css`

### Change the visual design
Edit CSS variables in `:root` in `static/css/style.css`. The entire color scheme, typography, and spacing are controlled by variables.

## Do NOT
- Introduce JavaScript to the blog (keep it static HTML/CSS)
- Add Python classes (functional style throughout)
- Hardcode API keys, affiliate IDs, or file paths
- Use `print()` for operational logging
- Generate diet culture language in prompts
- Skip error handling on external API calls
- Modify the `output/` directory manually (it's auto-generated)
