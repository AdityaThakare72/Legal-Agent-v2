"""Scraper for Nyaaya legal explainers (https://nyaaya.org).

Improvement over the v1 scraper (which walked a handful of category pages and
found ~68 articles): this version enumerates the full set of explainers from
Nyaaya's WordPress sitemap (``legal_explainer-sitemap.xml``), which lists ~869
explainer URLs, then fetches and cleans each one.

Politeness: a configurable delay (default 1.5s) between requests and a proper
User-Agent header. Output is written incrementally to CSV so a run can be
resumed, with columns matching what the standardizer expects
(``title``, ``url``, ``content``, ``category``, ``scraped_at``).

Usage:
    python data_pipeline/scrapers/nyaaya_scraper.py                 # scrape all
    python data_pipeline/scrapers/nyaaya_scraper.py --limit 20      # small test
    python data_pipeline/scrapers/nyaaya_scraper.py --out path.csv  # custom output
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(_REPO_ROOT, "data_pipeline", "data", "raw", "nyaaya_explainers.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nyaaya_scraper")

SITEMAP_URLS = [
    "https://nyaaya.org/legal_explainer-sitemap.xml",
    "https://nyaaya.org/sitemap_index.xml",  # fallback: discover explainer sitemap
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LegalAgentBot/1.0; +https://example.org/research)"
}
REQUEST_TIMEOUT = 25
DELAY_SECONDS = 1.5
CSV_FIELDS = ["title", "url", "content", "category", "scraped_at"]


def _get(url: str) -> requests.Response:
    """GET with headers, raising for HTTP errors."""
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def get_explainer_urls() -> list[str]:
    """Enumerate all explainer URLs from the Nyaaya sitemap(s)."""
    for sitemap in SITEMAP_URLS:
        try:
            text = _get(sitemap).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch %s: %s", sitemap, exc)
            continue
        locs = re.findall(r"<loc>(.*?)</loc>", text)
        explainers = [u for u in locs if "/legal-explainer/" in u]
        if explainers:
            logger.info("Found %d explainer URLs via %s", len(explainers), sitemap)
            return sorted(set(explainers))
        # If this was the index, recurse into an explainer sitemap it points to.
        for loc in locs:
            if "legal_explainer" in loc and loc not in SITEMAP_URLS:
                try:
                    sub = _get(loc).text
                except Exception:  # noqa: BLE001
                    continue
                explainers = [u for u in re.findall(r"<loc>(.*?)</loc>", sub) if "/legal-explainer/" in u]
                if explainers:
                    logger.info("Found %d explainer URLs via %s", len(explainers), loc)
                    return sorted(set(explainers))
    logger.error("No explainer URLs could be discovered from any sitemap.")
    return []


def _clean(text: str) -> str:
    """Collapse whitespace in extracted text."""
    return re.sub(r"\s+", " ", text or "").strip()


# Leading breadcrumb like "Home / Legal Explainers / Family / <Title> " — strip
# everything up to and including the last "/ " so the body starts at real content.
_BREADCRUMB_RE = re.compile(r"^\s*Home\s*/\s*(?:[^/]+/\s*)+", re.IGNORECASE)


def _strip_breadcrumb(text: str) -> tuple[str, str]:
    """Return (content_without_breadcrumb, category_from_breadcrumb)."""
    match = _BREADCRUMB_RE.match(text)
    if not match:
        return text, ""
    trail = match.group(0)
    parts = [p.strip() for p in trail.split("/") if p.strip()]
    # parts[0]="Home", parts[1]="Legal Explainers", parts[2]=top category.
    category = parts[2] if len(parts) >= 3 else ""
    return text[match.end():].strip(), category


def scrape_article(url: str) -> dict | None:
    """Fetch and parse a single explainer page.

    Returns a dict with title, url, content, category, scraped_at — or None on
    failure.
    """
    try:
        soup = BeautifulSoup(_get(url).content, "html.parser")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    # Drop obvious non-content chrome so it doesn't pollute the text.
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()
    for sel in ["nav", ".breadcrumb", ".breadcrumbs", ".site-header", ".menu"]:
        for el in soup.select(sel):
            el.decompose()

    title_el = soup.find("h1")
    title = _clean(title_el.get_text()) if title_el else ""

    # Main body: prefer the prose container, then article/main, then #primary.
    body_el = (
        soup.select_one("div.prose")
        or soup.find("article")
        or soup.select_one("main#primary")
        or soup.find("main")
    )
    content = _clean(body_el.get_text(" ")) if body_el else ""

    # Strip the leading "Home / Legal Explainers / <Category> / ... / Title"
    # breadcrumb, capturing the top-level category from it.
    content, category = _strip_breadcrumb(content)

    # Append FAQ Q/A pairs if present (accordion blocks).
    faqs = []
    for item in soup.select("div.accordion-block"):
        q = item.select_one(".accordion-block__title")
        a = item.select_one(".accordion-block__content")
        if q and a:
            faqs.append(f"Q: {_clean(q.get_text())} A: {_clean(a.get_text())}")
    if faqs:
        content = f"{content}\n\nFAQs:\n" + "\n".join(faqs)

    if not content:
        logger.warning("No content extracted from %s", url)
        return None

    return {
        "title": title or url.rstrip("/").split("/")[-1].replace("-", " ").title(),
        "url": url,
        "content": content,
        "category": category or "General",
        "scraped_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _already_scraped(out_path: str) -> set[str]:
    """Return URLs already present in the output CSV (for resume)."""
    if not os.path.exists(out_path):
        return set()
    done: set[str] = set()
    with open(out_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("url"):
                done.add(row["url"])
    return done


def scrape(out_path: str = DEFAULT_OUT, limit: int | None = None, delay: float = DELAY_SECONDS) -> int:
    """Scrape explainers to CSV. Returns the number of new rows written."""
    urls = get_explainer_urls()
    if not urls:
        return 0
    if limit:
        urls = urls[:limit]

    done = _already_scraped(out_path)
    is_new_file = not os.path.exists(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    failures = 0
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        for i, url in enumerate(urls, start=1):
            if url in done:
                continue
            record = scrape_article(url)
            if record:
                writer.writerow(record)
                fh.flush()
                written += 1
            else:
                failures += 1
            if i % 25 == 0:
                logger.info("Progress: %d/%d (%d written, %d failed)", i, len(urls), written, failures)
            time.sleep(delay)

    logger.info("Done. Wrote %d new rows to %s (%d failures).", written, out_path, failures)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Nyaaya legal explainers.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Max pages (for testing).")
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS, help="Delay between requests (s).")
    args = parser.parse_args()
    scrape(out_path=args.out, limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
