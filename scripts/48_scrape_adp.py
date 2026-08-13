"""Render JS-only ADP pages with Playwright and save them as CSV.

Respects robots.txt: FantasyPros disallows /api/, /json/, /ajax/ and /nfl/ranker/ — we only
load the public /nfl/adp/*.php pages, and honour the 5s crawl-delay between requests.
"""
import re, sys, time, pandas as pd
from playwright.sync_api import sync_playwright

TARGETS = [
    ("fantasypros_ppr_overall", "https://www.fantasypros.com/nfl/adp/ppr-overall.php"),
]
CRAWL_DELAY = 5

def scrape(pw, url):
    b = pw.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")
    pg.goto(url, wait_until="networkidle", timeout=60000)
    try:
        pg.wait_for_selector("table tbody tr", timeout=20000)
    except Exception:
        pass
    html = pg.content()
    b.close()
    return html

def tables(html):
    try:
        return pd.read_html(html)
    except Exception as e:
        print("  read_html failed:", e); return []

if __name__ == "__main__":
    with sync_playwright() as pw:
        for name, url in TARGETS:
            print(f"== {name}\n   {url}")
            html = scrape(pw, url)
            open(f"/tmp/{name}.html", "w").write(html)
            ts = tables(html)
            print(f"   tables found: {len(ts)}")
            for i, t in enumerate(ts):
                print(f"   [{i}] shape={t.shape} cols={list(t.columns)[:8]}")
                if t.shape[0] > 30 and t.shape[1] >= 3:
                    t.to_csv(f"data/adp/{name}_raw.csv", index=False)
                    print(f"   -> saved data/adp/{name}_raw.csv")
                    break
            time.sleep(CRAWL_DELAY)
