#!/usr/bin/env python3
"""
Data prerequisite for EDA_PLAN section 3.4 / section 6: historical FantasyFootballCalculator
ADP, PPR, 12-team, 2015-2025. Pull only, no analysis. Raw per-year dumps; never overwritten
(skips years already on disk, matching the fetch_data.py caching convention).
"""
import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/adp/historical"
URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"
YEARS = range(2015, 2026)
SLEEP_S = 2.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        dest = OUT / f"adp_ppr_{year}.csv"
        if dest.exists():
            print(f"{year}: cached, skipping")
            continue
        req = urllib.request.Request(URL.format(year=year),
                                     headers={"User-Agent": "wr-valuation-research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        players = payload.get("players", [])
        meta = payload.get("meta", {})
        if not players:
            print(f"{year}: WARNING - empty players array, meta={meta}")
            continue
        cols = sorted({k for p in players for k in p})
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(players)
        print(f"{year}: {len(players)} players -> {dest.name} "
              f"(drafts={meta.get('total_drafts')}, start={meta.get('start_date')})")
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
