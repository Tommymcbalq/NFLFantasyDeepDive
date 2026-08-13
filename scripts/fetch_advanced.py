#!/usr/bin/env python3
"""
fetch_advanced.py — idempotent downloader for the advanced-stats layer.

Every asset comes from nflverse-data GitHub releases (public, no auth, direct HTTP;
the release URL 302s to S3). Files land in data/advanced/<group>/ and are NEVER
overwritten: if a non-empty file already exists at the destination it is skipped.
Use --force to re-pull (writes to a .new temp then atomically replaces).

Usage:
    python3 scripts/fetch_advanced.py                # pull anything missing
    python3 scripts/fetch_advanced.py --force        # re-pull everything
    python3 scripts/fetch_advanced.py --list         # show plan, download nothing
    python3 scripts/fetch_advanced.py --build        # pull, then run build_advanced.py

Season coverage:
    PFR advanced stats  2018+ (source starts 2018)
    NGS                 2016+ (single all-season file per phase)
    snap counts         2012+ (we take 2018-2025; 2014-2025 already cached in data/snap_counts)
    play-by-play        1999+ (we take 2018-2025; parquet, ~20MB/season)
    participation       2016-2025 (offense/defense players on field, parquet)
    FTN charting        2022+
    ESPN QBR            2006+ (single file)
"""
import argparse
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADV = os.path.join(ROOT, "data", "advanced")
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

PBP_SEASONS = range(2018, 2026)
PART_SEASONS = range(2018, 2026)
SNAP_SEASONS = range(2018, 2026)
FTN_SEASONS = range(2022, 2026)


def plan():
    """(subdir, release_tag, asset_name) triples."""
    out = []
    for f in ("rec", "rush", "pass"):
        out.append(("pfr", "pfr_advstats", f"advstats_season_{f}.csv"))
    for f in ("receiving", "rushing", "passing"):
        out.append(("ngs", "nextgen_stats", f"ngs_{f}.csv.gz"))
    out.append(("espn", "espn_data", "qbr_season_level.csv"))
    out.append(("players", "players", "players.csv"))
    for y in PBP_SEASONS:
        out.append(("pbp", "pbp", f"play_by_play_{y}.parquet"))
    for y in PART_SEASONS:
        out.append(("participation", "pbp_participation", f"pbp_participation_{y}.parquet"))
    for y in SNAP_SEASONS:
        out.append(("snap_counts", "snap_counts", f"snap_counts_{y}.csv"))
    for y in FTN_SEASONS:
        out.append(("ftn", "ftn_charting", f"ftn_charting_{y}.csv"))
    return out


def fetch(subdir, tag, asset, force=False):
    dest_dir = os.path.join(ADV, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, asset)
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        return "cached", dest
    url = f"{BASE}/{tag}/{asset}"
    tmp = dest + ".new"
    req = urllib.request.Request(url, headers={"User-Agent": "nflverse-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if os.path.getsize(tmp) == 0:
        os.remove(tmp)
        raise RuntimeError(f"empty download: {url}")
    os.replace(tmp, dest)
    return "fetched", dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()

    items = plan()
    if a.list:
        for s, t, n in items:
            print(f"{s:14s} {BASE}/{t}/{n}")
        return

    n_new = n_cached = 0
    for s, t, n in items:
        try:
            status, dest = fetch(s, t, n, force=a.force)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {s}/{n}: {e}", file=sys.stderr)
            continue
        n_new += status == "fetched"
        n_cached += status == "cached"
        print(f"{status:8s} {os.path.relpath(dest, ROOT)} "
              f"({os.path.getsize(dest)/1e6:.1f} MB)")
    print(f"\n{n_new} fetched, {n_cached} already cached -> {os.path.relpath(ADV, ROOT)}")

    if a.build:
        for step in ("build_advanced.py", "document_advanced.py"):
            subprocess.check_call([sys.executable,
                                   os.path.join(ROOT, "scripts", step)])


if __name__ == "__main__":
    main()
