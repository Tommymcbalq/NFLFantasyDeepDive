"""§I1/§I4 Vegas sourcing pull. Run: python3 scripts/fetch_vegas.py

Historical team win totals, 2015-2025, all 32 teams/season, from the
Covers.com SportsOddsHistory archive (public page, permitted by robots.txt).
Each season page carries an "As of <early September>" stamp = pre-Week-1 number.

Cross-check source: nflverse/nfldata data/win_totals.csv (2003-2020 only).
"""
import csv, html, os, re, time, urllib.request

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "vegas")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")


def covers_win_totals(seasons=range(2015, 2026)):
    rows = []
    for y in seasons:
        s = get(f"https://www.covers.com/sportsoddshistory/nfl-win/?y={y}&sa=nfl&t=win")
        m = re.search(r"as of ([A-Z][a-z]+ \d+, \d{4})", s, re.I)
        asof = m.group(1) if m else ""
        tbl = re.search(r"<table.*?</table>", s, re.S)
        if not tbl:
            print(f"{y}: no table"); continue
        n = 0
        for tr in re.findall(r"<tr.*?</tr>", tbl.group(0), re.S):
            c = [html.unescape(re.sub("<[^>]+>", "", x)).strip()
                 for x in re.findall(r"<t[hd].*?</t[hd]>", tr, re.S)]
            if len(c) >= 6 and re.match(r"^[\d.]+$", c[1]):
                rows.append(dict(season=y, team=c[0], win_total=c[1], over=c[2],
                                 under=c[3], actual_wins=c[5], asof=asof))
                n += 1
        print(f"{y}: {n} teams, as of {asof}")
        time.sleep(1)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = covers_win_totals()
    p = os.path.join(OUT, "team_win_totals_2015_2025_covers.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")
