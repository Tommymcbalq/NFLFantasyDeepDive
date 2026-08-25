# Sleeper live-draft integration — research findings and build plan

## 1. What the Sleeper API actually gives us

Verified against the live endpoints today, not just the docs.

| endpoint | gives us | verified |
|---|---|---|
| `GET /v1/draft/<draft_id>` | status, type, settings, `draft_order`, `slot_to_roster_id` | yes |
| `GET /v1/draft/<draft_id>/picks` | every pick: `player_id`, `picked_by`, `roster_id`, `round`, `draft_slot`, `pick_no`, plus metadata | yes |
| `GET /v1/draft/<draft_id>/traded_picks` | `season`, `round`, `roster_id`, `previous_owner_id`, `owner_id` | yes |
| `GET /v1/league/<league_id>/drafts` | all drafts for the league | yes |
| `GET /v1/league/<league_id>/users` | user_id -> display_name | yes |
| `GET /v1/league/<league_id>/rosters` | roster_id -> owner | yes |
| `GET /v1/players/nfl` | full player index, ~12k players, 5MB | yes |

**Auth:** none. Read-only, public.
**Rate limit:** documented as "stay under 1000 calls/minute." Polling once every 5-10s is ~4 orders
of magnitude below that.
**Websocket / push:** **none documented and none found.** Polling is the only mechanism.
**Commercial use:** requires licensing from Sleeper. Personal use is fine.

### Identifiers established

```
league_id       1389519993686753280
draft_id (real) 1389519993699328000     status pre_draft, snake, 10 teams, 15 rounds
draft_id (mock) 1397557607622778880     status complete  -- useful as a replay fixture
me              timmy2tufff  /  user_id 1084179414633664512  /  roster_id 10  /  slot 5
pick timer      180s
roster          1 QB, 2 RB, 2 WR, 1 TE, 2 W/R FLEX, 1 DEF, 6 BN
```

Draft slots: 1 vcashdaddy, 2 JamesCS, 3 Crollll, 4 cosabosa, **5 timmy2tufff**, 6 WillHayes33,
7 SharksGeneralManager, 8 HouseofCB, 9 OscarDempsey, 10 drdray1.

### Player index carries an injury feed

`injury_status`, `injury_body_part`, `injury_notes`, `practice_participation`, `news_updated`.
This removes the dependence on web search for injury state. Caveat: in preseason a large number
of players carry `Questionable / Undisclosed`, which is precautionary noise. **Named body parts
are the signal.**

## 2. What is already built and working

`scripts/80_live_draft.py` — polls the draft, marks drafted players, computes picks-until-your-
next, and **re-simulates the remaining draft over the undrafted pool only**, so survival is
conditional on what has actually happened. Verified by replaying the completed mock at picks 15
and 20: it reproduces the owner's declared availabilities mid-draft while correctly collapsing
players the room has taken.

```
python3 scripts/80_live_draft.py                     # live, refresh every 8s
python3 scripts/80_live_draft.py --once              # snapshot
python3 scripts/80_live_draft.py --replay 20 \
        --draft 1397557607622778880                  # rehearse on the finished mock
```

**This is the fallback and it is done.** Everything below is UX on top of it.

## 3. Extension architecture

Manifest V3. `host_permissions` on `https://api.sleeper.app/*` means the service worker can call
the API directly — MV3 extension fetches are not subject to page CORS, so the undocumented CORS
policy does not matter.

```
manifest.json          MV3, host_permissions: api.sleeper.app, content script on sleeper.com/draft/*
content.js             reads draft_id from the URL, injects a fixed side panel
worker.js              polls /picks, runs the simulation, pushes state to the panel
board.json             the board, baked in: name, sleeper_id, tier, order, mu_cal, sig_cal, BOARD
panel.css/html         the display
```

**Two options for where the maths runs:**

**A — everything in JS (recommended).** Bake `board.json` in; port the Monte Carlo to JS. Per
refresh it is 4,000 draws over ~150 live players plus a rank — roughly 600k operations, a few
tens of milliseconds. Self-contained, nothing to run alongside, works if the laptop's Python
environment is untouched.

**B — extension as a thin client to a local Python server.** Reuses the code we already trust,
no port needed. But it requires a server running on localhost during the draft, and MV3 workers
are aggressively suspended, so it adds a failure mode during the one event where failure is
expensive.

**Recommendation: A.** With the CLI kept running in a terminal as the backstop, since it already
works and shares the same board file.

## 4. Build order

1. Export `board_LIVE_sleeper.csv` -> `board.json` (a script, so the extension can be refreshed
   whenever the board or the owner's tiers change).
2. Port `availability()` to JS. This is the only real logic: normal draws, rank, count.
3. Panel UI: your next pick, picks away, tier status, survival bars, "will not last" flags.
4. Load unpacked in Chrome (`chrome://extensions` -> developer mode -> load unpacked).
5. Validate against the mock replay before trusting it live.

## 5. What is needed from the owner

- **Nothing blocking.** draft_id, league_id, username, roster, slots, ownership and the board are
  all established.
- Preferences only: overlay panel on the Sleeper page vs a popup; and whether the panel should
  also surface injury flags from the player index.
- Chrome developer mode enabled to load an unpacked extension (it will not be on the Web Store).

## 6. Honest scoping

The extension is UX, not capability — it computes exactly what the CLI already computes. The
CLI is verified and costs nothing to run in a terminal beside the Sleeper tab. Given a 180-second
pick timer, the thing that actually matters is that the numbers are right and appear within a
second or two, and that is already true. **Build the extension after the draft, not before it.**
