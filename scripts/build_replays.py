"""MAINTAINER TOOL (not needed to run the app): rebuild data/replays from TracingInsights data.

The app ships with every race already converted, so normal users never run this.

Usage (needs internet):
    python scripts/build_replays.py --download 2019 2020 2021 2022 2023 2024 2025 2026
or from a folder of raw files laid out as <raw>/<year>/<Event name>/{session_laptimes.json,drivers.json}:
    python scripts/build_replays.py --raw path/to/raw

Excluded on purpose:
  * 2018: the source's tyre/stint data doesn't match the real pit stops.
  * races whose files lack driver codes, or with under 200 laps of timing data (e.g. Belgium 2021).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.data_sources.fastf1_source import DataSourceError  # noqa: E402
from pitwall.data_sources.replays import INDEX_PATH, RACES_DIR, SOURCE_TEXT, convert_raw, race_id  # noqa: E402

RAW_URL = "https://raw.githubusercontent.com/TracingInsights/{year}/main/{event}/Race/{file}"
EXCLUDED_YEARS = {2018}


def _get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def iter_raw_folder(raw: Path):
    for f in sorted(raw.glob("*/*/session_laptimes.json")):
        yield int(f.parts[-3]), f.parent.name, (lambda f=f: (json.loads(f.read_text(encoding="utf-8")),
                                                            json.loads((f.parent / "drivers.json").read_text(encoding="utf-8"))))


def build(items, out_dir: Path = RACES_DIR, index_path: Path = INDEX_PATH):
    out_dir.mkdir(parents=True, exist_ok=True)
    races, skipped = [], []
    for year, event, loader in items:
        if year in EXCLUDED_YEARS:
            skipped.append(f"{year} {event}: year excluded (unreliable tyre data)")
            continue
        try:
            laps, drivers = loader()
            snap = convert_raw(year, event, laps, drivers)
        except (DataSourceError, OSError, ValueError) as e:
            skipped.append(f"{year} {event}: {e}")
            continue
        rid = race_id(year, event)
        (out_dir / f"{rid}.json").write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
        races.append({"id": rid, "year": year, "event": event, "date": snap["meta"]["date"],
                      "laps": snap["meta"]["laps_winner"], "drivers": len(snap["drivers"])})
    index = {"source": SOURCE_TEXT, "licence": "MIT (TracingInsights archive)",
             "races": sorted(races, key=lambda r: (r["year"], r["date"] or "", r["event"])), "skipped": skipped}
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="folder of raw files <year>/<event>/...")
    ap.add_argument("--download", nargs="*", type=int, help="seasons to download (needs the event list file)")
    ap.add_argument("--events", type=Path, help="TSV of '<year>\\t<Event name>' rows to download")
    a = ap.parse_args()
    if a.raw:
        idx = build(iter_raw_folder(a.raw))
    elif a.events:
        rows = [line.split("\t") for line in a.events.read_text(encoding="utf-8").splitlines() if line.strip()]

        def items():
            for y, ev in rows:
                if a.download and int(y) not in a.download:
                    continue
                q = urllib.parse.quote(ev)
                yield int(y), ev, (lambda y=y, q=q: (_get(RAW_URL.format(year=y, event=q, file="session_laptimes.json")),
                                                     _get(RAW_URL.format(year=y, event=q, file="drivers.json"))))
        idx = build(items())
    else:
        ap.error("give --raw FOLDER or --events FILE")
    print(f"{len(idx['races'])} races written, {len(idx['skipped'])} skipped")
    for s in idx["skipped"]:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
