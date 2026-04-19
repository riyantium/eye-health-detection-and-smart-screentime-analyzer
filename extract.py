"""
extract.py — Production-grade ActivityWatch extractor
Fixes:
- Multi-bucket ingestion
- Proper timestamp parsing (timezone aware)
- Accurate hour + date extraction
- Debug visibility
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser

AW_BASE = "http://localhost:5600/api/0"


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

def is_running():
    try:
        r = requests.get(f"{AW_BASE}/info", timeout=2)
        return r.status_code == 200
    except:
        return False


# ─── FETCH BUCKETS ────────────────────────────────────────────────────────────

def get_buckets():
    try:
        r = requests.get(f"{AW_BASE}/buckets", timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return {}


# ─── CORE EXTRACTION ──────────────────────────────────────────────────────────

def get_events(days=7):
    """
    Returns DataFrame with:
    date (YYYY-MM-DD), app, minutes, hour
    """

    buckets = get_buckets()

    # 🔥 FIX 1: collect ALL window buckets
    window_buckets = [
        b for b in buckets
        if "window" in b.lower()
    ]

    if not window_buckets:
        print("❌ No window buckets found")
        return empty_df()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_events = []

    # 🔥 FIX 2: fetch ALL buckets
    for bucket in window_buckets:
        try:
            r = requests.get(
                f"{AW_BASE}/buckets/{bucket}/events",
                params={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": -1
                },
                timeout=15
            )
            r.raise_for_status()
            data = r.json()

            print(f"Bucket {bucket}: {len(data)} events")

            all_events.extend(data)

        except Exception as e:
            print(f"⚠️ Error fetching {bucket}:", e)

    print("Total raw events:", len(all_events))

    if not all_events:
        return empty_df()

    # ─── PROCESS EVENTS ──────────────────────────────────────────────────────

    rows = []

    for event in all_events:

        data = event.get("data", {})

        app = (
            data.get("app")
            or data.get("title")
            or "Unknown"
        )

        duration = event.get("duration", 0)
        timestamp = event.get("timestamp", "")

        if duration <= 0 or not timestamp:
            continue

        # 🔥 FIX 3: proper timestamp parsing
        try:
            dt = parser.isoparse(timestamp).astimezone()
            date = dt.date().isoformat()
            hour = dt.hour
        except:
            continue  # skip invalid timestamps completely

        rows.append({
            "date": date,
            "app": app,
            "minutes": duration / 60,
            "hour": hour
        })

    if not rows:
        return empty_df()

    df = pd.DataFrame(rows)

    # ─── CLEANING ────────────────────────────────────────────────────────────

    df = df[df["minutes"] > 0]

    # normalize app names
    df["app"] = df["app"].astype(str).str.strip()

    # round minutes
    df["minutes"] = df["minutes"].round(3)

    print("Final processed rows:", len(df))
    print("Total minutes (raw sum):", round(df["minutes"].sum()))

    return df


# ─── AGGREGATIONS ────────────────────────────────────────────────────────────

def get_daily_totals(df):
    if df.empty:
        return pd.DataFrame(columns=["date", "minutes"])

    return (
        df.groupby("date")["minutes"]
        .sum()
        .reset_index()
        .sort_values("date")
    )


def get_app_totals(df):
    if df.empty:
        return pd.DataFrame(columns=["app", "minutes"])

    return (
        df.groupby("app")["minutes"]
        .sum()
        .reset_index()
        .sort_values("minutes", ascending=False)
        .reset_index(drop=True)
    )


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def empty_df():
    return pd.DataFrame(columns=["date", "app", "minutes", "hour"])