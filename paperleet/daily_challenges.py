"""Versioned daily selection shared by the public app and trusted online service."""
from datetime import date, datetime, timedelta, timezone
from importlib.resources import files
import json


def schedule():
    return json.loads(files("paperleet").joinpath("daily_challenges.json").read_text())


def challenge(day=None):
    config = schedule()
    today = datetime.now(timezone.utc).date()
    selected = today if day is None else date.fromisoformat(day)
    if day is not None and selected.isoformat() != day:
        raise ValueError("Challenge date must use YYYY-MM-DD")
    offset = (selected - date.fromisoformat(config["starts_on"])).days
    if offset < 0 or selected > today:
        raise ValueError("Challenge date is outside the published calendar")
    return {"date": selected.isoformat(), "problem_id": config["modules"][offset % len(config["modules"])] ,
            "version": config["version"], "timezone": "UTC",
            "ends_at": (selected + timedelta(days=1)).isoformat() + "T00:00:00Z"}
