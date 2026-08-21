from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import csv
import io
import json
import math
import re
import zipfile

import numpy as np

from .cgm import CGMObservationConfig, CGMObservationModel
from .config import HumanConfig
from .metabolism_dallaman import DallaManMealModel
from .physiology import HumanState


@dataclass(frozen=True)
class DuBose2020Reference:
    """Published event-level CGM benchmarks from DuBose et al. 2020/2021.

    These targets describe free-living healthy participants wearing blinded Dexcom G6.
    Meal logs included start time but did not capture carbohydrate/fat quantity, which
    creates an unavoidable latent-input identifiability problem for mechanistic replay.
    """

    n_participants: int = 153
    exercise_sessions: int = 451
    exercise_baseline_mean_mg_dl: float = 99.0
    exercise_baseline_sd_mg_dl: float = 12.0
    exercise_nadir_mean_mg_dl: float = 85.0
    exercise_nadir_sd_mg_dl: float = 11.0
    exercise_change_mean_mg_dl: float = -15.0
    exercise_change_sd_mg_dl: float = 18.0
    exercise_duration_median_min: float = 45.0
    exercise_duration_iqr_min: tuple[float, float] = (30.0, 60.0)

    meal_participants: int = 56
    meal_events: int = 306
    meal_premeal_mean_mg_dl: float = 93.0
    meal_premeal_sd_mg_dl: float = 10.0
    meal_peak_mean_mg_dl: float = 130.0
    meal_peak_sd_mg_dl: float = 13.0
    meal_time_to_peak_mean_min: float = 97.0
    meal_time_to_peak_sd_min: float = 31.0
    meal_excursion_mean_mg_dl: float = 37.0
    meal_excursion_sd_mg_dl: float = 15.0

    citation: str = (
        "DuBose SN et al. Effect of Exercise and Meals on Continuous Glucose "
        "Monitor Data in Healthy Individuals Without Diabetes. J Diabetes Sci Technol. "
        "2021;15(3):593-599. DOI:10.1177/1932296820905904"
    )

    def as_dict(self) -> dict:
        return asdict(self)


DUBOSE_REFERENCE = DuBose2020Reference()


@dataclass(frozen=True)
class FreeLivingReplayProfile:
    """Event-replay calibration that is separate from the published Dalla Man core.

    The default values are fitted to the *aggregate* DuBose meal/exercise benchmarks,
    not to individual participant data.  They must therefore not be interpreted as
    measured meal carbohydrate or exercise intensity.
    """

    effective_meal_carbs_g: float = 36.0
    gastric_absorption_scale: float = 0.45
    insulin_sensitivity_scale: float = 1.0
    representative_exercise_intensity: float = 0.50
    exercise_vmax_gain: float = 3.65
    cgm_lag_tau_min: float = 6.0

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_REPLAY_PROFILE = FreeLivingReplayProfile()


@dataclass(frozen=True)
class EventRecord:
    participant_id: str
    kind: str
    start: datetime
    end: datetime | None = None
    subtype: str | None = None
    duration_min: float | None = None
    reported_carbs_g: float | None = None
    source_file: str | None = None
    source_row: int | None = None
    source_field: str | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"] = None if self.end is None else self.end.isoformat()
        return out


@dataclass(frozen=True)
class EventMetric:
    participant_id: str
    kind: str
    start: datetime
    baseline_mg_dl: float
    peak_mg_dl: float | None = None
    nadir_mg_dl: float | None = None
    excursion_mg_dl: float | None = None
    time_to_peak_min: float | None = None
    n_pre: int = 0
    n_post: int = 0
    source_file: str | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["start"] = self.start.isoformat()
        return out


_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M %p",
    "%d/%m/%Y %H:%M",
    "%d-%b-%Y %H:%M",
)
_TIME_ONLY_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p")
_DATE_ONLY_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d-%b-%Y")


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _to_float(value) -> float | None:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _parse_datetime(value: str | None) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    # ISO parser handles timezone offsets too; convert to naive local wall time because
    # diary and CGM timestamps in one participant must share the same study clock.
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Excel serial datetimes occur in some exported research spreadsheets.
    x = _to_float(s)
    if x is not None and 20000.0 <= x <= 80000.0:
        return datetime(1899, 12, 30) + timedelta(days=x)
    return None


def _parse_date(value: str | None) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    dt = _parse_datetime(s)
    if dt is not None:
        return datetime(dt.year, dt.month, dt.day)
    for fmt in _DATE_ONLY_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _combine_date_time(date_value: str | None, time_value: str | None) -> datetime | None:
    direct = _parse_datetime(time_value)
    if direct is not None and direct.year > 1900:
        return direct
    date = _parse_date(date_value)
    if date is None:
        return None
    t = str(time_value or "").strip()
    if not t:
        return None
    for fmt in _TIME_ONLY_FORMATS:
        try:
            tm = datetime.strptime(t, fmt).time()
            return datetime.combine(date.date(), tm)
        except ValueError:
            continue
    x = _to_float(t)
    if x is not None and 0.0 <= x < 1.0:
        return date + timedelta(days=x)
    return None


def _read_csv_bytes(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return list(reader.fieldnames or []), list(reader)


def archive_schema_report(archive: str | Path, sample_rows: int = 2) -> dict:
    """Return a non-destructive schema inventory for every CSV in a ZIP."""
    archive = Path(archive)
    tables = []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            try:
                fields, rows = _read_csv_bytes(zf.read(name))
            except Exception as exc:
                tables.append({"file": name, "error": repr(exc)})
                continue
            tables.append({
                "file": name,
                "n_rows": len(rows),
                "fields": fields,
                "sample": rows[: max(0, int(sample_rows))],
                "event_score": int(_event_table_score(name, fields, rows[:50])),
                "cgm_score": int(_cgm_table_score(name, fields, rows[:50])),
            })
    return {"archive": str(archive), "tables": tables}


def _subject_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if any(x in n for x in ("participant", "subject", "patient", "person")):
            score += 6
        if n in {"id", "pid", "subjectid", "participantid", "ptid"} or n.endswith("subjectid"):
            score += 4
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _date_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if n in {"date", "studydate", "logdate", "eventdate"}:
            score += 7
        if "date" in n:
            score += 4
        if "time" in n or "datetime" in n or "timestamp" in n:
            score -= 2
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _timestamp_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if n in {"timestamp", "datetime", "eventdatetime", "readingdatetime", "devicetimestamp"}:
            score += 9
        if "datetime" in n or "timestamp" in n:
            score += 7
        if "time" in n and "duration" not in n:
            score += 3
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _glucose_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if "glucose" in n:
            score += 9
        if "sensor" in n or "cgm" in n:
            score += 4
        if n in {"sg", "sgv", "sensorglucose", "glucosemgdl"}:
            score += 5
        if "mgdl" in n:
            score += 2
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _event_table_score(name: str, fields: list[str], rows: list[dict[str, str]]) -> int:
    text = " ".join([name] + fields).lower()
    score = 0
    for token in ("meal", "breakfast", "lunch", "dinner", "snack", "exercise", "activity", "sleep", "wake", "alcohol", "log", "diary"):
        if token in text:
            score += 3
    if _subject_column(fields):
        score += 2
    if _timestamp_column(fields) or _date_column(fields):
        score += 2
    # Long-format event values may carry the semantics instead of headers.
    values = " ".join(str(v).lower() for r in rows[:20] for v in r.values())
    if any(t in values for t in ("breakfast", "lunch", "dinner", "exercise", "aerobic", "resistance")):
        score += 5
    return score


def _cgm_table_score(name: str, fields: list[str], rows: list[dict[str, str]]) -> int:
    score = 0
    if _glucose_column(fields):
        score += 10
    if _subject_column(fields):
        score += 2
    if _timestamp_column(fields):
        score += 4
    if "cgm" in name.lower() or "sensor" in name.lower() or "glucose" in name.lower():
        score += 4
    return score


def _event_kind_from_text(value: str | None) -> tuple[str | None, str | None]:
    raw = str(value or "").strip()
    n = _norm(raw)
    if not n:
        return None, None
    if "breakfast" in n:
        return "meal", "breakfast"
    if "lunch" in n:
        return "meal", "lunch"
    if "dinner" in n or "supper" in n:
        return "meal", "dinner"
    if "snack" in n:
        return "snack", "snack"
    if "exercise" in n or "aerobic" in n or "resistance" in n or "activity" in n or "workout" in n:
        subtype = "aerobic" if "aerobic" in n else "resistance" if "resistance" in n else None
        return "exercise", subtype
    if "sleep" in n or "bedtime" in n or "bed" == n:
        return "sleep", None
    if "wake" in n:
        return "wake", None
    if "alcohol" in n or "drink" in n:
        return "alcohol", None
    if n in {"meal", "food"}:
        return "meal", None
    return None, None


def _duration_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if "duration" in n:
            score += 8
        if "minutes" in n or n.endswith("min"):
            score += 3
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _type_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if n in {"type", "eventtype", "activitytype", "exercisetype", "mealtype"}:
            score += 8
        if "type" in n or "event" in n or "activity" in n:
            score += 3
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def _carb_column(fields: Iterable[str]) -> str | None:
    scored = []
    for f in fields:
        n = _norm(f)
        score = 0
        if "carb" in n or "cho" == n:
            score += 8
        if "gram" in n:
            score += 2
        scored.append((score, f))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def extract_events_from_archive(archive: str | Path) -> dict:
    """Best-effort extraction of meal/exercise/sleep events from a study ZIP.

    The Jaeb release schema is not hardcoded because releases can change.  The
    function supports both long event tables and common wide daily-log layouts and
    returns a detection report so a local user can audit every extracted event.
    """
    archive = Path(archive)
    events: list[EventRecord] = []
    table_reports = []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            try:
                fields, rows = _read_csv_bytes(zf.read(name))
            except Exception as exc:
                continue
            score = _event_table_score(name, fields, rows[:50])
            if score <= 0:
                continue
            before = len(events)
            scol = _subject_column(fields)
            dcol = _date_column(fields)
            tscol = _timestamp_column(fields)
            tcol = _type_column(fields)
            durcol = _duration_column(fields)
            carbcol = _carb_column(fields)

            # Long-format event rows.  A wide daily log may have "Exercise Type"
            # plus several unrelated time columns; do not mistake that for a generic
            # event-type/timestamp table.
            long_type_name = _norm(tcol) if tcol else ""
            long_mode = bool(tcol and tscol and ("event" in long_type_name or long_type_name in {"type", "activity"}))
            if long_mode:
                for idx, row in enumerate(rows):
                    kind, subtype = _event_kind_from_text(row.get(tcol))
                    if kind is None:
                        continue
                    dt = _parse_datetime(row.get(tscol))
                    if dt is None and dcol and dcol != tscol:
                        dt = _combine_date_time(row.get(dcol), row.get(tscol))
                    if dt is None:
                        continue
                    duration = _to_float(row.get(durcol, "")) if durcol else None
                    sid = str(row.get(scol, "all")).strip() if scol else "all"
                    carbs = _to_float(row.get(carbcol, "")) if carbcol else None
                    end = dt + timedelta(minutes=duration) if duration and duration > 0 else None
                    events.append(EventRecord(sid or "all", kind, dt, end, subtype, duration, carbs, name, idx + 2, tcol))

            # Wide daily logs, e.g. Breakfast Time / Exercise Start / Sleep Time.
            for idx, row in enumerate(rows):
                sid = str(row.get(scol, "all")).strip() if scol else "all"
                base_date = row.get(dcol) if dcol else None
                for field in fields:
                    kind, subtype = _event_kind_from_text(field)
                    if kind is None:
                        continue
                    fn = _norm(field)
                    if not any(x in fn for x in ("time", "start", "bed", "wake", "breakfast", "lunch", "dinner", "snack")):
                        continue
                    cell = row.get(field, "")
                    if not str(cell).strip():
                        continue
                    dt = _parse_datetime(cell)
                    if dt is None or dt.year <= 1900:
                        dt = _combine_date_time(base_date, cell)
                    if dt is None:
                        continue
                    # Search related duration/type/carb fields on this row.
                    duration = None
                    carbs = None
                    local_subtype = subtype
                    stem = re.sub(r"(time|start|datetime|date)$", "", fn)
                    for f2 in fields:
                        n2 = _norm(f2)
                        if stem and stem not in n2:
                            continue
                        if duration is None and "duration" in n2:
                            duration = _to_float(row.get(f2))
                        if carbs is None and ("carb" in n2 or "cho" == n2):
                            carbs = _to_float(row.get(f2))
                        if local_subtype is None and "type" in n2:
                            text = str(row.get(f2, "")).strip()
                            if text:
                                local_subtype = text
                    end = dt + timedelta(minutes=duration) if duration and duration > 0 else None
                    events.append(EventRecord(sid or "all", kind, dt, end, local_subtype, duration, carbs, name, idx + 2, field))

            table_reports.append({"file": name, "score": score, "n_extracted": len(events) - before, "fields": fields})

    # De-duplicate events generated by a table that satisfies both long and wide rules.
    dedup: dict[tuple, EventRecord] = {}
    for e in events:
        key = (e.participant_id, e.kind, e.start, e.subtype or "")
        old = dedup.get(key)
        if old is None or (old.duration_min is None and e.duration_min is not None):
            dedup[key] = e
    events = sorted(dedup.values(), key=lambda e: (e.participant_id, e.start, e.kind))
    return {
        "archive": str(archive),
        "n_events": len(events),
        "counts": {k: sum(e.kind == k for e in events) for k in sorted({e.kind for e in events})},
        "events": [e.as_dict() for e in events],
        "tables": table_reports,
        "caveat": (
            "Automatic schema discovery must be reviewed locally against the Jaeb ReadMe/data dictionary. "
            "The original study logs recorded event timing/type but not meal carbohydrate or fat quantity."
        ),
    }


def load_cgm_series_from_archive(archive: str | Path) -> dict[str, list[tuple[datetime, float]]]:
    """Load the highest-scoring timestamped CGM CSV from an archive."""
    archive = Path(archive)
    candidates = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                fields, rows = _read_csv_bytes(zf.read(name))
            except Exception:
                continue
            score = _cgm_table_score(name, fields, rows[:50])
            gcol, scol, tscol, dcol = _glucose_column(fields), _subject_column(fields), _timestamp_column(fields), _date_column(fields)
            if not gcol or not tscol:
                continue
            parsed = []
            for row in rows:
                g = _to_float(row.get(gcol))
                if g is None or not (20.0 <= g <= 600.0):
                    continue
                dt = _parse_datetime(row.get(tscol))
                if dt is None and dcol and dcol != tscol:
                    dt = _combine_date_time(row.get(dcol), row.get(tscol))
                if dt is None:
                    continue
                sid = str(row.get(scol, "all")).strip() if scol else "all"
                parsed.append((sid or "all", dt, float(g)))
            if parsed:
                candidates.append((score, len(parsed), name, parsed, {"glucose": gcol, "subject": scol, "timestamp": tscol}))
    if not candidates:
        raise ValueError("No timestamped CGM table could be detected in the archive")
    score, _, name, parsed, columns = max(candidates, key=lambda x: (x[0], x[1]))
    groups: dict[str, list[tuple[datetime, float]]] = {}
    for sid, dt, g in parsed:
        groups.setdefault(sid, []).append((dt, g))
    for sid in groups:
        groups[sid].sort(key=lambda x: x[0])
    return {"file": name, "columns": columns, "series": groups, "score": score}


def _window_values(series: list[tuple[datetime, float]], start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    return [(t, g) for t, g in series if start <= t <= end]


def event_metrics_from_archive(archive: str | Path) -> dict:
    """Align automatically detected logs to CGM and calculate event metrics."""
    event_payload = extract_events_from_archive(archive)
    events = [
        EventRecord(
            participant_id=e["participant_id"], kind=e["kind"], start=datetime.fromisoformat(e["start"]),
            end=datetime.fromisoformat(e["end"]) if e["end"] else None, subtype=e.get("subtype"),
            duration_min=e.get("duration_min"), reported_carbs_g=e.get("reported_carbs_g"),
            source_file=e.get("source_file"), source_row=e.get("source_row"), source_field=e.get("source_field"),
        ) for e in event_payload["events"]
    ]
    cgm_payload = load_cgm_series_from_archive(archive)
    cgm = cgm_payload["series"]

    # Index meals for the exercise exclusion rule and day-level meal completeness rule.
    by_subject_events: dict[str, list[EventRecord]] = {}
    for e in events:
        by_subject_events.setdefault(e.participant_id, []).append(e)

    meal_metrics: list[EventMetric] = []
    exercise_metrics: list[EventMetric] = []
    exclusions = {"meal": {}, "exercise": {}}

    for sid, evs in by_subject_events.items():
        series = cgm.get(sid)
        if not series:
            # Some public archives use a global table with IDs stored as numerics vs strings.
            continue
        meals = [e for e in evs if e.kind in {"meal", "snack", "alcohol"}]
        main_meals = [e for e in evs if e.kind == "meal" and e.subtype in {"breakfast", "lunch", "dinner"}]
        days: dict[datetime.date, list[EventRecord]] = {}
        for e in main_meals:
            days.setdefault(e.start.date(), []).append(e)

        for e in evs:
            if e.kind == "meal":
                day = days.get(e.start.date(), [])
                types = {m.subtype for m in day}
                if not {"breakfast", "lunch", "dinner"}.issubset(types):
                    exclusions["meal"]["day_without_all_three_main_meals"] = exclusions["meal"].get("day_without_all_three_main_meals", 0) + 1
                    continue
                near = [m for m in meals if m is not e and abs((m.start - e.start).total_seconds()) < 4 * 3600]
                if near:
                    exclusions["meal"]["other_meal_or_alcohol_within_4h"] = exclusions["meal"].get("other_meal_or_alcohol_within_4h", 0) + 1
                    continue
                pre = _window_values(series, e.start - timedelta(minutes=15), e.start)
                post = _window_values(series, e.start, e.start + timedelta(minutes=240))
                if not pre or not any(t >= e.start + timedelta(minutes=60) for t, _ in post):
                    exclusions["meal"]["insufficient_cgm"] = exclusions["meal"].get("insufficient_cgm", 0) + 1
                    continue
                baseline = float(np.mean([g for _, g in pre]))
                tpeak, peak = max(post, key=lambda x: x[1])
                meal_metrics.append(EventMetric(sid, "meal", e.start, baseline, peak_mg_dl=float(peak),
                                                excursion_mg_dl=float(peak - baseline),
                                                time_to_peak_min=float((tpeak - e.start).total_seconds() / 60.0),
                                                n_pre=len(pre), n_post=len(post), source_file=e.source_file))

            elif e.kind == "exercise":
                recent_meal = [m for m in meals if timedelta(0) <= e.start - m.start <= timedelta(minutes=30)]
                if recent_meal:
                    exclusions["exercise"]["meal_or_snack_within_30min"] = exclusions["exercise"].get("meal_or_snack_within_30min", 0) + 1
                    continue
                duration = e.duration_min or DUBOSE_REFERENCE.exercise_duration_median_min
                end = e.end or e.start + timedelta(minutes=duration)
                pre = _window_values(series, e.start - timedelta(minutes=15), e.start)
                during = _window_values(series, e.start, end)
                if not pre or not during:
                    exclusions["exercise"]["insufficient_cgm"] = exclusions["exercise"].get("insufficient_cgm", 0) + 1
                    continue
                baseline = float(np.mean([g for _, g in pre]))
                _, nadir = min(during, key=lambda x: x[1])
                exercise_metrics.append(EventMetric(sid, "exercise", e.start, baseline,
                                                    nadir_mg_dl=float(nadir), excursion_mg_dl=float(nadir - baseline),
                                                    n_pre=len(pre), n_post=len(during), source_file=e.source_file))

    return {
        "archive": str(archive),
        "cgm_table": {k: v for k, v in cgm_payload.items() if k != "series"},
        "event_detection": {k: v for k, v in event_payload.items() if k != "events"},
        "n_meal_metrics": len(meal_metrics),
        "n_exercise_metrics": len(exercise_metrics),
        "meal_metrics": [m.as_dict() for m in meal_metrics],
        "exercise_metrics": [m.as_dict() for m in exercise_metrics],
        "exclusions": exclusions,
        "reference": DUBOSE_REFERENCE.as_dict(),
    }


def summarize_event_metrics(payload: dict) -> dict:
    meals = payload.get("meal_metrics", [])
    exercise = payload.get("exercise_metrics", [])

    def mean_sd(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "sd": None}
        a = np.asarray(values, dtype=float)
        return {"n": len(a), "mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0}

    return {
        "meal": {
            "baseline_mg_dl": mean_sd([m["baseline_mg_dl"] for m in meals]),
            "peak_mg_dl": mean_sd([m["peak_mg_dl"] for m in meals if m.get("peak_mg_dl") is not None]),
            "excursion_mg_dl": mean_sd([m["excursion_mg_dl"] for m in meals if m.get("excursion_mg_dl") is not None]),
            "time_to_peak_min": mean_sd([m["time_to_peak_min"] for m in meals if m.get("time_to_peak_min") is not None]),
        },
        "exercise": {
            "baseline_mg_dl": mean_sd([m["baseline_mg_dl"] for m in exercise]),
            "nadir_mg_dl": mean_sd([m["nadir_mg_dl"] for m in exercise if m.get("nadir_mg_dl") is not None]),
            "change_mg_dl": mean_sd([m["excursion_mg_dl"] for m in exercise if m.get("excursion_mg_dl") is not None]),
        },
        "reference": DUBOSE_REFERENCE.as_dict(),
    }


def _simulate_dalla_cgm(
    *,
    baseline_glucose_mg_dl: float,
    minutes: float,
    carbs_g: float = 0.0,
    exercise_intensity: float = 0.0,
    config: HumanConfig | None = None,
    cgm_lag_tau_min: float = 6.0,
    report_step_min: float = 5.0,
    internal_step_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast glucose-only replay used for fitting; avoids unnecessary organ solvers."""
    base = config or HumanConfig()
    kwargs = {
        "dalla_basal_glucose_mg_dl": float(baseline_glucose_mg_dl),
        "glucose_setpoint_mg_dl": float(baseline_glucose_mg_dl),
    }
    if internal_step_min is not None:
        kwargs["dalla_internal_step_min"] = float(internal_step_min)
    cfg = replace(base, **kwargs)
    model = DallaManMealModel(cfg)
    state = HumanState()
    model.initialize_state(state)
    if carbs_g > 0:
        model.add_meal(state, carbs_g)
    cgm_model = CGMObservationModel(CGMObservationConfig(lag_tau_min=cgm_lag_tau_min))
    cgm_state = cgm_model.initialize(state.glucose_mg_dl)

    dt = min(0.25, report_step_min)
    n = int(math.ceil(minutes / dt))
    times = [0.0]
    blood = [float(state.glucose_mg_dl)]
    sensor = [float(cgm_state.sensor_glucose_mg_dl)]
    next_report = report_step_min
    t = 0.0
    for _ in range(n):
        ds = min(dt, minutes - t)
        if ds <= 1e-12:
            break
        model.step(state, exercise=float(exercise_intensity), dt_min=ds)
        t += ds
        cgm_value = cgm_model.step(cgm_state, state.glucose_mg_dl, ds)
        if t + 1e-9 >= next_report or t >= minutes - 1e-9:
            times.append(float(t))
            blood.append(float(state.glucose_mg_dl))
            sensor.append(float(cgm_value))
            next_report += report_step_min
    return np.asarray(times), np.asarray(blood), np.asarray(sensor)


def simulate_reference_meal(profile: FreeLivingReplayProfile = DEFAULT_REPLAY_PROFILE, high_fidelity: bool = True) -> dict:
    cfg = replace(
        HumanConfig(),
        dalla_gastric_absorption_scale=profile.gastric_absorption_scale,
        dalla_insulin_sensitivity_scale=profile.insulin_sensitivity_scale,
    )
    t, blood, cgm = _simulate_dalla_cgm(
        baseline_glucose_mg_dl=DUBOSE_REFERENCE.meal_premeal_mean_mg_dl,
        minutes=240.0,
        carbs_g=profile.effective_meal_carbs_g,
        config=cfg,
        cgm_lag_tau_min=profile.cgm_lag_tau_min,
        internal_step_min=0.05 if high_fidelity else 0.25,
    )
    i = int(np.argmax(cgm))
    baseline = float(cgm[0])
    return {
        "baseline_mg_dl": baseline,
        "peak_mg_dl": float(cgm[i]),
        "time_to_peak_min": float(t[i]),
        "excursion_mg_dl": float(cgm[i] - baseline),
        "effective_meal_carbs_g": profile.effective_meal_carbs_g,
        "gastric_absorption_scale": profile.gastric_absorption_scale,
        "blood_peak_mg_dl": float(np.max(blood)),
    }


def simulate_reference_exercise(profile: FreeLivingReplayProfile = DEFAULT_REPLAY_PROFILE, high_fidelity: bool = True) -> dict:
    cfg = replace(HumanConfig(), dalla_exercise_vmax_gain=profile.exercise_vmax_gain)
    t, blood, cgm = _simulate_dalla_cgm(
        baseline_glucose_mg_dl=DUBOSE_REFERENCE.exercise_baseline_mean_mg_dl,
        minutes=DUBOSE_REFERENCE.exercise_duration_median_min,
        exercise_intensity=profile.representative_exercise_intensity,
        config=cfg,
        cgm_lag_tau_min=profile.cgm_lag_tau_min,
        internal_step_min=0.05 if high_fidelity else 0.25,
    )
    i = int(np.argmin(cgm))
    baseline = float(cgm[0])
    return {
        "baseline_mg_dl": baseline,
        "nadir_mg_dl": float(cgm[i]),
        "change_mg_dl": float(cgm[i] - baseline),
        "time_to_nadir_min": float(t[i]),
        "representative_exercise_intensity": profile.representative_exercise_intensity,
        "exercise_vmax_gain": profile.exercise_vmax_gain,
        "blood_nadir_mg_dl": float(np.min(blood)),
    }


def _reference_objective(meal: dict, exercise: dict) -> float:
    r = DUBOSE_REFERENCE
    z = [
        (meal["peak_mg_dl"] - r.meal_peak_mean_mg_dl) / r.meal_peak_sd_mg_dl,
        (meal["time_to_peak_min"] - r.meal_time_to_peak_mean_min) / r.meal_time_to_peak_sd_min,
        (meal["excursion_mg_dl"] - r.meal_excursion_mean_mg_dl) / r.meal_excursion_sd_mg_dl,
        (exercise["nadir_mg_dl"] - r.exercise_nadir_mean_mg_dl) / r.exercise_nadir_sd_mg_dl,
        (exercise["change_mg_dl"] - r.exercise_change_mean_mg_dl) / r.exercise_change_sd_mg_dl,
    ]
    return float(np.mean(np.square(z)))


def calibrate_aggregate_event_profile() -> dict:
    """Fit a small event-replay profile to DuBose aggregate targets.

    This deliberately does *not* modify HumanConfig defaults.  It estimates latent
    free-living inputs that were not recorded in the study (meal carbohydrate and
    exercise energetic intensity) and therefore belongs to the observation/replay
    layer, not the published Dalla Man physiology core.
    """
    # A small deterministic grid is dependency-free and fast with the coarse Dalla
    # step. Candidate ranges are intentionally narrow around physiological
    # mixed-meal magnitudes and the exercise-response extension.
    best = None
    for carbs in np.arange(28.0, 43.0, 2.0):
        for gastric in np.arange(0.35, 0.61, 0.05):
            p = FreeLivingReplayProfile(
                effective_meal_carbs_g=float(carbs),
                gastric_absorption_scale=float(gastric),
                representative_exercise_intensity=0.50,
                exercise_vmax_gain=3.65,
            )
            meal = simulate_reference_meal(p, high_fidelity=False)
            # Exercise is independent of meal candidates, but evaluating here keeps
            # the objective explicit and the grid tiny.
            ex = simulate_reference_exercise(p, high_fidelity=False)
            obj = _reference_objective(meal, ex)
            if best is None or obj < best[0]:
                best = (obj, p, meal, ex)

    # Refine exercise gain after meal fit.
    _, base_profile, _, _ = best
    best_ex = None
    for gain in np.arange(3.2, 4.11, 0.05):
        p = replace(base_profile, exercise_vmax_gain=float(gain))
        meal = simulate_reference_meal(p, high_fidelity=False)
        ex = simulate_reference_exercise(p, high_fidelity=False)
        obj = _reference_objective(meal, ex)
        if best_ex is None or obj < best_ex[0]:
            best_ex = (obj, p)

    _, profile = best_ex
    meal_hf = simulate_reference_meal(profile, high_fidelity=True)
    ex_hf = simulate_reference_exercise(profile, high_fidelity=True)
    return {
        "profile": profile.as_dict(),
        "high_fidelity": {"meal": meal_hf, "exercise": ex_hf},
        "objective_mean_squared_z": _reference_objective(meal_hf, ex_hf),
        "reference": DUBOSE_REFERENCE.as_dict(),
        "identifiability_warning": (
            "Meal carbohydrate/fat quantity and exercise intensity were not recorded in the source study. "
            "The fitted effective meal carbohydrate and exercise extension are latent replay parameters, "
            "not measurements and not unique physiological estimates."
        ),
    }


def infer_effective_meal_carbs(
    baseline_mg_dl: float,
    observed_peak_mg_dl: float,
    observed_time_to_peak_min: float | None,
    profile: FreeLivingReplayProfile = DEFAULT_REPLAY_PROFILE,
) -> dict:
    """Infer a latent effective carbohydrate dose from one logged meal response."""
    cfg = replace(
        HumanConfig(),
        dalla_gastric_absorption_scale=profile.gastric_absorption_scale,
        dalla_insulin_sensitivity_scale=profile.insulin_sensitivity_scale,
    )
    best = None
    for carbs in np.arange(5.0, 101.0, 2.5):
        t, _, cgm = _simulate_dalla_cgm(
            baseline_glucose_mg_dl=baseline_mg_dl,
            minutes=240.0,
            carbs_g=float(carbs),
            config=cfg,
            cgm_lag_tau_min=profile.cgm_lag_tau_min,
            internal_step_min=0.25,
        )
        i = int(np.argmax(cgm))
        peak = float(cgm[i])
        tp = float(t[i])
        obj = ((peak - observed_peak_mg_dl) / 13.0) ** 2
        if observed_time_to_peak_min is not None:
            obj += ((tp - observed_time_to_peak_min) / 31.0) ** 2
        if best is None or obj < best[0]:
            best = (obj, float(carbs), peak, tp)
    obj, carbs, peak, tp = best
    return {
        "effective_carbs_g": carbs,
        "predicted_peak_mg_dl": peak,
        "predicted_time_to_peak_min": tp,
        "objective": float(obj),
        "warning": "Effective carbohydrate is a latent model input because the Jaeb diary did not record meal carbohydrate quantity.",
    }


def save_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _split_participants(ids: Iterable[str], seed: int = 2020) -> dict[str, set[str]]:
    unique = sorted({str(x) for x in ids})
    if len(unique) < 3:
        raise ValueError("At least three participants with event metrics are required")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=object)[rng.permutation(len(unique))].tolist()
    n = len(shuffled)
    n_train = max(1, int(round(0.60 * n)))
    n_val = max(1, int(round(0.20 * n)))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return {
        "train": set(shuffled[:n_train]),
        "validation": set(shuffled[n_train:n_train+n_val]),
        "test": set(shuffled[n_train+n_val:]),
    }


def _aggregate_event_subset(payload: dict, ids: set[str]) -> dict:
    meals = [m for m in payload.get("meal_metrics", []) if str(m["participant_id"]) in ids]
    exs = [m for m in payload.get("exercise_metrics", []) if str(m["participant_id"]) in ids]
    def mean(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "n_participants": len(ids),
        "n_meals": len(meals),
        "n_exercise": len(exs),
        "meal_baseline_mg_dl": mean(meals, "baseline_mg_dl"),
        "meal_peak_mg_dl": mean(meals, "peak_mg_dl"),
        "meal_excursion_mg_dl": mean(meals, "excursion_mg_dl"),
        "meal_time_to_peak_min": mean(meals, "time_to_peak_min"),
        "exercise_baseline_mg_dl": mean(exs, "baseline_mg_dl"),
        "exercise_nadir_mg_dl": mean(exs, "nadir_mg_dl"),
        "exercise_change_mg_dl": mean(exs, "excursion_mg_dl"),
    }


def _simulate_profile_for_observed_baselines(
    profile: FreeLivingReplayProfile, observed: dict, high_fidelity: bool,
    *, simulate_meal: bool = True, simulate_exercise: bool = True,
) -> dict:
    meal = None
    if simulate_meal and observed.get("meal_baseline_mg_dl") is not None:
        cfg = replace(HumanConfig(),
                      dalla_gastric_absorption_scale=profile.gastric_absorption_scale,
                      dalla_insulin_sensitivity_scale=profile.insulin_sensitivity_scale)
        t, _, cgm = _simulate_dalla_cgm(
            baseline_glucose_mg_dl=observed["meal_baseline_mg_dl"], minutes=240.0,
            carbs_g=profile.effective_meal_carbs_g, config=cfg,
            cgm_lag_tau_min=profile.cgm_lag_tau_min,
            internal_step_min=0.05 if high_fidelity else 0.25)
        i = int(np.argmax(cgm))
        meal = {
            "baseline_mg_dl": float(cgm[0]), "peak_mg_dl": float(cgm[i]),
            "excursion_mg_dl": float(cgm[i] - cgm[0]), "time_to_peak_min": float(t[i]),
        }
    exercise = None
    if simulate_exercise and observed.get("exercise_baseline_mg_dl") is not None:
        cfg = replace(HumanConfig(), dalla_exercise_vmax_gain=profile.exercise_vmax_gain)
        t, _, cgm = _simulate_dalla_cgm(
            baseline_glucose_mg_dl=observed["exercise_baseline_mg_dl"],
            minutes=DUBOSE_REFERENCE.exercise_duration_median_min,
            exercise_intensity=profile.representative_exercise_intensity,
            config=cfg, cgm_lag_tau_min=profile.cgm_lag_tau_min,
            internal_step_min=0.05 if high_fidelity else 0.25)
        i = int(np.argmin(cgm))
        exercise = {
            "baseline_mg_dl": float(cgm[0]), "nadir_mg_dl": float(cgm[i]),
            "change_mg_dl": float(cgm[i] - cgm[0]), "time_to_nadir_min": float(t[i]),
        }
    return {"meal": meal, "exercise": exercise}


def _observed_prediction_error(observed: dict, predicted: dict) -> dict:
    r = DUBOSE_REFERENCE
    terms = {}
    if predicted.get("meal") and observed.get("meal_peak_mg_dl") is not None:
        terms["meal_peak_z"] = (predicted["meal"]["peak_mg_dl"] - observed["meal_peak_mg_dl"]) / r.meal_peak_sd_mg_dl
        terms["meal_excursion_z"] = (predicted["meal"]["excursion_mg_dl"] - observed["meal_excursion_mg_dl"]) / r.meal_excursion_sd_mg_dl
        terms["meal_tpeak_z"] = (predicted["meal"]["time_to_peak_min"] - observed["meal_time_to_peak_min"]) / r.meal_time_to_peak_sd_min
    if predicted.get("exercise") and observed.get("exercise_nadir_mg_dl") is not None:
        terms["exercise_nadir_z"] = (predicted["exercise"]["nadir_mg_dl"] - observed["exercise_nadir_mg_dl"]) / r.exercise_nadir_sd_mg_dl
        terms["exercise_change_z"] = (predicted["exercise"]["change_mg_dl"] - observed["exercise_change_mg_dl"]) / r.exercise_change_sd_mg_dl
    mse = float(np.mean(np.square(list(terms.values())))) if terms else None
    return {"mean_squared_z": mse, "terms": {k: float(v) for k, v in terms.items()}}


def fit_mechanistic_event_profile(payload: dict, seed: int = 2020) -> dict:
    """TRAIN-only calibration with held-out participant validation/test evaluation.

    The meal and exercise replay parameters are fitted separately because the two
    mechanisms are independent in the reduced glucose core.  This makes local
    calibration tractable while preserving a strict participant-level holdout.
    """
    ids = {str(m["participant_id"]) for m in payload.get("meal_metrics", [])}
    ids |= {str(m["participant_id"]) for m in payload.get("exercise_metrics", [])}
    splits = _split_participants(ids, seed=seed)
    observed = {name: _aggregate_event_subset(payload, sids) for name, sids in splits.items()}
    train = observed["train"]
    if train["n_meals"] == 0 and train["n_exercise"] == 0:
        raise ValueError("Training split contains no usable event metrics")

    profile = FreeLivingReplayProfile()

    # Fit latent meal amount + gastric timing using TRAIN meal-event summaries only.
    if train["n_meals"] > 0:
        best_meal = None
        for carbs in np.arange(20.0, 61.0, 4.0):
            for gastric in np.arange(0.30, 0.91, 0.10):
                p = replace(profile, effective_meal_carbs_g=float(carbs), gastric_absorption_scale=float(gastric))
                pred = _simulate_profile_for_observed_baselines(
                    p, train, high_fidelity=False, simulate_exercise=False
                )
                err = _observed_prediction_error(train, {"meal": pred["meal"], "exercise": None})["mean_squared_z"]
                if err is not None and (best_meal is None or err < best_meal[0]):
                    best_meal = (err, p)
        coarse = best_meal[1]
        best_meal2 = None
        for carbs in np.arange(max(5.0, coarse.effective_meal_carbs_g - 5.0), coarse.effective_meal_carbs_g + 5.01, 1.0):
            for gastric in np.arange(max(0.15, coarse.gastric_absorption_scale - 0.15), coarse.gastric_absorption_scale + 0.151, 0.05):
                p = replace(coarse, effective_meal_carbs_g=float(carbs), gastric_absorption_scale=float(gastric))
                pred = _simulate_profile_for_observed_baselines(
                    p, train, high_fidelity=False, simulate_exercise=False
                )
                err = _observed_prediction_error(train, {"meal": pred["meal"], "exercise": None})["mean_squared_z"]
                if err is not None and (best_meal2 is None or err < best_meal2[0]):
                    best_meal2 = (err, p)
        profile = best_meal2[1]

    # Fit only the glucose-utilization extension to TRAIN exercise events.  The
    # representative intensity remains explicit because the diary did not report
    # metabolic workload.
    if train["n_exercise"] > 0:
        best_ex = None
        for gain in np.arange(1.0, 6.01, 0.25):
            p = replace(profile, exercise_vmax_gain=float(gain))
            pred = _simulate_profile_for_observed_baselines(
                p, train, high_fidelity=False, simulate_meal=False
            )
            err = _observed_prediction_error(train, {"meal": None, "exercise": pred["exercise"]})["mean_squared_z"]
            if err is not None and (best_ex is None or err < best_ex[0]):
                best_ex = (err, p)
        coarse = best_ex[1]
        best_ex2 = None
        for gain in np.arange(max(0.25, coarse.exercise_vmax_gain - 0.4), coarse.exercise_vmax_gain + 0.401, 0.05):
            p = replace(coarse, exercise_vmax_gain=float(gain))
            pred = _simulate_profile_for_observed_baselines(
                p, train, high_fidelity=False, simulate_meal=False
            )
            err = _observed_prediction_error(train, {"meal": None, "exercise": pred["exercise"]})["mean_squared_z"]
            if err is not None and (best_ex2 is None or err < best_ex2[0]):
                best_ex2 = (err, p)
        profile = best_ex2[1]

    evaluation = {}
    for name in ("train", "validation", "test"):
        pred = _simulate_profile_for_observed_baselines(profile, observed[name], high_fidelity=True)
        evaluation[name] = {
            "observed": observed[name], "predicted": pred,
            "error": _observed_prediction_error(observed[name], pred),
        }
    overlap = {
        "train_validation": sorted(splits["train"] & splits["validation"]),
        "train_test": sorted(splits["train"] & splits["test"]),
        "validation_test": sorted(splits["validation"] & splits["test"]),
    }
    return {
        "seed": seed,
        "profile": profile.as_dict(),
        "split_subject_ids": {k: sorted(v) for k, v in splits.items()},
        "leakage_check": {"overlaps": overlap, "passed": not any(overlap.values())},
        "evaluation": evaluation,
        "reference": DUBOSE_REFERENCE.as_dict(),
        "limitations": [
            "Meal carbohydrate/fat quantity was not logged, so effective carbohydrate is latent and non-identifiable from CGM alone.",
            "Reported exercise type/duration does not provide metabolic intensity; representative intensity is a replay assumption.",
            "Sleep/wake events are parsed and synchronized but are not yet causal inputs to the glucose physiology model.",
            "Diary entries may omit meals/snacks/exercise, as noted by the source study.",
        ],
    }
