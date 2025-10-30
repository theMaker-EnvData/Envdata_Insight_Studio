#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
import requests
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator


ZONE_KST = ZoneInfo("Asia/Seoul")
MAP_BOUNDS_URL = "https://api.waqi.info/map/bounds/"
FEED_URL_TEMPLATE = "https://api.waqi.info/feed/@{uid}/"
DEFAULT_TOLERANCE_MIN = 90
STATION_CAP = 1500


class AqicnError(Exception):
    """Base error for AQICN fetches."""


class AqicnAuthError(AqicnError):
    """Authentication related errors."""


class AqicnFetchError(AqicnError):
    """Non-fatal fetch errors for individual stations."""


class BoundingBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    @field_validator("*", mode="before")
    def validate_coord(cls, value: Any) -> float:
        # Coerce numeric strings to float while enforcing numeric input.
        if not isinstance(value, (float, int)):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Bounding box coordinates must be numeric.") from exc
        return float(value)

    @classmethod
    def from_string(cls, bbox_str: str) -> "BoundingBox":
        try:
            parts = [float(part.strip()) for part in bbox_str.split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError("Bounding box must be comma separated floats.") from exc
        if len(parts) != 4:
            raise ValueError("Bounding box requires four comma separated values.")
        return cls(lon_min=parts[0], lat_min=parts[1], lon_max=parts[2], lat_max=parts[3])

    def to_latlng_query(self) -> str:
        return f"{self.lat_min},{self.lon_min},{self.lat_max},{self.lon_max}"

    def as_dict(self) -> Dict[str, float]:
        return {
            "lon_min": self.lon_min,
            "lat_min": self.lat_min,
            "lon_max": self.lon_max,
            "lat_max": self.lat_max,
        }


class NormalizedRow(BaseModel):
    uid: int
    station: Dict[str, Any]
    time_kst: Optional[str]
    delta_min: Optional[int]
    pm25: Optional[float]
    status: str
    error: Optional[str] = None


@dataclass
class FetchSettings:
    target_rps: float
    burst_rps: float
    burst_seconds: float
    max_concurrency: int
    timeout_sec: float
    max_retries: int
    retry_backoff: float
    retry_jitter: float
    connect_timeout: float
    read_timeout: float


class RateLimiter:
    def __init__(self, target_rps: float, burst_rps: float, burst_seconds: float) -> None:
        self.target_rps = max(target_rps, 0.1)
        self.burst_rps = max(burst_rps, self.target_rps)
        self.burst_seconds = max(burst_seconds, 0.0)
        self.lock = asyncio.Lock()
        self.start_time = monotonic()
        self.next_allowed = self.start_time

    async def acquire(self) -> None:
        async with self.lock:
            now = monotonic()
            elapsed = now - self.start_time
            current_rps = self.burst_rps if elapsed <= self.burst_seconds else self.target_rps
            min_interval = 1.0 / current_rps if current_rps > 0 else 0.0

            wait_time = max(0.0, self.next_allowed - now)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = monotonic()
                elapsed = now - self.start_time
                current_rps = self.burst_rps if elapsed <= self.burst_seconds else self.target_rps
                min_interval = 1.0 / current_rps if current_rps > 0 else 0.0

            self.next_allowed = max(now, self.next_allowed) + min_interval


def kst_now() -> datetime:
    return datetime.now(tz=ZONE_KST)


def load_cities(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_time_of_day(ts_str: str) -> dtime:
    try:
        hour_minute = datetime.strptime(ts_str, "%H:%M")
    except ValueError as exc:
        raise ValueError("Time string must be formatted HH:MM.") from exc
    return dtime(hour=hour_minute.hour, minute=hour_minute.minute)


def resolve_target_datetime(ts_str: str, out_date: Optional[str]) -> Tuple[str, datetime]:
    """Create a timezone-aware datetime in KST for the target time."""
    target_time = parse_time_of_day(ts_str)
    if out_date:
        try:
            day = datetime.strptime(out_date, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError("out_date must be formatted YYYYMMDD.") from exc
    else:
        day = datetime.now(ZONE_KST).date()

    target_dt = datetime.combine(day, target_time, tzinfo=ZONE_KST)
    return day.strftime("%Y%m%d"), target_dt


def configure_logger(out_date: str) -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"pipeline_{out_date}.log"

    logger = logging.getLogger("aqicn_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_output_paths(out_date: str, ts_str: str) -> Tuple[Path, Path]:
    ts_compact = ts_str.replace(":", "")
    base_dir = Path("data") / "raw" / "aqicn"
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_path = base_dir / f"aqicn_{out_date}_{ts_compact}_raw.json"
    norm_path = base_dir / f"aqicn_{out_date}_{ts_compact}_norm.json"
    return raw_path, norm_path


def fetch_station_listing(
    session: requests.Session,
    token: str,
    bbox: BoundingBox,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    params = {"token": token, "latlng": bbox.to_latlng_query()}
    logger.info("Requesting station listing for bbox %s", bbox.as_dict())

    response = session.get(MAP_BOUNDS_URL, params=params, timeout=30)
    if response.status_code == 401:
        raise AqicnAuthError("Unauthorized: check AQICN_TOKEN.")
    if response.status_code == 403:
        raise AqicnAuthError("Forbidden: token lacks permissions.")
    if response.status_code >= 500:
        raise AqicnError(f"Station listing failed with status {response.status_code}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise AqicnError("Station listing returned non-JSON response.") from exc

    if payload.get("status") != "ok":
        message = payload.get("data") or "Unknown error"
        if "token" in str(message).lower():
            raise AqicnAuthError(f"AQICN auth error: {message}")
        raise AqicnError(f"Station listing error: {message}")

    stations = payload.get("data") or []
    station_count = len(stations)
    logger.info("Station listing returned %s stations.", station_count)

    if station_count > STATION_CAP:
        logger.warning("Station listing exceeded cap (%s). Truncating.", STATION_CAP)
        stations = stations[:STATION_CAP]

    return stations, payload


def parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    if not header_value:
        return None
    header_value = header_value.strip()
    if not header_value:
        return None
    if header_value.isdigit():
        return float(header_value)
    try:
        retry_dt = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if retry_dt is None:
        return None
    if retry_dt.tzinfo is None:
        now = datetime.utcnow()
    else:
        now = datetime.now(tz=retry_dt.tzinfo)
    return max(0.0, (retry_dt - now).total_seconds())


def jitter_delay(base_delay: float, jitter: float) -> float:
    if jitter <= 0:
        return max(base_delay, 0.0)
    delta = random.uniform(-jitter, jitter)
    return max(base_delay + delta, 0.0)


async def fetch_station_feed_async(
    index: int,
    uid: int,
    session: aiohttp.ClientSession,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    settings: FetchSettings,
    logger: logging.Logger,
    token: str,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    url = FEED_URL_TEMPLATE.format(uid=uid)
    params = {"token": token}
    last_error: Optional[str] = None

    for attempt in range(settings.max_retries):
        await limiter.acquire()
        async with semaphore:
            try:
                async with session.get(url, params=params) as response:
                    status_code = response.status
                    if status_code in (401, 403):
                        raise AqicnAuthError(
                            f"AQICN auth error for station {uid}: HTTP {status_code}"
                        )
                    if status_code == 429:
                        retry_after = parse_retry_after(response.headers.get("Retry-After"))
                        if retry_after is None:
                            retry_after = 1.5 * (1.0 / settings.target_rps)
                        last_error = f"Station {uid} rate limited (HTTP 429)."
                        logger.warning(
                            "Station %s rate limited; retrying in %.2fs (attempt %s/%s).",
                            uid,
                            retry_after,
                            attempt + 1,
                            settings.max_retries,
                        )
                        if attempt == settings.max_retries - 1:
                            return index, None, last_error
                        await asyncio.sleep(retry_after)
                        continue
                    if status_code >= 500:
                        base_delay = settings.retry_backoff * (2 ** attempt)
                        retry_delay = jitter_delay(base_delay, settings.retry_jitter)
                        last_error = f"Station {uid} server error HTTP {status_code}."
                        logger.warning(
                            "Station %s server error HTTP %s; retrying in %.2fs (attempt %s/%s).",
                            uid,
                            status_code,
                            retry_delay,
                            attempt + 1,
                            settings.max_retries,
                        )
                        if attempt == settings.max_retries - 1:
                            return index, None, last_error
                        await asyncio.sleep(retry_delay)
                        continue

                    try:
                        payload = await response.json(content_type=None)
                    except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                        base_delay = settings.retry_backoff * (2 ** attempt)
                        retry_delay = jitter_delay(base_delay, settings.retry_jitter)
                        last_error = f"Station {uid} returned invalid JSON: {exc}"
                        logger.warning(
                            "Station %s invalid JSON; retrying in %.2fs (attempt %s/%s).",
                            uid,
                            retry_delay,
                            attempt + 1,
                            settings.max_retries,
                        )
                        if attempt == settings.max_retries - 1:
                            return index, None, last_error
                        await asyncio.sleep(retry_delay)
                        continue

                    if payload.get("status") == "ok":
                        return index, payload, None

                    message = payload.get("data")
                    last_error = (
                        f"Station {uid} returned status={payload.get('status')} data={message}"
                    )
                    if status_code == 404:
                        logger.warning("Station %s not found (404).", uid)
                        return index, None, last_error

                    base_delay = settings.retry_backoff * (2 ** attempt)
                    retry_delay = jitter_delay(base_delay, settings.retry_jitter)
                    logger.warning(
                        "Station %s status=%s; retrying in %.2fs (attempt %s/%s).",
                        uid,
                        payload.get("status"),
                        retry_delay,
                        attempt + 1,
                        settings.max_retries,
                    )
                    if attempt == settings.max_retries - 1:
                        return index, None, last_error
                    await asyncio.sleep(retry_delay)
                    continue
            except AqicnAuthError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                base_delay = settings.retry_backoff * (2 ** attempt)
                retry_delay = jitter_delay(base_delay, settings.retry_jitter)
                last_error = f"Station {uid} request error: {exc}"
                logger.warning(
                    "Station %s request error: %s; retrying in %.2fs (attempt %s/%s).",
                    uid,
                    exc,
                    retry_delay,
                    attempt + 1,
                    settings.max_retries,
                )
                if attempt == settings.max_retries - 1:
                    return index, None, last_error
                await asyncio.sleep(retry_delay)
                continue

    return index, None, last_error or f"Station {uid} failed without response."


def parse_station_time(time_payload: Optional[Dict[str, Any]]) -> Optional[datetime]:
    if not time_payload:
        return None
    s_value = time_payload.get("s")
    tz_value = time_payload.get("tz") or ""
    if isinstance(tz_value, str) and tz_value.upper().startswith("UTC"):
        tz_value = tz_value[3:]
    if isinstance(s_value, str):
        candidate = s_value.replace(" ", "T")
        if tz_value and tz_value not in candidate:
            if tz_value.startswith(("+", "-")) or tz_value == "Z":
                candidate = f"{candidate}{tz_value}"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = None
        if parsed and parsed.tzinfo:
            return parsed
        if parsed:
            if tz_value and (tz_value.startswith(("+", "-")) or tz_value == "Z"):
                try:
                    parsed = datetime.fromisoformat(candidate)
                    if parsed.tzinfo:
                        return parsed
                except ValueError:
                    return parsed.replace(tzinfo=ZONE_KST)
            return parsed.replace(tzinfo=ZONE_KST)
    epoch_value = time_payload.get("v")
    if isinstance(epoch_value, (int, float)):
        return datetime.fromtimestamp(float(epoch_value), tz=ZoneInfo("UTC"))
    return None


def extract_pm25(feed: Dict[str, Any]) -> Optional[float]:
    try:
        value = feed["data"]["iaqi"]["pm25"]["v"]
    except (KeyError, TypeError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_station_metadata(
    station_entry: Dict[str, Any],
    feed_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    station_meta = station_entry.get("station") or {}
    name = station_meta.get("name") or station_entry.get("name")
    feed_station = (
        (feed_payload or {}).get("data", {}).get("city", {}) if feed_payload else {}
    )

    geo = None
    feed_geo = feed_station.get("geo")
    if isinstance(feed_geo, (list, tuple)) and len(feed_geo) >= 2:
        geo = [float(feed_geo[0]), float(feed_geo[1])]
    else:
        lat = station_entry.get("lat")
        lon = station_entry.get("lon")
        if lat is not None and lon is not None:
            geo = [float(lat), float(lon)]

    return {
        "name": name,
        "geo": geo,
    }


def normalize_station(
    station_entry: Dict[str, Any],
    feed_payload: Optional[Dict[str, Any]],
    target_dt: datetime,
    tolerance_min: int,
) -> NormalizedRow:
    uid = station_entry.get("uid")
    error_message = None
    observed_dt = None
    delta_min = None
    pm25_value = None
    status = "STALE"

    if feed_payload:
        observed_dt = parse_station_time(feed_payload.get("data", {}).get("time"))
        pm25_value = extract_pm25(feed_payload)
        if observed_dt:
            observed_kst = observed_dt.astimezone(ZONE_KST)
            delta = observed_kst - target_dt
            delta_min = int(round(delta.total_seconds() / 60))
            within_tol = abs(delta_min) <= tolerance_min
            if within_tol:
                if pm25_value is not None:
                    status = "OK"
                else:
                    status = "MISSING"
            else:
                status = "STALE"
        else:
            status = "STALE"
    else:
        error_message = "Feed not available."
        status = "ERROR"

    station_payload = build_station_metadata(station_entry, feed_payload)
    time_kst = (
        observed_dt.astimezone(ZONE_KST).isoformat() if observed_dt else None
    )

    if status in {"STALE", "MISSING"} and feed_payload is None:
        status = "ERROR"

    return NormalizedRow(
        uid=int(uid) if uid is not None else -1,
        station=station_payload,
        time_kst=time_kst,
        delta_min=delta_min,
        pm25=pm25_value,
        status=status,
        error=error_message,
    )


def summarize_counts(rows: Iterable[NormalizedRow]) -> Tuple[int, int, int]:
    total = 0
    ok = 0
    stale_or_missing = 0
    for row in rows:
        total += 1
        if row.status == "OK":
            ok += 1
        else:
            stale_or_missing += 1
    return total, ok, stale_or_missing


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


async def gather_station_feeds(
    stations_with_uid: List[Tuple[Dict[str, Any], int]],
    token: str,
    settings: FetchSettings,
    target_dt: datetime,
    tolerance_min: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[NormalizedRow]]:
    if not stations_with_uid:
        return [], []

    limiter = RateLimiter(
        target_rps=settings.target_rps,
        burst_rps=settings.burst_rps,
        burst_seconds=settings.burst_seconds,
    )
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    timeout = aiohttp.ClientTimeout(
        total=settings.timeout_sec,
        connect=settings.connect_timeout,
        sock_read=settings.read_timeout,
    )

    headers = {"X-AQICN-TOKEN": token}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        tasks = [
            asyncio.create_task(
                fetch_station_feed_async(
                    index,
                    uid,
                    session,
                    limiter,
                    semaphore,
                    settings,
                    logger,
                    token,
                )
            )
            for index, (_, uid) in enumerate(stations_with_uid)
        ]

        raw_feeds: List[Dict[str, Any]] = [
            {"uid": uid} for _, uid in stations_with_uid
        ]
        norm_rows: List[Optional[NormalizedRow]] = [None] * len(stations_with_uid)

        counts = {"ok": 0, "stale": 0, "error": 0}
        completed = 0
        total = len(tasks)

        def update_counts(row: NormalizedRow) -> None:
            if row.status == "OK":
                counts["ok"] += 1
            elif row.status == "ERROR":
                counts["error"] += 1
            else:
                counts["stale"] += 1

        for task in asyncio.as_completed(tasks):
            index, payload, error_message = await task
            station, uid = stations_with_uid[index]

            if error_message:
                raw_feeds[index]["error"] = error_message
                logger.error("Station %s failed: %s", uid, error_message)
                normalized = normalize_station(station, None, target_dt, tolerance_min)
                normalized.error = error_message
            else:
                raw_feeds[index]["payload"] = payload
                normalized = normalize_station(station, payload, target_dt, tolerance_min)
            norm_rows[index] = normalized

            update_counts(normalized)
            completed += 1
            progress_line = (
                f"\rfetched {completed}/{total} "
                f"(ok={counts['ok']}, stale={counts['stale']}, err={counts['error']})"
            )
            print(progress_line, end="", flush=True)

        print()

    final_rows = [row for row in norm_rows if row is not None]
    return raw_feeds, final_rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch AQICN station data snapshot.")
    parser.add_argument("--ts", required=True, help="Target clock time (HH:MM) in KST.")
    parser.add_argument(
        "--out_date",
        help="Target date in YYYYMMDD (defaults to current date in KST).",
    )
    parser.add_argument(
        "--tol_min",
        type=int,
        default=DEFAULT_TOLERANCE_MIN,
        help=f"Tolerance window in minutes (default: {DEFAULT_TOLERANCE_MIN}).",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box override (lon_min,lat_min,lon_max,lat_max)."
        " Defaults to REGION_BBOX from environment.",
    )
    parser.add_argument(
        "--target_rps",
        type=float,
        default=16.0,
        help="Steady-state requests per second.",
    )
    parser.add_argument(
        "--burst_rps",
        type=float,
        default=40.0,
        help="Burst requests per second during initial window.",
    )
    parser.add_argument(
        "--burst_seconds",
        type=float,
        default=2.0,
        help="Duration of burst window in seconds.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=16,
        help="Maximum concurrent station fetches.",
    )
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=8.0,
        help="Overall per-request timeout seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per station feed.",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=0.6,
        help="Base retry backoff seconds (exponential).",
    )
    parser.add_argument(
        "--retry_jitter",
        type=float,
        default=0.15,
        help="Retry jitter seconds (+/-).",
    )
    parser.add_argument(
        "--connect_timeout",
        type=float,
        default=3.0,
        help="Connect timeout seconds.",
    )
    parser.add_argument(
        "--read_timeout",
        type=float,
        default=6.0,
        help="Read timeout seconds.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv(dotenv_path=Path("config/.env"))

    token = os.getenv("AQICN_TOKEN")
    if not token:
        print("Missing AQICN_TOKEN in environment.", file=sys.stderr)
        return 1

    bbox_source = args.bbox or os.getenv("REGION_BBOX")
    if not bbox_source:
        print("Missing REGION_BBOX (env or CLI --bbox).", file=sys.stderr)
        return 1

    try:
        bbox = BoundingBox.from_string(bbox_source)
    except (ValueError, ValidationError) as exc:
        print(f"Invalid bounding box: {exc}", file=sys.stderr)
        return 1

    try:
        out_date_str, target_dt = resolve_target_datetime(args.ts, args.out_date)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    settings = FetchSettings(
        target_rps=max(args.target_rps, 0.1),
        burst_rps=max(args.burst_rps, max(args.target_rps, 0.1)),
        burst_seconds=max(args.burst_seconds, 0.0),
        max_concurrency=max(args.max_concurrency, 1),
        timeout_sec=max(args.timeout_sec, 1.0),
        max_retries=max(args.max_retries, 1),
        retry_backoff=max(args.retry_backoff, 0.1),
        retry_jitter=max(args.retry_jitter, 0.0),
        connect_timeout=max(args.connect_timeout, 0.1),
        read_timeout=max(args.read_timeout, 0.1),
    )

    logger = configure_logger(out_date_str)
    logger.info("Starting AQICN fetch for target %s", target_dt.isoformat())

    raw_path, norm_path = create_output_paths(out_date_str, args.ts)

    try:
        with requests.Session() as session:
            station_list, listing_payload = fetch_station_listing(session, token, bbox, logger)
    except AqicnAuthError as exc:
        logger.error("Authentication error: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    except AqicnError as exc:
        logger.error("Listing error: %s", exc)
        print(str(exc), file=sys.stderr)
        return 3

    stations_with_uid: List[Tuple[Dict[str, Any], int]] = []
    for station in station_list:
        uid = station.get("uid")
        if uid is None:
            logger.warning("Skipping station without UID: %s", station)
            continue
        stations_with_uid.append((station, int(uid)))

    try:
        raw_feeds, norm_rows = asyncio.run(
            gather_station_feeds(
                stations_with_uid,
                token,
                settings,
                target_dt,
                args.tol_min,
                logger,
            )
        )
    except AqicnAuthError as exc:
        logger.error("Authentication error while fetching station feeds: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2

    raw_snapshot = {
        "generated_at_kst": kst_now().isoformat(),
        "target_ts_kst": target_dt.isoformat(),
        "bbox": bbox.as_dict(),
        "station_listing": listing_payload,
        "feeds": raw_feeds,
    }
    save_json(raw_path, raw_snapshot)
    save_json(norm_path, [row.model_dump() for row in norm_rows])

    total, ok_count, stale_missing = summarize_counts(norm_rows)
    logger.info(
        "Completed AQICN fetch: total=%s ok=%s stale_or_missing=%s",
        total,
        ok_count,
        stale_missing,
    )

    print(
        f"total_stations={total} ok_within_tol={ok_count} stale_or_missing={stale_missing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
