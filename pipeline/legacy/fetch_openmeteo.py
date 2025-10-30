#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator


ZONE_KST = ZoneInfo("Asia/Seoul")
DEFAULT_TOLERANCE_MIN = 0
AIR_QUALITY_FIELDS = [
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "carbon_monoxide",
    "sulphur_dioxide",
]
WEATHER_FIELDS = [
    "wind_speed_10m",
    "wind_direction_10m",
]
COMBINED_FIELDS = AIR_QUALITY_FIELDS + WEATHER_FIELDS
TARGET_COUNT = 9
TARGET_STEP_HOURS = 3
DEFAULT_GRID_DEG = 0.4
MAX_COORDS_PER_REQUEST = 80  # Tune within 300–500 to avoid URL bloat while limiting payload size.


class CoordinateChunk(NamedTuple):
    """Represents a contiguous slice of flattened coordinate pairs."""

    chunk_id: int
    start_index: int
    end_index: int
    latitudes: List[float]
    longitudes: List[float]


class OpenMeteoError(Exception):
    """Base error for Open-Meteo fetching."""


class OpenMeteoAuthError(OpenMeteoError):
    """Authentication related errors."""


class OpenMeteoRateLimitError(OpenMeteoError):
    """Raised when Open-Meteo responds with HTTP 429."""


class OpenMeteoNetworkError(OpenMeteoError):
    """Raised when network-related errors occur."""


class OpenMeteoClientError(OpenMeteoError):
    """Raised for 4xx client-side errors."""


class BoundingBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    @field_validator("*", mode="before")
    @classmethod
    def validate_coord(cls, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Bounding box coordinates must be numeric.") from exc

    @classmethod
    def from_string(cls, bbox_str: str) -> "BoundingBox":
        try:
            parts = [part.strip() for part in bbox_str.split(",")]
            if len(parts) != 4:
                raise ValueError
        except (AttributeError, ValueError) as exc:
            raise ValueError("Bounding box must be four comma-separated floats.") from exc
        return cls(lon_min=parts[0], lat_min=parts[1], lon_max=parts[2], lat_max=parts[3])

    def as_dict(self) -> Dict[str, float]:
        return {
            "lon_min": self.lon_min,
            "lat_min": self.lat_min,
            "lon_max": self.lon_max,
            "lat_max": self.lat_max,
        }

    def to_bounding_box_param(self) -> str:
        return f"{self.lat_min},{self.lon_min},{self.lat_max},{self.lon_max}"

    def south_west_north_east(self) -> Tuple[float, float, float, float]:
        return (self.lat_min, self.lon_min, self.lat_max, self.lon_max)


def configure_logger(out_date: str) -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"pipeline_{out_date}.log"

    logger = logging.getLogger("openmeteo_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def kst_now() -> datetime:
    return datetime.now(tz=ZONE_KST)


def parse_time_of_day(ts_str: str) -> dtime:
    try:
        parsed = datetime.strptime(ts_str, "%H:%M")
    except ValueError as exc:
        raise ValueError("Time string must be formatted HH:MM.") from exc
    return dtime(hour=parsed.hour, minute=parsed.minute)


def resolve_target_datetime(ts_str: str, out_date: Optional[str]) -> Tuple[str, datetime]:
    target_time = parse_time_of_day(ts_str)
    if out_date:
        try:
            day = datetime.strptime(out_date, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError("out_date must be formatted YYYYMMDD.") from exc
    else:
        day = kst_now().date()
    target_dt = datetime.combine(day, target_time, tzinfo=ZONE_KST)
    return day.strftime("%Y%m%d"), target_dt


def build_target_timestamps(start_dt: datetime) -> List[datetime]:
    return [start_dt + timedelta(hours=TARGET_STEP_HOURS * offset) for offset in range(TARGET_COUNT)]


def format_hour_param(dt: datetime, target_zone: ZoneInfo) -> str:
    return dt.astimezone(target_zone).strftime("%Y-%m-%dT%H:%M")



def normalize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    if isinstance(payload, tuple):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("error"):
        reason = payload.get("reason", "open-meteo error")
        raise RuntimeError(f"Open-Meteo error: {reason}")
    return payload


def chunk_pause(rate_limited: bool = False) -> None:
    """Ensure a minimum delay between chunk requests."""
    delay = 0.25
    if rate_limited:
        delay += random.uniform(0.0, 0.5)
    time.sleep(delay)


def backoff_sleep(attempt_idx: int) -> None:
    base = 1.0 * (2 ** attempt_idx)
    delay = min(base + random.uniform(0.0, 0.5), 20.0)
    time.sleep(delay)

def _inclusive_axis(start: float, stop: float, step: float) -> List[float]:
    axis = np.arange(start, stop + step / 2.0, step, dtype=float)
    if axis.size == 0:
        axis = np.array([start], dtype=float)
    if axis[-1] < stop - 1e-6:
        axis = np.append(axis, stop)
    else:
        axis[-1] = stop
    axis = np.round(axis, 6)
    expected_count = int(math.floor(((stop - start) + 1e-9) / step)) + 2
    if expected_count < 2:
        expected_count = 2
    while axis.size < expected_count:
        axis = np.append(axis, round(stop, 6))
    axis[-1] = round(stop, 6)
    return axis.tolist()


def generate_grid_points(bbox_swne: Tuple[float, float, float, float], grid_deg: float) -> Tuple[List[float], List[float]]:
    south, west, north, east = bbox_swne
    if grid_deg <= 0:
        raise ValueError("GRID_DEG must be > 0")

    lat_vals = _inclusive_axis(south, north, grid_deg)
    lon_vals = _inclusive_axis(west, east, grid_deg)

    return lat_vals, lon_vals


def build_flat_grid(lat_axis: Sequence[float], lon_axis: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Flatten grid using lat-major (south→north outer, west→east inner) order."""
    lat_array, lon_array = np.meshgrid(
        np.asarray(lat_axis, dtype=float),
        np.asarray(lon_axis, dtype=float),
        indexing="ij",
    )
    return lat_array.ravel().astype(float).tolist(), lon_array.ravel().astype(float).tolist()


def _compute_axis_weights(axis: np.ndarray, target: float) -> Tuple[int, int, float]:
    if target <= axis[0]:
        return 0, 0, 0.0
    if target >= axis[-1]:
        last = len(axis) - 1
        return last, last, 0.0
    pos = int(np.searchsorted(axis, target))
    upper_val = axis[pos]
    if math.isclose(target, upper_val, abs_tol=1e-6):
        return pos, pos, 0.0
    lower_idx = pos - 1
    lower_val = axis[lower_idx]
    if math.isclose(lower_val, upper_val, abs_tol=1e-12):
        return lower_idx, lower_idx, 0.0
    weight = float((target - lower_val) / (upper_val - lower_val))
    return lower_idx, pos, weight


def resample_cube_to_target_grid(
    cube: Dict[str, Any],
    target_lat_axis: Sequence[float],
    target_lon_axis: Sequence[float],
    field_methods: Dict[str, str],
    logger: logging.Logger,
    label: str,
    grid_deg: float,
) -> None:
    if not cube or not field_methods:
        return

    grid_block = cube.get("grid") or {}
    source_lats_seq = grid_block.get("lats") or []
    source_lons_seq = grid_block.get("lons") or []

    if not source_lats_seq or not source_lons_seq:
        flat_lats, flat_lons = build_flat_grid(target_lat_axis, target_lon_axis)
        cube.setdefault("grid", {})["lats"] = flat_lats
        cube.setdefault("grid", {})["lons"] = flat_lons
        return

    source_lats = np.asarray(source_lats_seq, dtype=float)
    source_lons = np.asarray(source_lons_seq, dtype=float)
    if source_lats.shape != source_lons.shape:
        logger.warning("%s grid lats/lons shape mismatch; forcing nearest resample.", label)

    flat_target_lats, flat_target_lons = build_flat_grid(target_lat_axis, target_lon_axis)
    target_cell_count = len(flat_target_lats)

    if len(source_lats_seq) == target_cell_count:
        matches = all(
            math.isclose(sl, tl, abs_tol=1e-6) and math.isclose(slon, tlon, abs_tol=1e-6)
            for sl, slon, tl, tlon in zip(source_lats_seq, source_lons_seq, flat_target_lats, flat_target_lons)
        )
        if matches:
            # Already aligned; ensure grid entries use canonical lat-major values.
            grid_block["lats"] = flat_target_lats
            grid_block["lons"] = flat_target_lons
            return

    rounded_lat = np.round(source_lats, 6)
    rounded_lon = np.round(source_lons, 6)
    unique_lat, lat_inverse = np.unique(rounded_lat, return_inverse=True)
    unique_lon, lon_inverse = np.unique(rounded_lon, return_inverse=True)
    lat_count = unique_lat.size
    lon_count = unique_lon.size
    cell_count = len(source_lats_seq)
    rectangular = lat_count * lon_count == cell_count

    if rectangular:
        lat_axis_values = np.array(
            [float(np.mean(source_lats[lat_inverse == idx])) for idx in range(lat_count)], dtype=float
        )
        lon_axis_values = np.array(
            [float(np.mean(source_lons[lon_inverse == idx])) for idx in range(lon_count)], dtype=float
        )
        lat_weights = [_compute_axis_weights(lat_axis_values, float(lat)) for lat in target_lat_axis]
        lon_weights = [_compute_axis_weights(lon_axis_values, float(lon)) for lon in target_lon_axis]
        source_index_grid = np.full((lat_count, lon_count), -1, dtype=int)
        for src_idx in range(cell_count):
            li = lat_inverse[src_idx]
            lj = lon_inverse[src_idx]
            source_index_grid[li, lj] = src_idx
    else:
        lat_axis_values = np.array([], dtype=float)
        lon_axis_values = np.array([], dtype=float)
        lat_weights = []
        lon_weights = []
        source_index_grid = np.full((0, 0), -1, dtype=int)

    source_pairs = np.column_stack((source_lats, source_lons))
    target_pairs = np.column_stack((flat_target_lats, flat_target_lons))
    fallback_indices: List[int] = []
    for t_lat, t_lon in target_pairs:
        diffs = source_pairs - np.array([t_lat, t_lon])
        distances = np.sum(diffs * diffs, axis=1)
        fallback_indices.append(int(np.argmin(distances)))

    values_block = cube.get("values") or {}
    region_means = cube.get("region_mean") or {}
    updated_region_means: Dict[str, List[Optional[float]]] = {}

    for field, method in field_methods.items():
        series = values_block.get(field)
        if series is None:
            continue
        resampled_series: List[Optional[List[Optional[float]]]] = []
        for time_slice in series:
            if not isinstance(time_slice, Sequence) or isinstance(time_slice, (bytes, str)):
                resampled_series.append(None)
                continue

            numeric = np.full(cell_count, np.nan, dtype=float)
            limit = min(len(time_slice), cell_count)
            for idx in range(limit):
                value = time_slice[idx]
                if value is None:
                    continue
                try:
                    numeric[idx] = float(value)
                except (TypeError, ValueError):
                    continue

            if rectangular:
                slice_grid = np.full((lat_count, lon_count), np.nan, dtype=float)
                for src_idx in range(cell_count):
                    li = lat_inverse[src_idx]
                    lj = lon_inverse[src_idx]
                    slice_grid[li, lj] = numeric[src_idx]
            else:
                slice_grid = None

            resampled_slice: List[Optional[float]] = []
            use_bilinear = (
                method == "bilinear"
                and rectangular
                and lat_count > 1
                and lon_count > 1
                and slice_grid is not None
            )

            fallback_needed = False
            cell_pos = 0
            if use_bilinear:
                for lat_idx, (lat_lo, lat_hi, lat_w) in enumerate(lat_weights):
                    for lon_idx, (lon_lo, lon_hi, lon_w) in enumerate(lon_weights):
                        neighbors = [
                            slice_grid[lat_lo, lon_lo],
                            slice_grid[lat_lo, lon_hi],
                            slice_grid[lat_hi, lon_lo],
                            slice_grid[lat_hi, lon_hi],
                        ]
                        value: float
                        if lat_lo == lat_hi and lon_lo == lon_hi:
                            value = neighbors[0]
                        elif lat_lo == lat_hi:
                            v0, v1 = neighbors[0], neighbors[1]
                            if np.isnan(v0) or np.isnan(v1):
                                value = np.nan
                            else:
                                value = v0 + (v1 - v0) * lon_w
                        elif lon_lo == lon_hi:
                            v0, v1 = neighbors[0], neighbors[2]
                            if np.isnan(v0) or np.isnan(v1):
                                value = np.nan
                            else:
                                value = v0 + (v1 - v0) * lat_w
                        else:
                            v00, v01, v10, v11 = neighbors
                            if (
                                np.isnan(v00)
                                or np.isnan(v01)
                                or np.isnan(v10)
                                or np.isnan(v11)
                            ):
                                value = np.nan
                            else:
                                i0 = v00 + (v01 - v00) * lon_w
                                i1 = v10 + (v11 - v10) * lon_w
                                value = i0 + (i1 - i0) * lat_w

                        if np.isnan(value):
                            fallback_needed = True
                            fallback_idx = fallback_indices[cell_pos]
                            fallback_val = numeric[fallback_idx] if 0 <= fallback_idx < numeric.size else np.nan
                            value = fallback_val

                        if np.isnan(value):
                            resampled_slice.append(None)
                        else:
                            resampled_slice.append(float(value))
                        cell_pos += 1
            else:
                for cell_pos, fallback_idx in enumerate(fallback_indices):
                    fallback_val = numeric[fallback_idx] if 0 <= fallback_idx < numeric.size else np.nan
                    if np.isnan(fallback_val):
                        resampled_slice.append(None)
                    else:
                        resampled_slice.append(float(fallback_val))

            resampled_series.append(resampled_slice)

        values_block[field] = resampled_series

        means: List[Optional[float]] = []
        for slice_vals in resampled_series:
            if not isinstance(slice_vals, list):
                means.append(None)
                continue
            filtered = [float(val) for val in slice_vals if val is not None]
            if not filtered:
                means.append(None)
            else:
                if field == "wind_direction_10m":
                    radians = np.radians(filtered)
                    sin_sum = float(np.sum(np.sin(radians)))
                    cos_sum = float(np.sum(np.cos(radians)))
                    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
                        means.append(None)
                    else:
                        angle = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
                        if angle < 0:
                            angle += 360.0
                        means.append(angle)
                else:
                    means.append(float(np.mean(filtered)))
        updated_region_means[field] = means

    if updated_region_means:
        region_means.update(updated_region_means)
        cube["region_mean"] = region_means

    grid_block["lats"] = flat_target_lats
    grid_block["lons"] = flat_target_lons
    cube["grid"] = grid_block
    logger.info("%s resampled to %.1f°", label, grid_deg)

def validate_cube_dimensions(
    cube: Dict[str, Any],
    lat_axis: Sequence[float],
    lon_axis: Sequence[float],
    kind: str,
) -> None:
    expected_points = len(lat_axis) * len(lon_axis)
    grid_block = cube.get("grid") or {}
    lats_flat = grid_block.get("lats") or []
    lons_flat = grid_block.get("lons") or []
    if len(lats_flat) != expected_points or len(lons_flat) != expected_points:
        raise OpenMeteoError(
            f"{kind} grid mismatch: expected {expected_points} points but got "
            f"{len(lats_flat)} latitude and {len(lons_flat)} longitude entries."
        )
    values_block = cube.get("values") or {}
    sample_series = next((series for series in values_block.values() if series), None)
    if sample_series and isinstance(sample_series, Sequence):
        first_slice = sample_series[0] if sample_series else None
        if (
            isinstance(first_slice, Sequence)
            and not isinstance(first_slice, (str, bytes, bytearray))
        ):
            slice_len = len(first_slice)
            if slice_len != expected_points:
                raise OpenMeteoError(
                    f"{kind} value slice mismatch: expected {expected_points} samples but got {slice_len}."
                )

def chunk_pairs(
    flat_latitudes: Sequence[float],
    flat_longitudes: Sequence[float],
    max_pairs_per_call: int,
) -> Iterable[CoordinateChunk]:
    if max_pairs_per_call <= 0:
        raise ValueError("max_pairs_per_call must be positive.")
    if len(flat_latitudes) != len(flat_longitudes):
        raise ValueError("Latitude and longitude sequences must have equal length.")

    total = len(flat_latitudes)
    for chunk_id, start in enumerate(range(0, total, max_pairs_per_call), start=1):
        end = min(start + max_pairs_per_call, total)
        lat_slice = [round(float(lat), 6) for lat in flat_latitudes[start:end]]
        lon_slice = [round(float(lon), 6) for lon in flat_longitudes[start:end]]
        yield CoordinateChunk(chunk_id, start, end, lat_slice, lon_slice)


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_timestamp_kst(dt: datetime) -> str:
    return dt.astimezone(ZONE_KST).isoformat(timespec="minutes")


def parse_api_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZONE_KST)
    return dt.astimezone(ZONE_KST)


def flatten_numeric_sequence(value: Any) -> Optional[List[Optional[float]]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            return [float(value)]
        except ValueError:
            return [None]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        flat: List[Optional[float]] = []
        for item in value:
            nested = flatten_numeric_sequence(item)
            if nested is None:
                flat.append(None)
            else:
                flat.extend(nested)
        return flat
    return None


def mean_ignore_missing(values: Optional[Sequence[Optional[float]]]) -> Optional[float]:
    if not values:
        return None
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def circular_mean_deg(values: Optional[Sequence[Optional[float]]]) -> Optional[float]:
    if not values:
        return None
    filtered = [float(v) % 360.0 for v in values if v is not None]
    if not filtered:
        return None
    radians = np.radians(filtered)
    sin_sum = float(np.sum(np.sin(radians)))
    cos_sum = float(np.sum(np.cos(radians)))
    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return None
    angle = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
    if angle < 0:
        angle += 360.0
    return angle


def normalize_hourly_series(data: Any, time_count: int) -> List[Any]:
    if not isinstance(data, list):
        return [None] * time_count
    if len(data) == time_count:
        return data
    if time_count and len(data) % time_count == 0:
        cell_count = len(data) // time_count
        return [data[idx * cell_count : (idx + 1) * cell_count] for idx in range(time_count)]
    return data[:time_count]


def prepare_cube_payload(
    payload: Dict[str, Any],
    target_datetimes: List[datetime],
    bbox: BoundingBox,
    grid_deg: float,
    fields: Sequence[str],
    logger: logging.Logger,
    request_url: str,
    source: str,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, int]:
    hourly = payload.get("hourly") or {}
    time_strings = hourly.get("time")
    if not time_strings:
        raise OpenMeteoError(
            f"Open-Meteo response missing hourly.time array for {source}. url={request_url}"
        )

    parsed_times = [parse_api_time(ts) for ts in time_strings]
    time_lookup = {format_timestamp_kst(dt): index for index, dt in enumerate(parsed_times)}

    timestamps_kst = [format_timestamp_kst(dt) for dt in target_datetimes]
    selected_indices: List[Optional[int]] = []
    ok_count = 0
    for ts in timestamps_kst:
        index = time_lookup.get(ts)
        if index is None:
            logger.warning("%s missing timestamp %s; inserting null slice.", source, ts)
        else:
            ok_count += 1
        selected_indices.append(index)

    grid_latitudes_raw = payload.get("latitude")
    grid_longitudes_raw = payload.get("longitude")
    grid_latitudes_flat = flatten_numeric_sequence(grid_latitudes_raw)
    grid_longitudes_flat = flatten_numeric_sequence(grid_longitudes_raw)

    cell_count = 0
    if grid_latitudes_flat:
        cell_count = len(grid_latitudes_flat)
    if grid_longitudes_flat:
        if cell_count == 0:
            cell_count = len(grid_longitudes_flat)
        elif len(grid_longitudes_flat) != cell_count:
            logger.warning(
                "%s longitude count %s differs from latitude count %s.",
                source,
                len(grid_longitudes_flat),
                cell_count,
            )

    values: Dict[str, List[Optional[List[Optional[float]]]]] = {field: [] for field in fields}
    region_mean: Dict[str, List[Optional[float]]] = {field: [] for field in fields}

    time_count = len(parsed_times)
    for field in fields:
        if field not in hourly:
            logger.warning("%s response missing field '%s'.", source, field)
        field_series = normalize_hourly_series(hourly.get(field), time_count)

        for idx in selected_indices:
            if idx is None or idx >= len(field_series):
                values[field].append(None)
                region_mean[field].append(None)
                continue

            entry = field_series[idx]
            flat_values = flatten_numeric_sequence(entry)
            if flat_values is None:
                values[field].append(None)
                region_mean[field].append(None)
                continue

            numeric_flat: List[Optional[float]] = []
            for val in flat_values:
                if val is None:
                    numeric_flat.append(None)
                else:
                    try:
                        numeric_flat.append(float(val))
                    except (TypeError, ValueError):
                        numeric_flat.append(None)

            values[field].append(numeric_flat)

            if numeric_flat:
                slice_len = len(numeric_flat)
                if cell_count == 0:
                    cell_count = slice_len
                elif slice_len != cell_count:
                    logger.warning(
                        "%s field %s returned %s cells; expected %s.",
                        source,
                        field,
                        slice_len,
                        cell_count,
                    )

            if field == "wind_direction_10m":
                region_mean[field].append(circular_mean_deg(numeric_flat))
            else:
                region_mean[field].append(mean_ignore_missing(numeric_flat))

    cube: Dict[str, Any] = {
        "source": source,
        "bbox": bbox.as_dict(),
        "grid_deg": float(grid_deg),
        "timestamps_kst": timestamps_kst,
        "fields": list(fields),
        "grid": {
            "lats": grid_latitudes_raw if grid_latitudes_raw is not None else [],
            "lons": grid_longitudes_raw if grid_longitudes_raw is not None else [],
        },
        "values": values,
        "region_mean": region_mean,
    }
    if model:
        cube["model"] = model
    return cube, ok_count, len(timestamps_kst) - ok_count


def fetch_openmeteo(base_url: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    try:
        response = requests.get(base_url, params=params, timeout=60)
    except requests.RequestException as exc:
        raise OpenMeteoNetworkError(f"Network error while calling Open-Meteo: {exc}") from exc

    if response.status_code == 401:
        raise OpenMeteoAuthError("Unauthorized: check Open-Meteo credentials or token.")
    if response.status_code == 403:
        raise OpenMeteoAuthError("Forbidden: access denied by Open-Meteo API.")
    if response.status_code == 429:
        raise OpenMeteoRateLimitError(f"Rate limited by Open-Meteo (HTTP 429) url={response.url}")
    if 400 <= response.status_code < 500:
        raise OpenMeteoClientError(f"Open-Meteo error: HTTP {response.status_code} url={response.url}")
    if response.status_code >= 500:
        raise OpenMeteoError(f"Server error: HTTP {response.status_code}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenMeteoError("Open-Meteo response was not valid JSON.") from exc
    return payload, response.url



def fetch_coordinate_chunks(
    base_url: str,
    fields: Sequence[str],
    flat_latitudes: Sequence[float],
    flat_longitudes: Sequence[float],
    params_base: Dict[str, Any],
    logger: logging.Logger,
    kind: str,
    max_pairs: int = MAX_COORDS_PER_REQUEST,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    coordinate_chunks: List[CoordinateChunk] = list(chunk_pairs(flat_latitudes, flat_longitudes, max_pairs))
    total_chunks = len(coordinate_chunks)
    if total_chunks == 0:
        logger.info("%s no coordinate chunks to fetch.", kind.upper())
        return [], [], 0

    results_by_index: Dict[int, Dict[str, Any]] = {}
    pending: List[CoordinateChunk] = coordinate_chunks[:]
    max_outer_rounds = 10

    for outer_round in range(1, max_outer_rounds + 1):
        if not pending:
            break
        logger.info("Starting %s fetch round %s/%s", kind.upper(), outer_round, max_outer_rounds)
        next_pending: List[CoordinateChunk] = []

        for chunk in pending:
            chunk_idx = chunk.chunk_id
            chunk_lats = chunk.latitudes
            chunk_lons = chunk.longitudes
            chunk_label = f"{kind.upper()} chunk {chunk_idx}"
            success = False
            last_error: Optional[Exception] = None

            for attempt in range(3):
                params = dict(params_base)
                params["latitude"] = ",".join(f"{lat:.6f}" for lat in chunk_lats)
                params["longitude"] = ",".join(f"{lon:.6f}" for lon in chunk_lons)
                try:
                    payload, url = fetch_openmeteo(base_url, params)
                    payload = normalize_payload(payload)
                    point_count = len(chunk_lats)
                    logger.info(
                        "%s %s/%s ok: points=%s (idx %s-%s)",
                        chunk_label,
                        chunk_idx,
                        total_chunks,
                        point_count,
                        chunk.start_index,
                        chunk.end_index - 1,
                    )
                    results_by_index[chunk_idx] = {
                        "chunk": chunk_idx,
                        "payload": payload,
                        "url": url,
                        "count": point_count,
                        "fields": list(fields),
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                    }
                    chunk_pause()
                    success = True
                    break
                except OpenMeteoRateLimitError as exc:
                    last_error = exc
                    logger.warning(
                        "%s retry after 429 (attempt %s/3)",
                        chunk_label,
                        attempt + 1,
                    )
                    chunk_pause(rate_limited=True)
                    backoff_sleep(attempt)
                    continue
                except OpenMeteoNetworkError as exc:
                    last_error = exc
                    logger.warning(
                        "%s retry after transient error (attempt %s/3): %s",
                        chunk_label,
                        attempt + 1,
                        exc,
                    )
                    chunk_pause()
                    backoff_sleep(attempt)
                    continue
                except OpenMeteoAuthError:
                    raise
                except OpenMeteoClientError as exc:
                    last_error = exc
                    if "HTTP 400" in str(exc) and attempt < 2:
                        logger.warning(
                            "%s retry after 400 (attempt %s/3)",
                            chunk_label,
                            attempt + 1,
                        )
                        chunk_pause()
                        backoff_sleep(attempt)
                        continue
                    logger.warning("%s failed with client error: %s", chunk_label, exc)
                    break
                except (OpenMeteoError, RuntimeError) as exc:
                    last_error = exc
                    logger.warning(
                        "%s retry after transient error (attempt %s/3): %s",
                        chunk_label,
                        attempt + 1,
                        exc,
                    )
                    chunk_pause()
                    backoff_sleep(attempt)
                    continue

            if not success:
                logger.warning(
                    "%s giving up after 3 attempts (idx %s-%s).",
                    chunk_label,
                    chunk.start_index,
                    chunk.end_index - 1,
                )
                if last_error:
                    logger.warning("%s last error: %s", chunk_label, last_error)
                next_pending.append(chunk)

        if not next_pending:
            logger.info(
                "%s all chunks completed successfully at round %s",
                kind.upper(),
                outer_round,
            )
            pending = []
            break

        if outer_round < max_outer_rounds:
            logger.warning(
                "%s %s chunks failed; retrying after 5s cooldown (round %s/%s)",
                kind.upper(),
                len(next_pending),
                outer_round,
                max_outer_rounds,
            )
            time.sleep(5)
            pending = next_pending
        else:
            pending = next_pending
            logger.warning(
                "%s giving up after %s rounds. %s chunks still failed.",
                kind.upper(),
                max_outer_rounds,
                len(pending),
            )

    failure_chunks = [chunk.chunk_id for chunk in pending]
    results = [results_by_index[idx] for idx in sorted(results_by_index)]
    return results, failure_chunks, total_chunks


def merge_coordinate_chunks(
    chunk_results: List[Dict[str, Any]],
    fields: Sequence[str],
    logger: logging.Logger,
    kind: str,
    grid_deg: float,
    total_points: int,
    ny: int,
    nx: int,
    flat_latitudes: Sequence[float],
    flat_longitudes: Sequence[float],
    max_pairs: int,
    failed_chunks: List[int],
    total_chunks: int,
) -> Tuple[Dict[str, Any], int]:
    if nx <= 0 or ny <= 0:
        raise OpenMeteoError("Grid dimensions must be positive.")

    usable_chunks: List[Tuple[int, Dict[str, Any], List[str], int, int]] = []
    final_times: Optional[List[str]] = None

    for result in chunk_results:
        chunk_idx = result["chunk"]
        start_index = int(result.get("start_index", 0))
        end_index = int(result.get("end_index", start_index))
        payload = normalize_payload(result["payload"])
        hourly = payload.get("hourly") or {}
        times = list(hourly.get("time") or [])
        if not times:
            logger.warning("%s chunk %s missing hourly.time; marking as failed.", kind, chunk_idx)
            failed_chunks.append(chunk_idx)
            continue
        if final_times is None:
            final_times = times
        else:
            time_set = set(times)
            intersection = [ts for ts in final_times if ts in time_set]
            if len(intersection) != len(final_times):
                logger.warning(
                    "%s chunk %s time mismatch; intersect %s -> %s",
                    kind,
                    chunk_idx,
                    len(final_times),
                    len(intersection),
                )
            final_times = intersection
        usable_chunks.append((chunk_idx, payload, times, start_index, end_index))

    if not usable_chunks or not final_times:
        raise OpenMeteoError(f"No successful {kind.lower()} chunks available.")

    time_count = len(final_times)
    total_points = int(total_points)
    nx = int(nx)
    ny = int(ny)

    hourly_grids: Dict[str, List[List[List[Optional[float]]]]] = {
        field: [[[None for _ in range(nx)] for _ in range(ny)] for _ in range(time_count)]
        for field in fields
    }

    # NOTE: No interpolation/regridding. Data placed by exact coordinate index.
    for chunk_idx, payload, times, start_index, end_index in usable_chunks:
        width = max(end_index - start_index, 0)
        hourly = payload.get("hourly") or {}
        time_lookup = {ts: pos for pos, ts in enumerate(times)}
        selected_positions = [time_lookup.get(ts) for ts in final_times]
        series_length = len(times)

        for field in fields:
            field_series = normalize_hourly_series(hourly.get(field), series_length)
            for time_pos, source_index in enumerate(selected_positions):
                if source_index is None or source_index >= len(field_series):
                    continue
                entry = field_series[source_index]
                slice_values = flatten_numeric_sequence(entry)
                if slice_values is None:
                    continue
                limit = min(len(slice_values), width)
                for offset in range(limit):
                    global_index = start_index + offset
                    if global_index >= total_points:
                        break
                    value = slice_values[offset]
                    numeric: Optional[float]
                    if value is None:
                        numeric = None
                    else:
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            numeric = None
                    row = global_index // nx
                    col = global_index % nx
                    if 0 <= row < ny and 0 <= col < nx:
                        hourly_grids[field][time_pos][row][col] = numeric

    def flatten_time_slice(slice_grid: List[List[Optional[float]]]) -> List[Optional[float]]:
        flat: List[Optional[float]] = []
        for row in slice_grid:
            flat.extend(row)
        return flat

    hourly_combined: Dict[str, Any] = {"time": list(final_times)}
    for field in fields:
        hourly_combined[field] = [flatten_time_slice(grid) for grid in hourly_grids[field]]

    first_payload = usable_chunks[0][1]
    merged_payload: Dict[str, Any] = {}
    for key, value in first_payload.items():
        if key in {"latitude", "longitude", "hourly"}:
            continue
        merged_payload[key] = value

    merged_payload["latitude"] = [round(float(lat), 6) for lat in flat_latitudes]
    merged_payload["longitude"] = [round(float(lon), 6) for lon in flat_longitudes]
    merged_payload["hourly"] = hourly_combined

    meta = dict(merged_payload.get("meta", {}))
    meta.update(
        {
            "grid_deg": float(grid_deg),
            "points": total_points,
            "chunk_size": int(max_pairs),
            "status": "OK" if not failed_chunks else "DATA-SPARSE",
            "failed_chunks": sorted(set(failed_chunks)),
            "chunks": int(total_chunks),
        }
    )
    merged_payload["meta"] = meta
    return merged_payload, total_points


def create_output_paths(out_date: str, ts_str: str, label: str) -> Tuple[Path, Path]:
    ts_compact = ts_str.replace(":", "")
    base_dir = Path("data") / "raw" / "openmeteo"
    raw_path = base_dir / f"openmeteo_{label}_{out_date}_{ts_compact}_raw.json"
    cube_path = base_dir / f"openmeteo_{label}_{out_date}_{ts_compact}_cube.json"
    return raw_path, cube_path


def create_combined_cube_path(out_date: str, ts_str: str) -> Path:
    ts_compact = ts_str.replace(":", "")
    base_dir = Path("data") / "raw" / "openmeteo"
    return base_dir / f"openmeteo_{out_date}_{ts_compact}_cube.json"


def build_combined_cube(
    air_cube: Dict[str, Any],
    weather_cube: Dict[str, Any],
    air_raw_path: Path,
    weather_raw_path: Path,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if air_cube.get("timestamps_kst") != weather_cube.get("timestamps_kst"):
        logger.warning("Skipping combined cube due to mismatched timestamps.")
        return None
    if air_cube.get("grid") != weather_cube.get("grid"):
        logger.warning("Skipping combined cube due to mismatched grids.")
        return None

    combined_fields = list(dict.fromkeys(COMBINED_FIELDS))
    combined_values: Dict[str, Any] = {}
    for field in AIR_QUALITY_FIELDS:
        combined_values[field] = air_cube["values"].get(field)
    for field in WEATHER_FIELDS:
        combined_values[field] = weather_cube["values"].get(field)

    combined_region_mean = dict(air_cube.get("region_mean", {}))
    combined_region_mean.update(weather_cube.get("region_mean", {}))

    combined_cube: Dict[str, Any] = {
        "source": "combined",
        "bbox": air_cube.get("bbox"),
        "grid_deg": air_cube.get("grid_deg"),
        "timestamps_kst": air_cube.get("timestamps_kst"),
        "fields": combined_fields,
        "grid": air_cube.get("grid"),
        "values": combined_values,
        "region_mean": combined_region_mean,
        "sources": {
            "air_quality": str(air_raw_path),
            "weather": str(weather_raw_path),
        },
    }
    if air_cube.get("model"):
        combined_cube["model"] = air_cube["model"]
    return combined_cube

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo air quality and weather cubes.")
    parser.add_argument("--ts", required=True, help="Target clock time (HH:MM) in KST.")
    parser.add_argument(
        "--out_date",
        help="Target date in YYYYMMDD (defaults to current date in KST).",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box override (lon_min,lat_min,lon_max,lat_max). Defaults to REGION_BBOX from environment.",
    )
    parser.add_argument(
        "--grid_deg",
        type=float,
        help="Grid spacing in degrees (defaults to GRID_DEG from environment).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv(dotenv_path=Path("config/.env"))

    air_base_url = os.getenv("OPENMETEO_BASE")
    if not air_base_url:
        print("Missing OPENMETEO_BASE in environment.", file=sys.stderr)
        return 1

    weather_base_url = os.getenv("OPENMETEO_BASE_WEATHER")
    if not weather_base_url:
        print("Missing OPENMETEO_BASE_WEATHER in environment.", file=sys.stderr)
        return 1

    timezone_name = os.getenv("TZ")
    if not timezone_name:
        print("Missing TZ in environment.", file=sys.stderr)
        return 1
    try:
        timezone_info = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        print(f"Invalid TZ in environment: {timezone_name}", file=sys.stderr)
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

    grid_deg = args.grid_deg if args.grid_deg is not None else float(os.getenv("GRID_DEG", DEFAULT_GRID_DEG))

    logger = configure_logger(out_date_str)
    logger.info("Starting Open-Meteo fetch for target %s", target_dt.isoformat())

    target_timestamps = build_target_timestamps(target_dt)
    start_ts = target_timestamps[0]
    end_ts = target_timestamps[-1]
    end_query_ts = end_ts + timedelta(minutes=1)

    try:
        lat_list, lon_list = generate_grid_points(bbox.south_west_north_east(), grid_deg)
    except ValueError as exc:
        logger.error("Invalid grid configuration: %s", exc)
        print(str(exc), file=sys.stderr)
        return 1

    ny = len(lat_list)
    nx = len(lon_list)
    total_points = ny * nx
    flat_latitudes, flat_longitudes = build_flat_grid(lat_list, lon_list)
    max_pairs = MAX_COORDS_PER_REQUEST
    num_chunks_est = math.ceil(total_points / max_pairs) if total_points else 0
    logger.info(
        "Grid size: %d×%d, total_pairs=%d, max_pairs=%d, num_chunks=%d",
        ny,
        nx,
        total_points,
        max_pairs,
        num_chunks_est,
    )

    air_params_base = {
        "hourly": ",".join(AIR_QUALITY_FIELDS),
        "timezone": timezone_name,
        "domains": "cams_global",
        "start_hour": format_hour_param(start_ts, timezone_info),
        "end_hour": format_hour_param(end_query_ts, timezone_info),
    }

    weather_params_base = {
        "hourly": ",".join(WEATHER_FIELDS),
        "timezone": timezone_name,
        "start_hour": format_hour_param(start_ts, timezone_info),
        "end_hour": format_hour_param(end_query_ts, timezone_info),
    }

    try:
        air_chunk_results, air_failed_chunks, total_air_chunks = fetch_coordinate_chunks(
            air_base_url,
            AIR_QUALITY_FIELDS,
            flat_latitudes,
            flat_longitudes,
            air_params_base,
            logger,
            "AQ",
            max_pairs,
        )
    except OpenMeteoAuthError as exc:
        logger.error("Air-quality authentication error: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    except OpenMeteoError as exc:
        logger.error("Failed to fetch air-quality chunks: %s", exc)
        print(str(exc), file=sys.stderr)
        return 3

    if not air_chunk_results:
        message = "No air-quality chunks succeeded; aborting."
        logger.error(message)
        print(message, file=sys.stderr)
        return 3

    time.sleep(2.0)

    try:
        weather_chunk_results, weather_failed_chunks, total_weather_chunks = fetch_coordinate_chunks(
            weather_base_url,
            WEATHER_FIELDS,
            flat_latitudes,
            flat_longitudes,
            weather_params_base,
            logger,
            "WX",
            max_pairs,
        )
    except OpenMeteoAuthError as exc:
        logger.error("Weather authentication error: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    except OpenMeteoError as exc:
        logger.error("Failed to fetch weather chunks: %s", exc)
        print(str(exc), file=sys.stderr)
        return 3

    if not weather_chunk_results:
        message = "No weather chunks succeeded; aborting."
        logger.error(message)
        print(message, file=sys.stderr)
        return 3

    air_failed_list = list(air_failed_chunks)
    weather_failed_list = list(weather_failed_chunks)

    try:
        air_payload, air_cell_total = merge_coordinate_chunks(
            air_chunk_results,
            AIR_QUALITY_FIELDS,
            logger,
            "AQ",
            grid_deg,
            total_points,
            ny,
            nx,
            flat_latitudes,
            flat_longitudes,
            max_pairs,
            air_failed_list,
            total_air_chunks,
        )
    except OpenMeteoError as exc:
        logger.error("Failed to merge air-quality chunks: %s", exc)
        print(str(exc), file=sys.stderr)
        return 4

    try:
        weather_payload, weather_cell_total = merge_coordinate_chunks(
            weather_chunk_results,
            WEATHER_FIELDS,
            logger,
            "WX",
            grid_deg,
            total_points,
            ny,
            nx,
            flat_latitudes,
            flat_longitudes,
            max_pairs,
            weather_failed_list,
            total_weather_chunks,
        )
    except OpenMeteoError as exc:
        logger.error("Failed to merge weather chunks: %s", exc)
        print(str(exc), file=sys.stderr)
        return 4

    air_raw_path, air_cube_path = create_output_paths(out_date_str, args.ts, "airquality")
    weather_raw_path, weather_cube_path = create_output_paths(out_date_str, args.ts, "weather")
    combined_cube_path = create_combined_cube_path(out_date_str, args.ts)

    save_json(air_raw_path, air_payload)
    save_json(weather_raw_path, weather_payload)

    air_url_ref = air_chunk_results[0]["url"] if air_chunk_results else "air-quality-chunks"
    weather_url_ref = weather_chunk_results[0]["url"] if weather_chunk_results else "weather-chunks"

    try:
        air_cube, air_ok, air_miss = prepare_cube_payload(
            air_payload,
            target_timestamps,
            bbox,
            grid_deg,
            AIR_QUALITY_FIELDS,
            logger,
            air_url_ref,
            source="air-quality",
            model="cams_global",
        )
    except OpenMeteoError as exc:
        logger.error("Failed to prepare air-quality cube: %s", exc)
        print(str(exc), file=sys.stderr)
        return 4

    air_field_methods = {field: "bilinear" for field in AIR_QUALITY_FIELDS}
    resample_cube_to_target_grid(
        air_cube,
        lat_list,
        lon_list,
        air_field_methods,
        logger,
        "OM AQ",
        grid_deg,
    )
    validate_cube_dimensions(air_cube, lat_list, lon_list, "AQ")

    try:
        weather_cube, weather_ok, weather_miss = prepare_cube_payload(
            weather_payload,
            target_timestamps,
            bbox,
            grid_deg,
            WEATHER_FIELDS,
            logger,
            weather_url_ref,
            source="weather",
        )
    except OpenMeteoError as exc:
        logger.error("Failed to prepare weather cube: %s", exc)
        print(str(exc), file=sys.stderr)
        return 4

    weather_field_methods = {
        "wind_speed_10m": "bilinear",
        "wind_direction_10m": "nearest",
    }
    resample_cube_to_target_grid(
        weather_cube,
        lat_list,
        lon_list,
        weather_field_methods,
        logger,
        "OM WX",
        grid_deg,
    )
    validate_cube_dimensions(weather_cube, lat_list, lon_list, "WX")

    save_json(air_cube_path, air_cube)
    save_json(weather_cube_path, weather_cube)

    combined_cube = build_combined_cube(
        air_cube,
        weather_cube,
        air_raw_path,
        weather_raw_path,
        logger,
    )
    if combined_cube is not None:
        save_json(combined_cube_path, combined_cube)

    aq_status = air_payload.get("meta", {}).get("status", "UNKNOWN")
    wx_status = weather_payload.get("meta", {}).get("status", "UNKNOWN")
    aq_failed_display = sorted(set(air_failed_list))
    wx_failed_display = sorted(set(weather_failed_list))

    logger.info(
        "AQ chunks=%s succeeded=%s failed=%s status=%s cells_total=%s",
        total_air_chunks,
        len(air_chunk_results),
        len(aq_failed_display),
        aq_status,
        air_cell_total,
    )
    logger.info(
        "WX chunks=%s succeeded=%s failed=%s grid_deg=%.3f total_points=%s status=%s",
        total_weather_chunks,
        len(weather_chunk_results),
        len(wx_failed_display),
        grid_deg,
        total_points,
        wx_status,
    )

    summary = f"AQ slices=9 ok={air_ok} miss={air_miss} | WX slices=9 ok={weather_ok} miss={weather_miss}"
    logger.info(summary)
    print(summary)
    print(f"aq_raw={air_raw_path} aq_cube={air_cube_path}")
    print(f"wx_raw={weather_raw_path} wx_cube={weather_cube_path}")
    if combined_cube is not None:
        print(f"combined_cube={combined_cube_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
