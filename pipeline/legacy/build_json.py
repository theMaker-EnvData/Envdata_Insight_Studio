#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from dotenv import load_dotenv


ZONE_KST = ZoneInfo("Asia/Seoul")
HOURLY_COUNT = 9
STEP_HOURS = 3
COMPASS_LABELS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
COMPASS_STEP = 360.0 / len(COMPASS_LABELS)


@dataclass
class City:
    name: str
    lat: float
    lon: float


def configure_logger(out_date: str) -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"pipeline_{out_date}.log"

    logger = logging.getLogger("build_json_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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
        day = datetime.now(ZONE_KST).date()

    target_dt = datetime.combine(day, target_time, tzinfo=ZONE_KST)
    return day.strftime("%Y%m%d"), target_dt


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cities(path: Path) -> Dict[str, List[City]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cities: Dict[str, List[City]] = {}
    for country_code, entries in raw.items():
        country_cities: List[City] = []
        for entry in entries or []:
            country_cities.append(
                City(
                    name=entry.get("name"),
                    lat=float(entry.get("lat")),
                    lon=float(entry.get("lon")),
                )
            )
        cities[country_code] = country_cities
    return cities


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(ZONE_KST).isoformat()


def format_timestamp_minutes(dt: datetime) -> str:
    return dt.astimezone(ZONE_KST).isoformat(timespec="minutes")


def build_target_timestamps(start_dt: datetime) -> List[datetime]:
    return [start_dt + timedelta(hours=STEP_HOURS * offset) for offset in range(HOURLY_COUNT)]


def make_wind_label(deg: Optional[float]) -> Optional[str]:
    """Map degrees to 16-point compass labels."""
    if deg is None or not isinstance(deg, (int, float)) or math.isnan(deg):
        return None
    normalized = deg % 360.0
    index = int((normalized + COMPASS_STEP / 2.0) // COMPASS_STEP) % len(COMPASS_LABELS)
    return COMPASS_LABELS[index]


def nearest_cell_value(
    lats: Sequence[float],
    lons: Sequence[float],
    values: Sequence[Optional[float]],
    city_lat: float,
    city_lon: float,
) -> Optional[float]:
    if not lats or not lons or not values:
        return None
    if len(lats) != len(lons) or len(lats) != len(values):
        return None
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    vals_arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(vals_arr)
    if not np.any(mask):
        return None
    lats_arr = lats_arr[mask]
    lons_arr = lons_arr[mask]
    vals_arr = vals_arr[mask]
    distances = np.sqrt((lats_arr - city_lat) ** 2 + (lons_arr - city_lon) ** 2)
    if distances.size == 0:
        return None
    return float(vals_arr[np.argmin(distances)])


def mean_ignore_missing(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def circular_mean_deg(values: Sequence[Optional[float]]) -> Optional[float]:
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


def circular_mean_series(series: Sequence[Any]) -> List[Optional[float]]:
    """Compute circular mean per entry for nested degree lists."""
    results: List[Optional[float]] = []
    for entry in series or []:
        angles: List[float] = []
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            for val in entry:
                if val is None:
                    continue
                try:
                    angles.append(float(val) % 360.0)
                except (TypeError, ValueError):
                    continue
        elif entry is not None:
            try:
                angles.append(float(entry) % 360.0)
            except (TypeError, ValueError):
                angles = []

        if not angles:
            results.append(None)
            continue

        radians = np.radians(angles)
        sin_sum = float(np.sum(np.sin(radians)))
        cos_sum = float(np.sum(np.cos(radians)))
        if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
            results.append(None)
            continue
        angle = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
        if angle < 0:
            angle += 360.0
        results.append(angle)
    return results


def compute_region_means(
    country_cities: Dict[str, List[City]],
    lats: Sequence[float],
    lons: Sequence[float],
    pm25_values: Sequence[Optional[float]],
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for code, cities in country_cities.items():
        city_values: List[Optional[float]] = []
        for city in cities:
            value = nearest_cell_value(lats, lons, pm25_values, city.lat, city.lon)
            city_values.append(value)
        result[code] = mean_ignore_missing(city_values)
    return result


def determine_max_city(
    country_cities: Dict[str, List[City]],
    lats: Sequence[float],
    lons: Sequence[float],
    pm25_values: Sequence[Optional[float]],
) -> Dict[str, Optional[Any]]:
    max_city_name: Optional[str] = None
    max_value: Optional[float] = None
    for cities in country_cities.values():
        for city in cities:
            value = nearest_cell_value(lats, lons, pm25_values, city.lat, city.lon)
            if value is None:
                continue
            if max_value is None or value > max_value:
                max_value = value
                max_city_name = city.name
    if max_city_name is None:
        return {"name": None, "value": None}
    return {"name": max_city_name, "value": max_value}


def fetch_aqicn_current(
    aqicn_rows: List[Dict[str, Any]],
    country_cities: Dict[str, List[City]],
    target_iso: str,
) -> Tuple[Dict[str, Optional[float]], Dict[str, List[Optional[float]]]]:
    city_values_by_country: Dict[str, List[Optional[float]]] = {
        code: [None] * len(cities) for code, cities in country_cities.items()
    }
    city_values_flat: Dict[str, Optional[float]] = {}

    for row in aqicn_rows:
        station = row.get("station") or {}
        name = station.get("name")
        pm25 = row.get("pm25")
        time_kst = row.get("time_kst")
        if name is None:
            continue
        city_values_flat[name] = float(pm25) if pm25 is not None else None

    region_mean: Dict[str, Optional[float]] = {}
    for code, cities in country_cities.items():
        collected: List[Optional[float]] = []
        for idx, city in enumerate(cities):
            value = city_values_flat.get(city.name)
            city_values_by_country[code][idx] = value
            collected.append(value)
        region_mean[code] = mean_ignore_missing(collected)

    return region_mean, city_values_by_country


def compute_sparse_countries(
    city_values_by_country: Dict[str, List[Optional[float]]],
) -> List[str]:
    sparse: List[str] = []
    for code, values in city_values_by_country.items():
        valid_count = sum(1 for item in values if item is not None)
        if values and valid_count < len(values) / 2:
            sparse.append(code.lower())
    return sparse


def build_tile_objects(
    timestamps: List[datetime],
    cube: Dict[str, Any],
    country_cities: Dict[str, List[City]],
    tiles_dir: Path,
    out_date_str: str,
    logger: logging.Logger,
    pm25_region_overrides: Optional[List[Optional[Dict[str, Optional[float]]]]] = None,
    max_city_overrides: Optional[List[Optional[Dict[str, Optional[Any]]]]] = None,
) -> List[Dict[str, Any]]:
    lats = cube.get("grid", {}).get("lats") or []
    lons = cube.get("grid", {}).get("lons") or []
    values = cube.get("values") or {}
    region_mean_cube = cube.get("region_mean") or {}

    pm25_series = values.get("pm2_5") or []
    wind_speed_series = values.get("wind_speed_10m") or []
    wind_dir_series = values.get("wind_direction_10m") or []
    pm25_region_means = region_mean_cube.get("pm2_5") or []
    wind_speed_means = region_mean_cube.get("wind_speed_10m") or []
    wind_dir_modes = region_mean_cube.get("wind_dir_mode_deg") or []

    tiles: List[Dict[str, Any]] = []

    for idx, ts in enumerate(timestamps):
        ts_iso = format_timestamp_minutes(ts)
        pm25_slice = pm25_series[idx] if idx < len(pm25_series) else None
        wind_speed_slice = wind_speed_series[idx] if idx < len(wind_speed_series) else None
        wind_dir_slice = wind_dir_series[idx] if idx < len(wind_dir_series) else None
        region_override = None
        if pm25_region_overrides and idx < len(pm25_region_overrides):
            region_override = pm25_region_overrides[idx]
        max_override = None
        if max_city_overrides and idx < len(max_city_overrides):
            max_override = max_city_overrides[idx]

        status = "OK"
        if pm25_slice is None:
            status = "DATA-SPARSE"

        pm25_means = None
        max_city = {"name": None, "value": None}
        if isinstance(pm25_slice, list):
            pm25_means = compute_region_means(country_cities, lats, lons, pm25_slice)
            max_city = determine_max_city(country_cities, lats, lons, pm25_slice)
        elif pm25_region_means:
            pm25_means = {}
            for code in country_cities.keys():
                pm25_means[code] = (
                    pm25_region_means[idx] if idx < len(pm25_region_means) else None
                )
        else:
            pm25_means = {code: None for code in country_cities.keys()}

        if isinstance(region_override, dict):
            pm25_means = {code: region_override.get(code) for code in country_cities.keys()}

        if isinstance(max_override, dict):
            max_city = {
                "name": max_override.get("name"),
                "value": max_override.get("value"),
            }

        mean_speed = (
            float(wind_speed_means[idx]) if idx < len(wind_speed_means) else None
        )
        mode_dir = (
            float(wind_dir_modes[idx]) if idx < len(wind_dir_modes) else None
        )
        mode_label = make_wind_label(mode_dir)

        if status != "DATA-SPARSE" and mean_speed is None and mode_dir is None:
            status = "ESTIMATED"

        tile_path = tiles_dir / f"tile_{out_date_str}_{ts.strftime('%H')}.png"

        tile_obj = {
            "ts_kst": ts_iso,
            "pm25_region_mean": pm25_means,
            "max_city": max_city,
            "wind": {
                "mean_ms": mean_speed,
                "mode_deg": mode_dir,
                "mode_lbl": mode_label,
            },
            "png": str(tile_path),
            "status": status,
        }
        tiles.append(tile_obj)

    return tiles


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined report JSON.")
    parser.add_argument("--ts", required=True, help="Target clock time (HH:MM) in KST.")
    parser.add_argument(
        "--out_date",
        help="Target date in YYYYMMDD (defaults to current date in KST).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv(dotenv_path=Path("config/.env"))

    public_dir = os.getenv("PUBLIC_DIR", "public")
    tiles_base = os.getenv("TILES_DIR", "public/tiles")
    reports_dir = os.getenv("REPORTS_DIR", "data/derived")
    bbox_str = os.getenv("REGION_BBOX")
    if not bbox_str:
        print("Missing REGION_BBOX in environment.", file=sys.stderr)
        return 1

    try:
        out_date_str, target_dt = resolve_target_datetime(args.ts, args.out_date)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        bbox_parts = [float(part.strip()) for part in bbox_str.split(",")]
        if len(bbox_parts) != 4:
            raise ValueError
        bbox = {
            "lon_min": bbox_parts[0],
            "lat_min": bbox_parts[1],
            "lon_max": bbox_parts[2],
            "lat_max": bbox_parts[3],
        }
    except (ValueError, AttributeError):
        print("Invalid REGION_BBOX format.", file=sys.stderr)
        return 1

    grid_deg = float(os.getenv("GRID_DEG", 0.4))

    logger = configure_logger(out_date_str)
    logger.info("Building report JSON for %s", target_dt.isoformat())

    aqicn_path = Path("data") / "raw" / "aqicn" / f"aqicn_{out_date_str}_0600_norm.json"
    combined_path = (
        Path("data") / "raw" / "openmeteo" / f"openmeteo_{out_date_str}_0600_cube.json"
    )
    air_path = (
        Path("data") / "raw" / "openmeteo" / f"openmeteo_airquality_{out_date_str}_0600_cube.json"
    )
    wx_path = (
        Path("data") / "raw" / "openmeteo" / f"openmeteo_weather_{out_date_str}_0600_cube.json"
    )
    aqicn_grid_path = (
        Path("data") / "derived" / f"aqicn_pm25_{out_date_str}_0600_grid0p4.json"
    )
    if not aqicn_path.exists():
        print(f"Missing AQICN file: {aqicn_path}", file=sys.stderr)
        return 2
    if not aqicn_grid_path.exists():
        print(f"Missing AQICN grid file: {aqicn_grid_path}", file=sys.stderr)
        return 2
    use_combined = combined_path.exists()
    if not use_combined and not (air_path.exists() and wx_path.exists()):
        missing: List[str] = []
        if not combined_path.exists():
            missing.append(str(combined_path))
        if not air_path.exists():
            missing.append(str(air_path))
        if not wx_path.exists():
            missing.append(str(wx_path))
        print("Missing Open-Meteo files: " + ", ".join(missing), file=sys.stderr)
        return 2

    try:
        aqicn_rows = load_json(aqicn_path)
    except (OSError, ValueError) as exc:
        logger.error("Failed to load input JSON: %s", exc)
        print(f"Failed to load input JSON: {exc}", file=sys.stderr)
        return 3

    try:
        aqicn_grid = load_json(aqicn_grid_path)
        air_cube = load_json(air_path)
        wx_cube = load_json(wx_path)
        use_combined = False
        logger.info(
            "Open-Meteo input: split cubes (%s, %s)", air_path.name, wx_path.name
        )

        timestamps_air = air_cube.get("timestamps_kst") or []
        timestamps_wx = wx_cube.get("timestamps_kst") or []
        if timestamps_air and len(timestamps_air) != HOURLY_COUNT:
            logger.warning(
                "Air-quality cube has %s timestamps; expected %s.",
                len(timestamps_air),
                HOURLY_COUNT,
            )
        if timestamps_wx and len(timestamps_wx) != HOURLY_COUNT:
            logger.warning(
                "Weather cube has %s timestamps; expected %s.",
                len(timestamps_wx),
                HOURLY_COUNT,
            )

        aq_grid = (aqicn_grid.get("grid") or {})
        air_grid = (air_cube.get("grid") or {})
        wx_grid = (wx_cube.get("grid") or {})
        aq_lats = aq_grid.get("lats") or []
        aq_lons = aq_grid.get("lons") or []
        air_lats = air_grid.get("lats") or []
        air_lons = air_grid.get("lons") or []
        wx_lats = wx_grid.get("lats") or []
        wx_lons = wx_grid.get("lons") or []

        if not aq_lats or not aq_lons:
            raise ValueError("AQICN grid missing coordinates.")
        if len(aq_lats) != len(air_lats) or len(aq_lons) != len(air_lons):
            raise ValueError("Grid mismatch between AQICN and Open-Meteo AQ.")
        if len(aq_lats) != len(wx_lats) or len(aq_lons) != len(wx_lons):
            raise ValueError("Grid mismatch between AQICN and Open-Meteo WX.")

        def coords_close(ref: Sequence[float], other: Sequence[float]) -> bool:
            return all(
                math.isclose(float(r), float(o), abs_tol=1e-6)
                for r, o in zip(ref, other)
            )

        if not coords_close(aq_lats, air_lats) or not coords_close(aq_lons, air_lons):
            raise ValueError("AQ and AQICN grids not aligned.")
        if not coords_close(aq_lats, wx_lats) or not coords_close(aq_lons, wx_lons):
            raise ValueError("WX and AQICN grids not aligned.")

        aq_pm25_flat_raw = (aqicn_grid.get("values") or {}).get("pm2_5")
        if aq_pm25_flat_raw is None:
            raise ValueError("AQICN grid missing pm2_5 values.")
        if isinstance(aq_pm25_flat_raw, list):
            aq_pm25_flat = [float(val) if val is not None else None for val in aq_pm25_flat_raw]
        else:
            aq_pm25_flat = list(aq_pm25_flat_raw)

        air_values = (air_cube.get("values") or {})
        wx_values = (wx_cube.get("values") or {})

        pm25_series_air = air_values.get("pm2_5") or []

        pm25_series: List[Optional[List[Optional[float]]]] = []
        for idx in range(HOURLY_COUNT):
            if idx == 0:
                pm25_series.append(
                    [float(val) if val is not None else None for val in aq_pm25_flat]
                )
            else:
                pm_slice = pm25_series_air[idx] if idx < len(pm25_series_air) else None
                pm25_series.append(pm_slice)

        values: Dict[str, Any] = {"pm2_5": pm25_series}
        for key in [
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "carbon_monoxide",
            "sulphur_dioxide",
        ]:
            if key in air_values and air_values.get(key) is not None:
                values[key] = air_values[key]

        for key in ["wind_speed_10m", "wind_direction_10m"]:
            if key in wx_values and wx_values.get(key) is not None:
                values[key] = wx_values[key]

        region_mean: Dict[str, Any] = {}
        wx_region = (wx_cube.get("region_mean") or {})
        if wx_region.get("wind_speed_10m") is not None:
            region_mean["wind_speed_10m"] = wx_region["wind_speed_10m"]
        if wx_region.get("wind_direction_10m") is not None:
            region_mean["wind_dir_mode_deg"] = circular_mean_series(
                wx_region.get("wind_direction_10m") or []
            )

        cube_payload = {
            "grid": {"lats": aq_lats, "lons": aq_lons},
            "values": values,
            "region_mean": region_mean,
        }
        timestamps = timestamps_air or timestamps_wx
        if timestamps:
            cube_payload["timestamps_kst"] = timestamps
        bbox_candidate = air_cube.get("bbox") or wx_cube.get("bbox")
        if bbox_candidate:
            cube_payload["bbox"] = bbox_candidate
        cube_payload["grid_deg"] = cube_payload.get("grid_deg") or grid_deg
        fields = list(values.keys())
        if fields:
            cube_payload["fields"] = fields
    except (OSError, ValueError) as exc:
        logger.error("Failed to load Open-Meteo cubes: %s", exc)
        print(f"Failed to load Open-Meteo cubes: {exc}", file=sys.stderr)
        return 3

    cities_path = Path("config") / "cities.yaml"
    try:
        country_cities = load_cities(cities_path)
    except (OSError, ValueError) as exc:
        logger.error("Failed to load cities: %s", exc)
        print(f"Failed to load cities: {exc}", file=sys.stderr)
        return 3

    region_mean_payload = cube_payload.setdefault("region_mean", {})
    if "wind_dir_mode_deg" not in region_mean_payload:
        wind_series = cube_payload.get("values", {}).get("wind_direction_10m") or []
        computed_modes: List[Optional[float]] = []
        for entry in wind_series:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                computed_modes.append(circular_mean_deg(entry))
            elif entry is None:
                computed_modes.append(None)
            else:
                try:
                    computed_modes.append(float(entry) % 360.0)
                except (TypeError, ValueError):
                    computed_modes.append(None)
        if computed_modes:
            region_mean_payload["wind_dir_mode_deg"] = computed_modes

    pm25_region_overrides: List[Dict[str, Optional[float]]] = []
    aq_region_map = (aqicn_grid.get("region_mean") or {}).get("pm2_5") or {}
    for idx in range(HOURLY_COUNT):
        if idx == 0:
            override_map = {
                code: (
                    float(aq_region_map.get(code))
                    if aq_region_map.get(code) is not None
                    else None
                )
                for code in country_cities.keys()
            }
        else:
            pm_slice = pm25_series_air[idx] if idx < len(pm25_series_air) else None
            if isinstance(pm_slice, list):
                override_map = compute_region_means(country_cities, aq_lats, aq_lons, pm_slice)
            else:
                override_map = {code: None for code in country_cities.keys()}
        pm25_region_overrides.append(override_map)

    max_city_meta = aqicn_grid.get("max_city") or {}
    max_city_overrides: List[Dict[str, Optional[Any]]] = [
        {
            "name": max_city_meta.get("name"),
            "value": max_city_meta.get("value"),
        }
    ]
    for _ in range(1, HOURLY_COUNT):
        max_city_overrides.append({"name": None, "value": None})

    city_pm25_map = aqicn_grid.get("city_pm25") or {}
    city_values_by_country: Dict[str, List[Optional[float]]] = {
        code: [
            city_pm25_map.get(city.name)
            for city in cities
        ]
        for code, cities in country_cities.items()
    }

    logger.info("build_json mixing: aqicn@t0, openmeteo@t1..t8")

    target_timestamps = build_target_timestamps(target_dt)
    timestamps_iso = [format_timestamp_minutes(ts) for ts in target_timestamps]

    tiles_dir = Path(tiles_base) / target_dt.strftime("%Y-%m-%d")
    ensure_directory(tiles_dir)

    tiles = build_tile_objects(
        target_timestamps,
        cube_payload,
        country_cities,
        tiles_dir,
        out_date_str,
        logger,
        pm25_region_overrides=pm25_region_overrides,
        max_city_overrides=max_city_overrides,
    )

    sparse_flags = compute_sparse_countries(city_values_by_country)

    wind_speed_series = region_mean_payload.get("wind_speed_10m") or []
    wind_dir_series = region_mean_payload.get("wind_dir_mode_deg") or []
    wind_mean_now = (
        float(wind_speed_series[0]) if wind_speed_series and wind_speed_series[0] is not None else None
    )
    wind_dir_now = (
        float(wind_dir_series[0]) if wind_dir_series and wind_dir_series[0] is not None else None
    )
    wind_mode_label = make_wind_label(wind_dir_now)

    current_section = {
        "ts_kst": timestamps_iso[0],
        "pm25_city": city_pm25_map,
        "pm25_region_mean": pm25_region_overrides[0],
        "max_city": max_city_overrides[0],
        "wind": {
            "mean_ms": wind_mean_now,
            "mode_deg": wind_dir_now,
            "mode_lbl": wind_mode_label,
        },
    }
    if sparse_flags:
        current_section["note"] = f"sparse_{'_'.join(sorted(sparse_flags))}"

    for tile in tiles:
        if tile["png"].startswith("public/"):
            continue
        tile["png"] = str(Path(public_dir) / Path(tile["png"]).name)

    cities_dict = {
        code: [
            {"name": city.name, "lat": city.lat, "lon": city.lon}
            for city in cities
        ]
        for code, cities in country_cities.items()
    }

    pm25_source = ["aqicn"] + ["openmeteo"] * (HOURLY_COUNT - 1)
    aqicn_method = (aqicn_grid.get("meta") or {}).get("method") or "unknown"
    meta_section = {
        "pm25_source": pm25_source,
        "max_city_tolerance_h": 24,
        "resample": {
            "aqicn": aqicn_method,
            "openmeteo": "bilinear",
        },
        "grid_deg": float(grid_deg),
    }

    report_payload = {
        "report_ts_kst": format_timestamp(target_dt),
        "bbox": {**bbox, "grid_deg": grid_deg},
        "cities": cities_dict,
        "current": current_section,
        "tiles": tiles,
        "units": {
            "pm25": "µg/m³",
            "wind_speed": "m/s",
            "wind_dir": "deg",
        },
        "meta": meta_section,
    }

    out_path = Path(reports_dir) / f"report_{out_date_str}_0600.json"
    save_json(out_path, report_payload)

    logger.info("Report written to %s", out_path)
    print(f"report_json={out_path} tiles_dir={tiles_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
