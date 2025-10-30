"""Fetch CAMS and wind data from Google Earth Engine and store GeoTIFFs locally."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time, timezone as _tz
from pathlib import Path
from typing import Sequence

from zoneinfo import ZoneInfo

import ee
import requests
from dotenv import load_dotenv

load_dotenv()

CAMS_DATASET_ID = "ECMWF/CAMS/NRT"
CAMS_BANDS = [
    "particulate_matter_d_less_than_25_um_surface",
    "particulate_matter_d_less_than_10_um_surface",
    "total_column_nitrogen_dioxide_surface",
    "total_column_sulphur_dioxide_surface",
    "total_column_carbon_monoxide_surface",
    "gems_total_column_ozone_surface",
]
CAMS_RENAME = ["pm25", "pm10", "no2", "so2", "co", "o3"]
CAMS_SCALE = 44479

WIND_BANDS = ["u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"]
WIND_RENAME = ["u10", "v10"]
DEFAULT_WIND_SCALE = 27830

DOWNLOAD_TIMEOUT = 120


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    project_root: Path
    raw_dir: Path
    cache_dir: Path
    config_dir: Path
    log_root: Path
    region_bbox: tuple[float, float, float, float]
    cutoff_label: str
    time_step_hours: int
    time_steps_count: int
    timezone_name: str
    enable_cams: bool
    wind_dataset_id: str
    log_level: int
    retry_max: int
    retry_backoff_min: int

    @classmethod
    def from_env(cls) -> Settings:
        project_root = Path(os.getenv("PROJECT_ROOT", "/root/EnvData_Insight_Studio"))
        raw_dir = Path(os.getenv("RAW_DIR", str(project_root / "data" / "raw")))
        cache_dir = Path(os.getenv("CACHE_DIR", str(project_root / "cache")))
        config_dir = Path(os.getenv("CONFIG_DIR", str(project_root / "config")))
        log_root = Path(os.getenv("LOG_DIR", str(project_root / "logs")))

        bbox_str = os.getenv("REGION_BBOX")
        if not bbox_str:
            raise ValueError("REGION_BBOX environment variable is required.")
        try:
            lon_min, lat_min, lon_max, lat_max = [float(part.strip()) for part in bbox_str.split(",")]
        except ValueError as exc:
            raise ValueError("REGION_BBOX must contain four comma-separated numbers.") from exc

        cutoff_label = os.getenv("REPORT_CUTOFF_KST", "06:00")
        time_step_hours = int(os.getenv("TIME_STEP_HOURS", "3"))
        time_steps_count = int(os.getenv("TIME_STEPS_COUNT", "9"))
        timezone_name = os.getenv("TZ", "Asia/Seoul")
        enable_cams = parse_bool(os.getenv("ENABLE_CAMS", "true"), True)

        wind_raw = os.getenv("GEE_WIND_DATASET", "NOAA/GFS0P25")
        wind_first = wind_raw.split(",")[0].strip()
        if not wind_first:
            raise ValueError("GEE_WIND_DATASET must provide at least one dataset id.")

        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        retry_max = int(os.getenv("RETRY_MAX", "3"))
        retry_backoff_min = int(os.getenv("RETRY_BACKOFF_MIN", "5"))

        return cls(
            project_root=project_root,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            config_dir=config_dir,
            log_root=log_root,
            region_bbox=(lon_min, lat_min, lon_max, lat_max),
            cutoff_label=cutoff_label,
            time_step_hours=time_step_hours,
            time_steps_count=time_steps_count,
            timezone_name=timezone_name,
            enable_cams=enable_cams,
            wind_dataset_id=wind_first,
            log_level=log_level,
            retry_max=retry_max,
            retry_backoff_min=retry_backoff_min,
        )


def configure_logger(log_dir: Path, level: int) -> logging.Logger:
    logger = logging.getLogger("fetch_gee_data")
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fetch_gee.log"

    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def initialize_earth_engine(logger: logging.Logger) -> None:
    try:
        ee.Initialize()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize Earth Engine: %s", exc)
        logger.error("Run `earthengine authenticate --auth_mode=notebook` and retry.")
        raise SystemExit(1) from exc


def region_param(region: ee.Geometry) -> str | list:
    try:
        return json.dumps(ee.Geometry(region).getInfo())
    except Exception:
        return ee.Geometry(region).coordinates().getInfo()


def generate_times(report_day: datetime.date, cutoff_time: dt_time, step_hours: int,
                   count: int, tzinfo: ZoneInfo) -> list[datetime]:
    start = datetime.combine(report_day, cutoff_time, tzinfo=tzinfo)
    return [start + timedelta(hours=step_hours * idx) for idx in range(count)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_utc_label(dt_utc: datetime) -> str:
    return dt_utc.strftime("%Y%m%d_%H00")

def format_kst_label(dt_local: datetime) -> str:
    return dt_local.strftime("%Y%m%d_%H00")

def download_image(url: str, target_path: Path, retries: int, backoff_min: int,
                   logger: logging.Logger) -> bool:
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    attempt = 0
    while attempt <= retries:
        try:
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            ensure_dir(tmp_path.parent)
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            tmp_path.replace(target_path)
            return True
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt > retries:
                logger.warning("Download failed for %s (%s)", target_path.name, exc)
                return False
            sleep_for = backoff_min * (2 ** (attempt - 1))
            logger.warning("Download attempt %s failed for %s (%s); retrying in %ss",
                           attempt, target_path.name, exc, sleep_for)
            time.sleep(sleep_for)
    return False


def run_cams(times_local: Sequence[datetime], region: ee.Geometry, region_param_obj: str | list,
             base_output_dir: Path, settings: Settings, logger: logging.Logger) -> int:
    saved = 0
    emitted_bandlist = False

    ymd_kst = times_local[0].astimezone(ZoneInfo(settings.timezone_name)).strftime("%Y%m%d")
    day_dir = base_output_dir / ymd_kst
    ensure_dir(day_dir)

    for t_local in times_local:
        start_utc = t_local.astimezone(_tz.utc)
        end_utc   = start_utc + timedelta(hours=1)
        start_kst = t_local.astimezone(ZoneInfo(settings.timezone_name))
        kst_label = format_kst_label(start_kst)  
        label_str = f"{kst_label}KST"
        target_path = day_dir / f"gee_cams_{kst_label}_native.tif"

        try:
            ic = (
                ee.ImageCollection(CAMS_DATASET_ID)
                .filterBounds(region)
                .filterDate(start_utc.isoformat(), end_utc.isoformat())
            )
            size = int(ic.size().getInfo())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAMS] %s query failed (%s); skipped", label_str, exc)
            continue
        if size == 0:
            logger.info("[CAMS] empty @ %s (skipped)", label_str)
            continue
        if not emitted_bandlist:
            try:
                sample_band_list = ee.Image(ic.first()).bandNames().getInfo()
                logger.debug("[CAMS] available bands (once): %s", sample_band_list)
            except Exception:
                pass
            emitted_bandlist = True
        try:
            ic_sel = ic.select(CAMS_BANDS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAMS] %s select failed (%s); skipped", label_str, exc)
            continue
        img = ic_sel.mean().rename(CAMS_RENAME).clip(region)
        params = {
            "scale": CAMS_SCALE,
            "region": region_param_obj,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
        try:
            url = img.getDownloadURL(params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAMS] %s URL generation failed (%s); skipped", label_str, exc)
            continue
        logger.info("[CAMS] %s downloading -> %s", label_str, target_path)
        if download_image(url, target_path, settings.retry_max, settings.retry_backoff_min, logger):
            saved += 1
        else:
            logger.warning("[CAMS] %s download failed (skipped)", label_str)
    return saved


def determine_wind_scale(dataset_id: str) -> int:
    return DEFAULT_WIND_SCALE


def run_wind(times_local: Sequence[datetime], region: ee.Geometry, region_param_obj: str | list,
             base_output_dir: Path, settings: Settings, logger: logging.Logger) -> int:
    saved = 0
    scale = determine_wind_scale(settings.wind_dataset_id)
    emitted_bandlist = False
    if not times_local:
        return 0

    ymd_kst = times_local[0].astimezone(ZoneInfo(settings.timezone_name)).strftime("%Y%m%d")
    day_dir = base_output_dir / ymd_kst
    ensure_dir(day_dir)

    first_start_utc = times_local[0].astimezone(_tz.utc)
    last_start_utc  = times_local[-1].astimezone(_tz.utc)
    base_start = first_start_utc - timedelta(days=5)
    last_window_end = last_start_utc + timedelta(hours=1)
    base_end = last_window_end + timedelta(hours=1)

    first_start_utc = times_local[0].astimezone(_tz.utc)
    last_start_utc = times_local[-1].astimezone(_tz.utc)
    base_start = first_start_utc - timedelta(days=5)
    last_window_end = last_start_utc + timedelta(hours=1)
    base_end = last_window_end + timedelta(hours=1)
    try:
        base_ic = (
            ee.ImageCollection(settings.wind_dataset_id)
            .filterBounds(region)
            .filterDate(base_start.isoformat(), base_end.isoformat())
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[WIND] base query failed (%s); all wind exports skipped", exc)
        return 0

    def annotate_valid(img: ee.Image) -> ee.Image:
        run_time = ee.Date(img.get("system:time_start"))
        forecast_hours = ee.Number(ee.Algorithms.If(img.get("forecast_hours"), img.get("forecast_hours"), 0))
        valid_time = run_time.advance(forecast_hours, "hour")
        return img.set("valid_time", valid_time.millis())

    with_valid = base_ic.map(annotate_valid)

    
    for t_local in times_local:
        start_utc = t_local.astimezone(_tz.utc)
        end_utc   = start_utc + timedelta(hours=1)
        start_kst = t_local.astimezone(ZoneInfo(settings.timezone_name))
        kst_label = format_kst_label(start_kst)
        label_str = f"{kst_label}KST"
        target_path = day_dir / f"gee_wind_{kst_label}_native.tif"
        try:
            window = with_valid.filter(
                ee.Filter.And(
                    ee.Filter.gte("valid_time", ee.Date(start_utc.isoformat()).millis()),
                    ee.Filter.lt("valid_time", ee.Date(end_utc.isoformat()).millis()),
                )
            )
            size = int(window.size().getInfo())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WIND] %s filter failed (%s); skipped", label_str, exc)
            continue
        if size == 0:
            logger.info("[WIND] empty @ %s (skipped)", label_str)
            continue
        if not emitted_bandlist:
            try:
                sample_band_list = ee.Image(window.first()).bandNames().getInfo()
                logger.debug("[WIND] available bands (once): %s", sample_band_list)
            except Exception:
                pass
            emitted_bandlist = True
        try:
            ic_sel = window.select(WIND_BANDS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WIND] %s select failed (%s); skipped", label_str, exc)
            continue
        img = ic_sel.mean().rename(WIND_RENAME).clip(region)
        params = {
            "scale": scale,
            "region": region_param_obj,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
        try:
            url = img.getDownloadURL(params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WIND] %s URL generation failed (%s); skipped", label_str, exc)
            continue
        logger.info("[WIND] exporting %s frames=%d -> %s", label_str, size, target_path)
        if download_image(url, target_path, settings.retry_max, settings.retry_backoff_min, logger):
            saved += 1
        else:
            logger.warning("[WIND] %s download failed (skipped)", label_str)
    return saved

def main() -> None:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    tzinfo = ZoneInfo(settings.timezone_name)
    try:
        cutoff_time = dt_time.fromisoformat(settings.cutoff_label)
    except ValueError as exc:
        print(f"[ERROR] REPORT_CUTOFF_KST invalid: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    force = os.getenv("REPORT_FORCE_DATE")
    if force:
        try:
            report_day = datetime.strptime(force, "%Y-%m-%d").date()
        except Exception as exc:
            print(f"[ERROR] REPORT_FORCE_DATE invalid: {exc}", file=sys.stderr)
            raise SystemExit(1)
    else:
        now_local = datetime.now(tzinfo)
        report_day = now_local.date() if now_local.time() < cutoff_time else (now_local + timedelta(days=1)).date()

    log_dir = settings.log_root / f"{report_day:%Y}" / f"{report_day:%m}" / f"{report_day:%d}"
    logger = configure_logger(log_dir, settings.log_level)
    logger.info("REPORT_DAY(KST cutoff day) = %s  cutoff=%s  TZ=%s", report_day, settings.cutoff_label, settings.timezone_name)

    initialize_earth_engine(logger)

    times_local = generate_times(report_day, cutoff_time, settings.time_step_hours,
                                 settings.time_steps_count, tzinfo)
    if len(times_local) != settings.time_steps_count:
        logger.error("Expected %s forecast times but computed %s.", settings.time_steps_count, len(times_local))
        raise SystemExit(1)

    preview = []
    for t_local in times_local:
        preview.append(f"{t_local.strftime('%Y-%m-%d %H:%M')} KST / {t_local.astimezone(_tz.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    logger.info("TIMES (KST / UTC): %s", " | ".join(preview))

    lon_min, lat_min, lon_max, lat_max = settings.region_bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False)
    region_param_obj = region_param(region)

    output_dir = settings.raw_dir / "gee"

    cams_saved = 0
    if settings.enable_cams:
        cams_saved = run_cams(times_local, region, region_param_obj, output_dir, settings, logger)

    wind_saved = run_wind(times_local, region, region_param_obj, output_dir, settings, logger)

    logger.info("CAMS saved: %s/%s, WIND saved: %s/%s",
                cams_saved if settings.enable_cams else 0,
                settings.time_steps_count,
                wind_saved,
                settings.time_steps_count)



if __name__ == "__main__":
    main()
