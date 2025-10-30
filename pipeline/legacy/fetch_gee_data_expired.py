"""Fetch CAMS and wind data from Google Earth Engine for nine 3-hour slices."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time, timezone as _tz
from pathlib import Path
from typing import Dict, Sequence, Tuple

from zoneinfo import ZoneInfo

import ee
from dotenv import load_dotenv

load_dotenv()

CAMS_BAND_MAP: Dict[str, str] = {
    "pm2_5": "particulate_matter_d_less_than_25_um_surface",
    "pm10": "particulate_matter_d_less_than_10_um_surface",
    "no2": "total_column_nitrogen_dioxide_surface",
    "o3": "gems_total_column_ozone_surface",
    "co": "total_column_carbon_monoxide_surface",
    "so2": "total_column_sulphur_dioxide_surface",
}

DEFAULT_WIND_DATASET = "NOAA/GFS0P25"
WIND_ORIGINAL_BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m"]


@dataclass
class Settings:
    raw_dir: Path
    log_root: Path
    region_bbox: Tuple[float, float, float, float]
    cutoff_label: str
    timezone_name: str
    cams_dataset_id: str
    wind_dataset_id: str
    wind_datasets_raw: str | None
    retry_max: int
    retry_backoff_min: int

    @classmethod
    def from_env(cls) -> Settings:
        raw_dir = Path(os.getenv("RAW_DIR", "/root/EnvData_Insight_Studio/data/raw"))
        log_root = Path(os.getenv("LOG_DIR", "logs"))
        bbox_str = os.getenv("REGION_BBOX")
        if not bbox_str:
            raise ValueError("REGION_BBOX environment variable is required.")
        try:
            lon_min, lat_min, lon_max, lat_max = [float(part.strip()) for part in bbox_str.split(",")]
        except ValueError as exc:
            raise ValueError("REGION_BBOX must contain four comma-separated numbers.") from exc
        cutoff_label = os.getenv("REPORT_CUTOFF_KST", "06:00")
        timezone_name = os.getenv("TZ", "Asia/Seoul")
        cams_dataset_id = os.getenv("GEE_DATASET_ID", "ECMWF/CAMS/NRT")
        wind_dataset_id = os.getenv("GEE_WIND_DATASET", DEFAULT_WIND_DATASET)
        wind_datasets_raw = os.getenv("GEE_WIND_DATASETS")
        retry_max = int(os.getenv("RETRY_MAX", "3"))
        retry_backoff_min = int(os.getenv("RETRY_BACKOFF_MIN", "5"))

        return cls(
            raw_dir=raw_dir,
            log_root=log_root,
            region_bbox=(lon_min, lat_min, lon_max, lat_max),
            cutoff_label=cutoff_label,
            timezone_name=timezone_name,
            cams_dataset_id=cams_dataset_id,
            wind_dataset_id=wind_dataset_id,
            wind_datasets_raw=wind_datasets_raw,
            retry_max=retry_max,
            retry_backoff_min=retry_backoff_min,
        )


def configure_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fetch_gee_data")
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fetch_gee.log"

    logger.setLevel(logging.INFO)
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
        raise SystemExit(3) from exc


def region_param(region: ee.Geometry) -> str | list:
    try:
        return json.dumps(ee.Geometry(region).getInfo())
    except Exception:
        return ee.Geometry(region).coordinates().getInfo()


def generate_times(report_day: datetime.date, cutoff_time: dt_time, tzinfo: ZoneInfo) -> list[datetime]:
    start = datetime.combine(report_day, cutoff_time, tzinfo=tzinfo)
    return [start + timedelta(hours=3 * idx) for idx in range(9)]


def wind_dataset_candidates(settings: Settings) -> list[str]:
    if settings.wind_datasets_raw:
        candidates = [s.strip() for s in settings.wind_datasets_raw.split(",") if s.strip()]
        if candidates:
            return candidates
    candidate = settings.wind_dataset_id.strip()
    if candidate:
        return [candidate]
    return [DEFAULT_WIND_DATASET]


def discover_cams(settings: Settings, region: ee.Geometry, start_iso: str, end_iso: str, logger: logging.Logger):
    sample = (
        ee.ImageCollection(settings.cams_dataset_id)
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
        .first()
    )
    if sample is None:
        raise SystemExit("CAMS sample image not found for band discovery.")

    band_names = sample.bandNames().getInfo()
    available_lower = {name.lower(): name for name in band_names}
    selected = {
        logical: available_lower[band.lower()]
        for logical, band in CAMS_BAND_MAP.items()
        if band.lower() in available_lower
    }
    if not selected:
        logger.error("No CAMS bands matched. Available: %s", sorted(band_names))
        raise SystemExit(4)
    logical_names = list(selected.keys())
    original_names = [selected[key] for key in logical_names]
    proj = sample.select([original_names[0]]).projection()
    scale = int(proj.nominalScale().getInfo())
    logger.info("CAMS bands: %s", ",".join(logical_names))
    logger.info("CAMS nominal scale (native): %s m", scale)
    return original_names, logical_names, proj, scale


def discover_wind(wind_candidates: Sequence[str], region: ee.Geometry, start_iso: str, end_iso: str,
                  logger: logging.Logger) -> dict[str, tuple[ee.Projection, int]]:
    cache: dict[str, tuple[ee.Projection, int]] = {}
    for ds in wind_candidates:
        ic = (
            ee.ImageCollection(ds)
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .select(WIND_ORIGINAL_BANDS)
        )
        size = int(ic.size().getInfo())
        if size == 0:
            logger.info("[WIND] %s no frames in %s..%s", ds, start_iso, end_iso)
            continue
        sample = ic.first()
        if sample is None:
            logger.info("[WIND] %s sample unavailable in %s..%s", ds, start_iso, end_iso)
            continue
        proj = sample.select([WIND_ORIGINAL_BANDS[0]]).projection()
        scale = int(proj.nominalScale().getInfo())
        cache[ds] = (proj, scale)
        logger.info("[WIND] dataset=%s discovery scale=%s m", ds, scale)
    if not cache:
        logger.warning("Unable to locate wind sample image; wind exports will be skipped.")
    return cache


def write_summary_file(summary_path: Path, cams_dataset: str, wind_dataset: str,
                       cams_exports: Sequence[tuple[str, str]], wind_exports: Sequence[tuple[str, str]]) -> None:
    lines = [
        f"CAMS dataset: {cams_dataset}",
        "CAMS exports:",
    ]
    if cams_exports:
        lines.extend(f"  - {label} task={task_id}" for label, task_id in cams_exports)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"WIND dataset: {wind_dataset}")
    lines.append("WIND exports:")
    if wind_exports:
        lines.extend(f"  - {label} task={task_id}" for label, task_id in wind_exports)
    else:
        lines.append("  (none)")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    tzinfo = ZoneInfo(settings.timezone_name)
    cutoff_str = settings.cutoff_label
    try:
        cutoff_time = dt_time.fromisoformat(cutoff_str)
    except ValueError as exc:
        print(f"[ERROR] REPORT_CUTOFF_KST invalid: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    now_local = datetime.now(tzinfo)
    report_day = now_local.date() if now_local.time() < cutoff_time else (now_local + timedelta(days=1)).date()

    log_dir = settings.log_root / f"{now_local:%Y}" / f"{now_local:%m}" / f"{now_local:%d}"
    logger = configure_logger(log_dir)

    logger.info("Using GEE CAMS dataset %s", settings.cams_dataset_id)

    wind_candidates = wind_dataset_candidates(settings)
    logger.info("[WIND] candidates: %s", wind_candidates)

    start_perf = time.perf_counter()

    initialize_earth_engine(logger)

    lon_min, lat_min, lon_max, lat_max = settings.region_bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False)
    region_param_str = region_param(region)

    times_local = generate_times(report_day, cutoff_time, tzinfo)
    if len(times_local) != 9:
        logger.error("Expected 9 forecast times but computed %s.", len(times_local))
        raise SystemExit(5)

    output_dir = settings.raw_dir / "gee" / f"{report_day:%Y%m%d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    global_start_utc = times_local[0].astimezone(_tz.utc)
    global_end_utc = times_local[-1].astimezone(_tz.utc) + timedelta(hours=1)
    wind_discovery_end_utc = times_local[-1].astimezone(_tz.utc) + timedelta(hours=2)

    cams_original, cams_logical, cams_proj, cams_scale = discover_cams(
        settings,
        region,
        global_start_utc.isoformat(),
        global_end_utc.isoformat(),
        logger,
    )

    wind_proj_cache = discover_wind(
        wind_candidates,
        region,
        global_start_utc.isoformat(),
        wind_discovery_end_utc.isoformat(),
        logger,
    )

    cams_exports: list[tuple[str, str]] = []
    wind_exports: list[tuple[str, str]] = []

    for idx, t_local in enumerate(times_local):
        start_utc = t_local.astimezone(_tz.utc)
        end_utc = start_utc + timedelta(hours=1)
        utc_label = start_utc.strftime("%Y%m%d_%H00")

        cams_slice = (
            ee.ImageCollection(settings.cams_dataset_id)
            .filterBounds(region)
            .filterDate(start_utc.isoformat(), end_utc.isoformat())
            .select(cams_original)
        )
        if int(cams_slice.size().getInfo()) == 0:
            logger.warning("No CAMS data for %s UTC; skipping export.", utc_label)
        else:
            cams_image = (
                cams_slice.mean()
                .rename(cams_logical)
                .clip(region)
                .setDefaultProjection(cams_proj)
            )
            task = ee.batch.Export.image.toDrive(
                image=cams_image,
                description=f"CAMS_{utc_label}",
                fileNamePrefix=f"gee_cams_{utc_label}_native",
                region=region_param_str,
                scale=cams_scale,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            task.start()
            logger.info("[CAMS] exporting %s (task=%s)", utc_label, task.id)
            cams_exports.append((utc_label, task.id))

        if wind_candidates:
            wind_window_start = start_utc.isoformat()
            wind_window_end = (start_utc + timedelta(hours=2)).isoformat()
            wind_exported = False
            for ds in wind_candidates:
                ic = (
                    ee.ImageCollection(ds)
                    .filterBounds(region)
                    .filterDate(wind_window_start, wind_window_end)
                    .select(WIND_ORIGINAL_BANDS)
                )
                if int(ic.size().getInfo()) == 0:
                    logger.info("[WIND] %s no frames in %s..%s", ds, wind_window_start, wind_window_end)
                    continue
                proj_scale = wind_proj_cache.get(ds)
                if proj_scale is None:
                    sample = ic.first()
                    if sample is None:
                        logger.info("[WIND] %s sample unavailable in %s..%s", ds, wind_window_start, wind_window_end)
                        continue
                    proj = sample.select([WIND_ORIGINAL_BANDS[0]]).projection()
                    scale = int(proj.nominalScale().getInfo())
                    wind_proj_cache[ds] = (proj, scale)
                else:
                    proj, scale = proj_scale
                wind_image = (
                    ic.mean()
                    .rename(["u10", "v10"])
                    .clip(region)
                    .setDefaultProjection(proj)
                )
                task = ee.batch.Export.image.toDrive(
                    image=wind_image,
                    description=f"WIND_{utc_label}",
                    fileNamePrefix=f"gee_wind_{utc_label}_native",
                    region=region_param_str,
                    scale=scale,
                    maxPixels=1e13,
                    fileFormat="GeoTIFF",
                )
                task.start()
                logger.info("[WIND] exporting %s via %s (task=%s)", utc_label, ds, task.id)
                logger.info("[WIND] picked dataset=%s for %s", ds, utc_label)
                wind_exports.append((utc_label, task.id))
                wind_exported = True
                break
            if not wind_exported:
                logger.warning("[WIND] all candidates empty for %s (skipped)", utc_label)

    cutoff_label_compact = cutoff_str.replace(":", "") or "0600"
    summary_path = output_dir / f"gee_cams_{report_day:%Y%m%d}_{cutoff_label_compact}_summary.txt"
    write_summary_file(summary_path, settings.cams_dataset_id, ", ".join(wind_candidates), cams_exports, wind_exports)
    logger.info("Summary: %s", summary_path)

    duration = time.perf_counter() - start_perf
    logger.info("Duration %.1f s", duration)


if __name__ == "__main__":
    main()
