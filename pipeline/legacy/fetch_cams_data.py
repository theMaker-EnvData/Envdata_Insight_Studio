"""LEGACY — Replaced by GEOS-CF single-source adapter on 2025-10-24."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import List

import earthkit.data as ekd
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv()

DATASET_NAME = "cams-global-atmospheric-composition-forecasts"
VARIABLES = [
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
    "sulphur_dioxide",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

LEADTIME_HOURS = ["9", "12", "15", "18", "21", "24", "27", "30", "33"]

GRID_BBOX = os.getenv("GRID_BBOX", "")
REPORT_CUTOFF_KST = os.getenv("REPORT_CUTOFF_KST", "06:00")
TIMEZONE_NAME = os.getenv("TZ", "Asia/Seoul")

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))


def configure_logger(log_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both console and a dated log file."""

    logger = logging.getLogger("fetch_cams_data")
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fetch_cams.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_area(bbox_str: str) -> List[float]:
    """Convert GRID_BBOX 'W,S,E,N' into CAMS expected [N, W, S, E]."""

    if not bbox_str:
        raise ValueError("GRID_BBOX is required (expected comma separated string).")

    parts = [float(value.strip()) for value in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"GRID_BBOX should contain four comma separated numbers, received: {bbox_str}")

    west, south, east, north = parts
    return [north, west, south, east]


def determine_report_day(now_kst: datetime, cutoff_time: dt_time) -> datetime.date:
    """Return the KST report day based on the 06:00 cut-off."""

    if now_kst.time() < cutoff_time:
        return now_kst.date()
    return (now_kst + timedelta(days=1)).date()


def fetch_cams_data(report_day: datetime.date, logger: logging.Logger) -> Path:
    """
    Fetch CAMS forecast data for the provided report day.

    Args:
        report_day: Date in KST representing the 06:00 report cut-off.
        logger: Logger instance for status output.

    Returns:
        Path to the downloaded GRIB file.
    """

    cycle_date_str = (report_day - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Using CAMS cycle %s 12:00 UTC", cycle_date_str)
    logger.info("Leadtime hours: 9..33 (%s)", ",".join(LEADTIME_HOURS))
    logger.info("Variables: %s", ",".join(VARIABLES))

    area = parse_area(GRID_BBOX)

    yyyymmdd = report_day.strftime("%Y%m%d")
    output_dir = RAW_DIR / "cams" / yyyymmdd
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / f"cams_{yyyymmdd}_0600_grid0p4.grib"

    start_time = time.perf_counter()

    try:
        ds = ekd.from_source(
            "ads",
            DATASET_NAME,
            variable=VARIABLES,
            date=[cycle_date_str],
            time=["12:00"],
            leadtime_hour=LEADTIME_HOURS,
            type=["forecast"],
            format="grib",
            download_format="unarchived",
            area=area,
        )
        ds.to_target("file", str(target_path))
    except Exception:
        if target_path.exists():
            target_path.unlink()
        raise

    duration = time.perf_counter() - start_time
    size_mb = target_path.stat().st_size / (1024 * 1024)
    logger.info("Saved: %s (%.1f MB)", target_path, size_mb)
    logger.info("Fetch duration: %.1f seconds", duration)
    logger.info("This file will be used for the %s KST report.", REPORT_CUTOFF_KST)

    return target_path


def main() -> None:
    tzinfo = ZoneInfo(TIMEZONE_NAME)
    cutoff_time = dt_time.fromisoformat(REPORT_CUTOFF_KST)
    now_kst = datetime.now(tzinfo)

    report_day = determine_report_day(now_kst, cutoff_time)

    log_dir = LOG_DIR / now_kst.strftime("%Y") / now_kst.strftime("%m") / now_kst.strftime("%d")
    logger = configure_logger(log_dir)

    try:
        fetch_cams_data(report_day, logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fetch failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
