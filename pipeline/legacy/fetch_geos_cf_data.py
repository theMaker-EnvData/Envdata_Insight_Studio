"""LEGACY — Replaced by GEE + CAMS adapter on 2025-10-24."""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv()

TARGET_LEAD_HOURS: List[int] = [9, 12, 15, 18, 21, 24, 27, 30, 33]
VAR_NAME_MAP: Dict[str, str] = {
    "PM25_RH35_GCC": "pm2_5",
    "NO2": "no2",
    "O3": "o3",
    "CO": "co",
    "SO2": "so2",
    "U10M": "u10",
    "V10M": "v10",
}
SUMMARY_TEMPLATE = """📡 GEOS-CF Forecast Summary (06 KST)
Cycle UTC: {cycle_date} {cycle_hour}Z
Lead hours: 9–33 (3 h interval)
Vars: PM2.5, NO2, O3, CO, SO2, U10, V10
Subset: within REGION_BBOX (no regridding)
Output: {output_path}
"""


@dataclass
class Settings:
    raw_dir: Path
    log_root: Path
    region_bbox: Tuple[float, float, float, float]
    cutoff_label: str
    timezone_name: str
    gds_url: str
    retry_max: int
    retry_backoff_min: int
    timeout: int

    @classmethod
    def from_env(cls) -> "Settings":
        raw_dir = Path(os.getenv("RAW_DIR", "data/raw"))
        log_root = Path(os.getenv("LOG_DIR", "logs"))
        bbox_str = os.getenv("REGION_BBOX")
        if not bbox_str:
            raise ValueError("REGION_BBOX environment variable is required.")
        try:
            lon_min, lat_min, lon_max, lat_max = [float(part.strip()) for part in bbox_str.split(",")]
        except ValueError as exc:
            raise ValueError("REGION_BBOX must be four comma-separated numbers.") from exc
        region_bbox = (lon_min, lat_min, lon_max, lat_max)

        cutoff_label = os.getenv("REPORT_CUTOFF_KST", "06:00")
        timezone_name = os.getenv("TZ", "Asia/Seoul")
        gds_url = os.getenv(
            "GEOS_CF_GDS_URL",
            "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/chm_inst_3hr_g1440x721_h35",
        )
        retry_max = int(os.getenv("RETRY_MAX", "3"))
        retry_backoff_min = int(os.getenv("RETRY_BACKOFF_MIN", "5"))
        timeout = int(os.getenv("GEOS_CF_TIMEOUT", "120"))

        return cls(
            raw_dir=raw_dir,
            log_root=log_root,
            region_bbox=region_bbox,
            cutoff_label=cutoff_label,
            timezone_name=timezone_name,
            gds_url=gds_url,
            retry_max=retry_max,
            retry_backoff_min=retry_backoff_min,
            timeout=timeout,
        )


def configure_logger(log_dir: Path) -> logging.Logger:
    """Configure module logger with file + console handlers."""

    logger = logging.getLogger("fetch_geos_cf_data")
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fetch_geos_cf.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def ensure_netrc(logger: logging.Logger) -> None:
    """Ensure Earthdata credentials are available for OpenDAP access."""

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.is_file():
        logger.error("Missing ~/.netrc credentials. Please configure your NASA Earthdata login.")
        raise SystemExit(2)


def classify_error(exc: Exception) -> str:
    """Classify OpenDAP access error into coarse categories."""

    message = str(exc).lower()
    if "401" in message or "unauthorized" in message or "forbidden" in message or "403" in message:
        return "auth"
    if "404" in message or "not found" in message:
        return "missing"
    if "timeout" in message or "timed out" in message or "temporary failure" in message:
        return "timeout"
    return "other"


def open_dataset_with_retry(url: str, settings: Settings, logger: logging.Logger) -> xr.Dataset:
    """Attempt to open an OpenDAP dataset with retry/backoff and engine fallback."""

    engines = ("netcdf4", "pydap")
    last_exc: Exception | None = None

    for attempt in range(1, settings.retry_max + 1):
        for engine in engines:
            try:
                return xr.open_dataset(url, engine=engine)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                category = classify_error(exc)
                if category == "auth":
                    logger.error(
                        "Authentication failure when accessing %s with engine '%s': %s",
                        url,
                        engine,
                        exc,
                    )
                    logger.error("Check ~/.netrc and GES DISC authorization.")
                    raise
                if category == "missing":
                    logger.error("Dataset unavailable at %s (missing resource).", url)
                    raise
                continue

        wait_seconds = settings.retry_backoff_min * (2 ** (attempt - 1))
        if attempt < settings.retry_max:
            logger.warning(
                "Attempt %s/%s failed (%s). Retrying in %s s...",
                attempt,
                settings.retry_max,
                last_exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
        else:
            logger.error("Failed to open dataset after %s attempts: %s", settings.retry_max, last_exc)
            if last_exc:
                raise last_exc
            raise RuntimeError("Unable to open dataset.")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown failure when opening dataset.")


def choose_coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    """Select the first matching coordinate name from candidates."""

    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of the coordinate candidates {candidates} were found.")


def determine_lat_slice(lat_values: np.ndarray, lat_min: float, lat_max: float) -> slice:
    """Return latitude slice respecting native orientation."""

    descending = lat_values[0] > lat_values[-1]
    if descending:
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def convert_lon_to_domain(lon: float, use_360: bool) -> float:
    """Convert longitude to target domain (0..360 or -180..180)."""

    if use_360:
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def determine_lon_slices(
    lon_values: np.ndarray,
    lon_min: float,
    lon_max: float,
) -> Tuple[bool, List[Tuple[float, float]]]:
    """Return lon domain flag and slice ranges respecting dateline crossing."""

    lon_min_native = float(lon_values.min())
    lon_max_native = float(lon_values.max())
    use_360 = lon_max_native > 180.0 + 1e-6

    lon_min_adj = convert_lon_to_domain(lon_min, use_360)
    lon_max_adj = convert_lon_to_domain(lon_max, use_360)

    if lon_min_adj <= lon_max_adj:
        return use_360, [(lon_min_adj, lon_max_adj)]

    return use_360, [
        (lon_min_adj, lon_max_native),
        (lon_min_native, lon_max_adj),
    ]


def subset_dataset(
    ds: xr.Dataset,
    time_name: str,
    lat_name: str,
    lon_name: str,
    lat_slice: slice,
    lon_slices: Sequence[Tuple[float, float]],
) -> xr.Dataset:
    """Subset dataset to the provided time/lat/lon slices."""

    subset = ds.sel({lat_name: lat_slice})

    if len(lon_slices) == 1:
        lon_min, lon_max = lon_slices[0]
        subset = subset.sel({lon_name: slice(lon_min, lon_max)})
    else:
        parts = []
        for lon_min, lon_max in lon_slices:
            part = subset.sel({lon_name: slice(lon_min, lon_max)})
            parts.append(part)
        subset = xr.concat(parts, dim=lon_name)
        subset = subset.sortby(lon_name)

    return subset.sortby(time_name).transpose(time_name, lat_name, lon_name)


def build_output_dataset(
    subset: xr.Dataset,
    time_name: str,
    lat_name: str,
    lon_name: str,
    use_360: bool,
    logger: logging.Logger,
) -> xr.Dataset:
    """Construct output dataset with standardized variables."""

    coords = {
        time_name: subset[time_name],
        lat_name: subset[lat_name],
        lon_name: subset[lon_name],
    }

    data_vars = {}
    for source_name, target_name in VAR_NAME_MAP.items():
        if source_name in subset.data_vars:
            data_vars[target_name] = subset[source_name].astype("float32").rename(target_name)
        else:
            logger.warning("Variable %s missing; filling with NaNs.", source_name)
            shape = (
                subset.sizes[time_name],
                subset.sizes[lat_name],
                subset.sizes[lon_name],
            )
            filler = xr.DataArray(
                np.full(shape, np.nan, dtype="float32"),
                coords=coords,
                dims=(time_name, lat_name, lon_name),
                name=target_name,
            )
            data_vars[target_name] = filler

    output_ds = xr.Dataset(data_vars, coords=coords)
    output_ds = output_ds.sortby(time_name).transpose(time_name, lat_name, lon_name)

    lon_domain = "0..360" if use_360 else "-180..180"
    output_ds[lon_name].attrs["geospatial_domain"] = lon_domain

    return output_ds


def write_summary(summary_path: Path, output_path: Path, cycle_dt: datetime) -> None:
    """Write textual summary alongside NetCDF."""

    summary_text = SUMMARY_TEMPLATE.format(
        cycle_date=cycle_dt.strftime("%Y-%m-%d"),
        cycle_hour=cycle_dt.strftime("%H"),
        output_path=str(output_path),
    )
    summary_path.write_text(summary_text, encoding="utf-8")


def main() -> None:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    socket.setdefaulttimeout(settings.timeout)

    tzinfo = ZoneInfo(settings.timezone_name)
    now_kst = datetime.now(tzinfo)
    report_day = now_kst.date()
    cycle_dt_utc = datetime.combine(
        report_day - timedelta(days=1),
        dt_time(12, 0, tzinfo=timezone.utc),
    )

    cutoff_label_compact = settings.cutoff_label.replace(":", "")
    if not cutoff_label_compact:
        cutoff_label_compact = "0600"

    log_dir = settings.log_root / f"{now_kst:%Y}" / f"{now_kst:%m}" / f"{now_kst:%d}"
    logger = configure_logger(log_dir)

    ensure_netrc(logger)

    start_time = time.perf_counter()

    logger.info("Using GEOS-CF GDS cycle %s", cycle_dt_utc.strftime("%Y-%m-%d 12Z"))
    logger.info("Lead hours 9–33 (3 h interval)")
    logger.info("Vars: %s", ",".join(VAR_NAME_MAP.values()))
    logger.info("Subset bbox within REGION_BBOX")

    dataset: xr.Dataset | None = None

    try:
        try:
            dataset = open_dataset_with_retry(settings.gds_url, settings, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unable to open GEOS-CF dataset: %s", exc)
            raise SystemExit(1) from exc

        try:
            time_name = choose_coord_name(dataset, ("time", "Time"))
        except KeyError:
            logger.error("Dataset missing time coordinate.")
            raise SystemExit(1)

        try:
            lat_name = choose_coord_name(dataset, ("lat", "latitude", "Lat", "Latitude"))
            lon_name = choose_coord_name(dataset, ("lon", "longitude", "Lon", "Longitude"))
        except KeyError:
            logger.error("Dataset missing lat/lon coordinates.")
            raise SystemExit(1)

        time_values = dataset[time_name].values
        if time_values.size == 0:
            logger.error("Dataset has no time steps.")
            raise SystemExit(4)

        start_utc = cycle_dt_utc + timedelta(hours=9)
        end_window_utc = cycle_dt_utc + timedelta(hours=36)
        start_np = pd.Timestamp(start_utc).to_datetime64()
        end_window_np = pd.Timestamp(end_window_utc).to_datetime64()

        dataset = dataset.sel({time_name: slice(start_np, end_window_np)})
        if dataset.sizes.get(time_name, 0) == 0:
            logger.error(
                "No time steps between %s and %s UTC found.",
                start_utc.isoformat(),
                (cycle_dt_utc + timedelta(hours=33)).isoformat(),
            )
            raise SystemExit(4)

        max_lead_utc = cycle_dt_utc + timedelta(hours=33)
        max_lead_np = pd.Timestamp(max_lead_utc).to_datetime64()
        dataset = dataset.sel({time_name: slice(start_np, max_lead_np)})
        if dataset.sizes.get(time_name, 0) == 0:
            logger.error(
                "Time subset empty after limiting to %s UTC.",
                max_lead_utc.isoformat(),
            )
            raise SystemExit(4)

        for optional in ("time_bnds", "time_bounds"):
            if optional in dataset.variables:
                dataset = dataset.drop_vars(optional)

        if dataset.dims.get(time_name, 0) != len(TARGET_LEAD_HOURS):
            logger.error(
                "Expected %s lead slices but found %s.",
                len(TARGET_LEAD_HOURS),
                dataset.dims.get(time_name, 0),
            )
            raise SystemExit(4)

        times_utc = pd.to_datetime(dataset[time_name].values)
        logger.info(
            "Selected %s slices from %s to %s UTC",
            dataset.dims[time_name],
            times_utc[0],
            times_utc[-1],
        )

        lat_min, lat_max = settings.region_bbox[1], settings.region_bbox[3]
        lat_slice = determine_lat_slice(dataset[lat_name].values, lat_min, lat_max)

        lon_min, lon_max = settings.region_bbox[0], settings.region_bbox[2]
        use_360, lon_slices = determine_lon_slices(dataset[lon_name].values, lon_min, lon_max)

        subset = subset_dataset(dataset, time_name, lat_name, lon_name, lat_slice, lon_slices)

        if subset.sizes.get(lat_name, 0) == 0 or subset.sizes.get(lon_name, 0) == 0:
            native_lon_span = (
                float(dataset[lon_name].values.min()),
                float(dataset[lon_name].values.max()),
            )
            native_lat_span = (
                float(dataset[lat_name].values.min()),
                float(dataset[lat_name].values.max()),
            )
            logger.error(
                "Spatial subset is empty. Native lon span: %s, native lat span: %s.",
                native_lon_span,
                native_lat_span,
            )
            raise SystemExit(5)

        output_ds = build_output_dataset(subset, time_name, lat_name, lon_name, use_360, logger)

        bbox_attr = (
            f"{settings.region_bbox[0]:.4f},{settings.region_bbox[1]:.4f},"
            f"{settings.region_bbox[2]:.4f},{settings.region_bbox[3]:.4f}"
        )

        output_ds.attrs.update(
            {
                "source": "NASA GMAO GEOS-CF GDS subset",
                "cycle_utc": cycle_dt_utc.strftime("%Y-%m-%dT%H:%MZ"),
                "lead_hours": ",".join(str(val) for val in TARGET_LEAD_HOURS),
                "subset_bbox": bbox_attr,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "grid_native": "0.25x0.3125",
                "opendap_url": settings.gds_url,
            }
        )

        output_dir = settings.raw_dir / "geos_cf" / report_day.strftime("%Y%m%d")
        output_dir.mkdir(parents=True, exist_ok=True)

        netcdf_name = f"geos_cf_{report_day:%Y%m%d}_{cutoff_label_compact}_native.nc"
        netcdf_path = output_dir / netcdf_name
        summary_path = output_dir / f"geos_cf_{report_day:%Y%m%d}_{cutoff_label_compact}_summary.txt"

        encoding = {name: {"dtype": "float32", "_FillValue": np.nan} for name in VAR_NAME_MAP.values()}
        output_ds.to_netcdf(netcdf_path, format="NETCDF4", engine="netcdf4", encoding=encoding)
        write_summary(summary_path, netcdf_path, cycle_dt_utc)

        file_size_mb = netcdf_path.stat().st_size / (1024 * 1024)
        duration = time.perf_counter() - start_time
        logger.info("Saved %s (%.1f MB)", netcdf_path, file_size_mb)
        logger.info("Duration %.1f s", duration)
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == "__main__":
    main()
