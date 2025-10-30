# Project Bundle (EnvData_Insight_Studio)

```json
{
  "project_name": "EnvData_Insight_Studio",
  "declared_base_dir": "/opt/EnvData_Insight_Studio",
  "detected_root": "/opt/EnvData_Insight_Studio",
  "timestamp_local": "2025-10-30 20:19:47",
  "python": {
    "version": "3.12.3",
    "executable": "/opt/EnvData_Insight_Studio/.venv/bin/python"
  },
  "git": null,
  "config": {
    "include_ext": [
      ".bat",
      ".conf",
      ".css",
      ".dockerfile",
      ".env.example",
      ".env.template",
      ".html",
      ".ini",
      ".js",
      ".json",
      ".jsx",
      ".md",
      ".ps1",
      ".py",
      ".sh",
      ".sql",
      ".toml",
      ".ts",
      ".tsx",
      ".yaml",
      ".yml"
    ],
    "exclude_ext": [
      ".7z",
      ".avi",
      ".bmp",
      ".bz2",
      ".db",
      ".dll",
      ".dylib",
      ".feather",
      ".flac",
      ".gz",
      ".ico",
      ".jpeg",
      ".jpg",
      ".mkv",
      ".mov",
      ".mp3",
      ".mp4",
      ".parquet",
      ".pdf",
      ".pickle",
      ".pkl",
      ".png",
      ".pyc",
      ".pyd",
      ".pyo",
      ".so",
      ".sqlite",
      ".sqlite3",
      ".tar",
      ".tgz",
      ".tif",
      ".tiff",
      ".wav",
      ".webp",
      ".xz",
      ".zip"
    ],
    "exclude_dirs": [
      ".DS_Store",
      ".cache",
      ".env",
      ".git",
      ".idea",
      ".mypy_cache",
      ".parcel-cache",
      ".playwright",
      ".pytest_cache",
      ".ruff_cache",
      ".sass-cache",
      ".venv",
      ".vscode",
      "__pycache__",
      "build",
      "coverage",
      "dist",
      "env",
      "git",
      "node_modules",
      "site-packages",
      "venv"
    ],
    "max_file_bytes": 200000,
    ".bundleignore": []
  }
}
```


---

## README.md

```md

```


---

## app/__init__.py

```python

```


---

## app/main.py

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os, datetime as dt

BASE = Path(__file__).resolve().parent.parent
PUBLIC = BASE / "public"
TILES = PUBLIC / "tiles"
REPORTS = PUBLIC / "reports"

app = FastAPI(title="EnvData_Insight_Studio")
app.mount("/static", StaticFiles(directory=str(BASE / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "app" / "templates"))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # 최신 날짜 폴더/파일 탐색
    dates = sorted([p.name for p in TILES.glob("*") if p.is_dir()], reverse=True)
    latest = dates[0] if dates else ""
    return templates.TemplateResponse("index.html", {"request": request, "latest": latest, "dates": dates})

@app.get("/report/{yyyymmdd}", response_class=HTMLResponse)
def report_view(yyyymmdd: str, request: Request):
    # HTML 렌더 (타일/요약 JSON을 템플릿에서 읽어 표시)
    return templates.TemplateResponse("report.html", {"request": request, "date": yyyymmdd})

@app.get("/download/pdf/{yyyymmdd}")
def download_pdf(yyyymmdd: str):
    pdf = REPORTS / f"report_{yyyymmdd}.pdf"
    if pdf.exists():
        return FileResponse(str(pdf), media_type="application/pdf", filename=pdf.name)
    return {"error": "PDF not found"}

```


---

## app/templates/index.html

```
<!doctype html><html><body>
<h2>EnvData Insight Studio</h2>
<p>Latest: {{ latest }}</p>
<ul>
  {% for d in dates %}
    <li><a href="/report/{{ d|replace('-','') }}">{{ d }}</a></li>
  {% endfor %}
</ul>
</body></html>

```


---

## app/templates/report.html

```
<!doctype html><html><body>
<h2>Daily AQ Report — {{ date }}</h2>
<p><a href="/download/pdf/{{ date }}">Download PDF</a></p>
<div id="tiles">[tiles go here]</div>
</body></html>

```


---

## bundle_code.py

```python
#!/usr/bin/env python3
# bundle_copy.py — Zero-config project bundler (run in the project folder)
# - CWD = declared_base_dir (auto)
# - detected_root = git root if exists else CWD
# - Supports .bundleignore (glob patterns)
# - Safe excludes: .venv/venv/node_modules, caches, binaries, images, >120KB

import os, sys, time, pathlib, mimetypes, subprocess, argparse, json, fnmatch

SAFE_EXCLUDE_DIRS = {
    ".git","git","venv",".venv","env",".env",
    "node_modules","__pycache__","dist","build",
    ".pytest_cache",".mypy_cache",".ruff_cache",".cache",
    ".playwright",".idea",".vscode","coverage","site-packages",
    ".DS_Store",".sass-cache",".parcel-cache"
}
SAFE_EXCLUDE_EXT = {
    ".pyc",".pyo",".pyd",".so",".dll",".dylib",
    ".zip",".tar",".tgz",".gz",".xz",".bz2",".7z",
    ".pdf",".png",".jpg",".jpeg",".webp",".tif",".tiff",".bmp",".ico",
    ".db",".sqlite",".sqlite3",".parquet",".feather",".pickle",".pkl",
    ".mp4",".mov",".mkv",".avi",".mp3",".wav",".flac"
}
SAFE_INCLUDE_EXT = {
    ".py",".js",".jsx",".ts",".tsx",
    ".json",".yml",".yaml",".toml",".ini",".conf",
    ".md",".sql",".sh",".bat",".ps1",".dockerfile",
    ".html",".css",".env.example",".env.template"
}
LANG_MAP = {
    ".py":"python",".js":"javascript",".jsx":"jsx",".ts":"ts",".tsx":"tsx",
    ".sql":"sql",".sh":"bash",".yml":"yaml",".yaml":"yaml",".toml":"toml",
    ".ini":"ini",".md":"md",".json":"json",".ps1":"powershell",
}

def lang_for(path: pathlib.Path):
    if path.name.lower()=="dockerfile": return "dockerfile"
    return LANG_MAP.get(path.suffix.lower(), "")

def looks_binary(p: pathlib.Path) -> bool:
    typ, _ = mimetypes.guess_type(str(p))
    if typ and (typ.startswith("text/") or "json" in typ or "xml" in typ):
        try:
            with p.open("rb") as f:
                chunk = f.read(4096)
            return b"\x00" in chunk
        except:
            return True
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except:
        return True

def safe_run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=cwd).decode().strip()
    except Exception:
        return ""

def detect_git_root(cwd: pathlib.Path) -> pathlib.Path:
    root = safe_run(["git","rev-parse","--show-toplevel"], cwd=str(cwd))
    return pathlib.Path(root).resolve() if root else cwd

def load_bundleignore(root: pathlib.Path):
    """Return list of glob patterns from .bundleignore if present."""
    p = root / ".bundleignore"
    patterns = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            patterns.append(s)
    return patterns

def matches_any(path_rel: str, patterns):
    return any(fnmatch.fnmatch(path_rel, pat) for pat in patterns)

def gather_files(root: pathlib.Path, include_ext, exclude_ext, exclude_dirs, max_bytes: int, bundleignore_patterns):
    files = []
    for r, dirs, names in os.walk(root):
        rp = pathlib.Path(r)
        # prune excluded dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for n in names:
            p = rp / n
            rel = p.relative_to(root).as_posix()
            # .bundleignore patterns
            if matches_any(rel, bundleignore_patterns):
                continue
            # directory components exclude
            if any(part in exclude_dirs for part in p.parts):
                continue
            # ext filters
            ext = p.suffix.lower()
            if ext in exclude_ext: 
                continue
            if p.name.lower()!="dockerfile" and include_ext and ext not in include_ext:
                continue
            # size + binary
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except:
                continue
            if looks_binary(p):
                continue
            files.append(p)
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser(description="Bundle project into a single Markdown snapshot (zero-config).")
    ap.add_argument("--max-bytes", type=int, default=int(os.environ.get("BUNDLE_MAX_FILE_BYTES","200000")),
                    help="Per-file size limit (bytes). Default 200000.")
    ap.add_argument("--outfile", default="", help="Output .md name. Default: <folder>_bundle_YYYYMMDD_HHMM.md")
    ap.add_argument("--no-git", action="store_true", help="Skip git info lookup.")
    # 고급 사용자용(필요할 때만)
    ap.add_argument("--include-ext", nargs="*", default=None, help="Override include extensions")
    ap.add_argument("--exclude-ext", nargs="*", default=None, help="Override exclude extensions")
    ap.add_argument("--exclude-dir", nargs="*", default=None, help="Override exclude directory names")
    args = ap.parse_args()

    # CWD가 곧 'declared_base_dir'
    declared_base = pathlib.Path.cwd().resolve()
    # git root 있으면 거기로, 없으면 CWD
    detected_root = detect_git_root(declared_base)

    project_name = detected_root.name
    ts = time.strftime("%Y%m%d_%H%M")
    out = pathlib.Path(args.outfile or f"{project_name}_bundle_{ts}.md").resolve()

    include_ext = set(args.include_ext) if args.include_ext is not None else set(SAFE_INCLUDE_EXT)
    exclude_ext = set(args.exclude_ext) if args.exclude_ext is not None else set(SAFE_EXCLUDE_EXT)
    exclude_dirs = set(args.exclude_dir) if args.exclude_dir is not None else set(SAFE_EXCLUDE_DIRS)
    max_bytes = int(args.max_bytes)

    git_info = {}
    if not args.no_git and (detected_root / ".git").exists():
        git_info = {
            "branch": safe_run(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(detected_root)),
            "commit": safe_run(["git","rev-parse","--short","HEAD"], cwd=str(detected_root)),
            "root":   str(detected_root),
            "status": safe_run(["git","status","--porcelain"], cwd=str(detected_root)),
        }

    bundleignore = load_bundleignore(detected_root)

    files = gather_files(detected_root, include_ext, exclude_ext, exclude_dirs, max_bytes, bundleignore)

    meta = {
        "project_name": project_name,
        "declared_base_dir": str(declared_base),
        "detected_root": str(detected_root),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "git": git_info or None,
        "config": {
            "include_ext": sorted(include_ext),
            "exclude_ext": sorted(exclude_ext),
            "exclude_dirs": sorted(exclude_dirs),
            "max_file_bytes": max_bytes,
            ".bundleignore": bundleignore
        }
    }

    total_bytes = 0
    with out.open("w", encoding="utf-8", errors="ignore") as f:
        f.write(f"# Project Bundle ({meta['project_name']})\n\n")
        f.write("```json\n" + json.dumps(meta, indent=2, ensure_ascii=False) + "\n```\n")
        for p in files:
            rel = p.relative_to(detected_root)
            fence = lang_for(p) or ""
            f.write(f"\n\n---\n\n## {rel}\n\n```{fence}\n")
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                total_bytes += len(txt.encode("utf-8", errors="ignore"))
                f.write(txt)
            except Exception as e:
                f.write(f"<<READ_ERROR: {e}>>")
            f.write("\n```\n")

    print(f"[OK] Wrote: {out}")
    print(f"Files: {len(files)}, approx text bytes: {total_bytes:,}")

if __name__ == "__main__":
    main()

```


---

## config/adapters.yaml

```yaml
# config/adapters.yaml
version: 2
description: >
  CAMS-only pipeline using ecmwf-datastores-client. Server-side subsetting for
  variables/time/area; download as a single GRIB (unarchived). Output normalized
  to the common schema used by build_json.

defaults:
  timezone_raw: "UTC"           # raw time zone from source
  out_units:
    concentration: "µg/m3"
    wind_speed: "m/s"
    wind_dir: "deg"

schema:  # unified output schema for downstream steps (JSON, R plots)
  fields:
    - lat
    - lon
    - ts            # ISO8601 in UTC
    - pm2_5
    - pm10
    - no2
    - o3
    - co
    - so2
    - wind_u
    - wind_v
    - wind_speed
    - wind_dir

sources:
  cams:
    enabled: true
    client: "ecmwf-datastores-client"

    # ---- Request contract passed to the client (Codex consumes these keys) ----
    request:
      # ADS dataset id (forecasts). If you later prefer reanalysis, switch here.
      dataset: "cams-global-atmospheric-composition-forecasts"

      # Variables to retrieve (minimal set). Names follow ADS variable ids.
      variables:
        # mixing ratio (kg/kg): converted to µg/m3 downstream
        - pm2p5_mmr
        - pm10_mmr
        - no2_mmr
        - o3_mmr
        - co_mmr
        - so2_mmr
        # winds at 10 m (m/s)
        - u10
        - v10

      # Spatial subset — bbox as "W,S,E,N" (client will normalize if needed)
      area_bbox_env: "${GRID_BBOX}"     # e.g., 116.4,28.2,140.0,45.6

      # Temporal subset — computed by runner from REPORT_CUTOFF_KST=05:00,
      # TIME_STEP_HOURS=3, TIME_STEPS_COUNT=9, TZ=Asia/Seoul, then converted to UTC.
      time_policy:
        reference_kst_env: "${REPORT_CUTOFF_KST}"
        step_hours_env: "${TIME_STEP_HOURS}"
        steps_count_env: "${TIME_STEPS_COUNT}"
        tz_env: "${TZ}"

      # File/transport format from server
      data_format: "grib"               # prefer GRIB; open via cfgrib/xarray
      download_format: "unarchived"     # single file, not zipped
      filename_pattern: "cams_{YYYYMMDD}_0500_grid0p4.grib"

      # Performance/robustness knobs (optional)
      retry:
        max_env: "${RETRY_MAX}"
        backoff_min_env: "${RETRY_BACKOFF_MIN}"

    # ---- Field mapping & unit conversion (applied after download/decode) ----
    mapping:
      lat: "lat"
      lon: "lon"
      ts: "time"   # keep UTC

      # mixing ratio (kg/kg) -> µg/m3. Converter name resolved by transformer.
      pm2_5:
        from: "pm2p5_mmr"
        convert: "mmr_to_ugm3"
        # PM is mixture; use density-based factor at STP or runtime profile
        molar_mass: 1.0
      pm10:
        from: "pm10_mmr"
        convert: "mmr_to_ugm3"
        molar_mass: 1.0
      no2:
        from: "no2_mmr"
        convert: "mmr_to_ugm3"
        molar_mass: 46.0055
      o3:
        from: "o3_mmr"
        convert: "mmr_to_ugm3"
        molar_mass: 47.9982
      co:
        from: "co_mmr"
        convert: "mmr_to_ugm3"
        molar_mass: 28.0101
      so2:
        from: "so2_mmr"
        convert: "mmr_to_ugm3"
        molar_mass: 64.066

      wind_u:
        from: "u10"
        units: "m/s"
      wind_v:
        from: "v10"
        units: "m/s"
      wind_speed:
        compute: "sqrt(u10^2 + v10^2)"
        units: "m/s"
      wind_dir:
        # meteorological direction: 0°=North, clockwise
        compute: "deg_met(atan2(u10, v10))"
        units: "deg"

    # ---- Quality control rules ----
    qc:
      rules:
        - name: bbox_filter
          kind: "drop_outside_bbox"
          params:
            bbox_env: "${GRID_BBOX}"
        - name: non_negative_pm
          kind: "range"
          fields: [pm2_5, pm10]
          min: 0
          max: 1000
        - name: gas_range
          kind: "range"
          fields: [no2, o3, co, so2]
          min: 0
          max: 5000
        - name: wind_reasonable
          kind: "range"
          fields: [wind_speed]
          min: 0
          max: 60
      on_fail: "drop_and_log"   # drop record, write reason to logs

  # KMA adapter kept as a stub for optional future fallback/merge (disabled)
  kma:
    enabled: false
    notes: "Use only if ENABLE_KMA=true; acts as fallback for winds or met vars."
    request:
      service: "VilageFcst"
      variables: [UUU, VVV]
    mapping:
      wind_u: "UUU"
      wind_v: "VVV"
    merge:
      strategy: "cams_primary_kma_fallback"

```


---

## config/cities.yaml

```yaml
KR:
  - { name: "Seoul",     lat: 37.5665, lon: 126.9780 }
  - { name: "Daejeon",   lat: 36.3504, lon: 127.3845 }
  - { name: "Gwangju",   lat: 35.1595, lon: 126.8526 }
  - { name: "Busan",     lat: 35.1796, lon: 129.0756 }
  - { name: "Daegu",     lat: 35.8714, lon: 128.6014 }
CN:
  - { name: "Shanghai",  lat: 31.2304, lon: 121.4737 }
  - { name: "Qingdao",   lat: 36.0671, lon: 120.3826 }
  - { name: "Tianjin",   lat: 39.3434, lon: 117.3616 }
JP:
  - { name: "Fukuoka",   lat: 33.5904, lon: 130.4017 }
  - { name: "Osaka",     lat: 34.6937, lon: 135.5023 }
  - { name: "Hiroshima", lat: 34.3853, lon: 132.4553 }

```


---

## pipeline/adapters/fetch_gee_data.py

```python
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

```


---

## pipeline/legacy/build_json.py

```python
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

```


---

## pipeline/legacy/fetch_aqicn.py

```python
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

```


---

## pipeline/legacy/fetch_cams_data.py

```python
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

```


---

## pipeline/legacy/fetch_gee_data_expired.py

```python
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

```


---

## pipeline/legacy/fetch_geos_cf_data.py

```python
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

```


---

## pipeline/legacy/fetch_openmeteo.py

```python
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

```


---

## pipeline/legacy/run_all.sh

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export $(grep -v '^#' config/.env | xargs)

TS_LABEL=${1:-"06:00"}        # 표기용 기준시각
ATTEMPT=${2:-"1"}             # 재시도 번호

echo "[Run] $(date +'%F %T %Z') attempt=${ATTEMPT} ts=${TS_LABEL}"

python3 pipeline/fetch_aqicn.py --ts "${TS_LABEL}"
python3 pipeline/fetch_openmeteo.py --ts "${TS_LABEL}"
python3 pipeline/build_json.py --ts "${TS_LABEL}"

# R로 현재/타일 이미지 생성
Rscript r/current_map_pm25.R  --ts "${TS_LABEL}"
Rscript r/make_tiles_pm25.R   --ts "${TS_LABEL}"

# PDF 생성 (wkhtmltopdf 사용 예시; 보고서 HTML을 먼저 렌더해 두는 전제)
YYYYMMDD=$(date +'%Y%m%d')
wkhtmltopdf "http://127.0.0.1:8000/report/${YYYYMMDD}" "public/reports/report_${YYYYMMDD}.pdf"

echo "[Done] ${YYYYMMDD}"

```


---

## pipeline/render_report_html.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_report_html.py  (FINAL v6)
- metrics JSON : public/metrics_<YYYYMMDD>_KST.json
- montage PNG  : public/tiles/<YYYYMMDD>/pm25_wind_<YYYYMMDD>_KST_montage.png
- output HTML  : public/reports/report_<YYYYMMDD>_0600.html

업데이트:
  • 몽타주: 테두리 제거, 본문 폭 100% (양쪽 맞춤)
  • 표: 전칸 가운데 정렬, '시간' 열 볼드
  • 불릿(개조체): “국내: … / 동아시아: …” 같은 라벨 금지
      - 첫 줄: 한반도(국내 5도시) 핵심 분석 1줄
      - 둘째 줄: 동아시아 전체(중국+일본 7도시) 핵심 분석 1줄
"""

import os, json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# .env 로드 (프로젝트 루트의 .env)
try:
    from dotenv import load_dotenv
    import pathlib, os as _os
    root = pathlib.Path(__file__).resolve().parents[1]  # repo 루트(/opt/EnvData_Insight_Studio)
    load_dotenv(root / ".env")
    print(f"[ENV] ENABLE_AI_SECTIONS={_os.getenv('ENABLE_AI_SECTIONS')} "
          f"OPENAI_MODEL={_os.getenv('OPENAI_MODEL')} "
          f"OPENAI_API_KEY={'set' if bool(_os.getenv('OPENAI_API_KEY')) else 'missing'}")
except Exception as _e:
    print(f"[ENV] dotenv load skipped ({_e})")

DOMESTIC = ["서울", "대전", "광주", "부산", "대구"]
CHINA    = ["상하이", "칭다오", "톈진"]
JAPAN    = ["후쿠오카", "오사카", "히로시마"]
GROUPS   = [("국내", DOMESTIC), ("중국", CHINA), ("일본", JAPAN)]

TITLE = "EnvData Insight Studio – Daily Report"
DATA_SOURCE = "ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)"

METRICS_DIR = "public"
TILES_DIR   = "public/tiles"
REPORTS_DIR = "public/reports"

# 경로: FS(실제 파일) / HREF(HTML src) 분리
PUBLIC_DIR = "public"
PUBLIC_TILES_FS = os.path.join(PUBLIC_DIR, "tiles")   # 존재 확인용
TILES_HREF_PREFIX = "../tiles"                        # HTML img src용

def montage_fs_for(ymd_):   # 실제 파일
    return os.path.join(PUBLIC_TILES_FS, ymd_, f"pm25_wind_{ymd_}_KST_montage.png")

def montage_href_for(ymd_): # HTML src
    return f"{TILES_HREF_PREFIX}/{ymd_}/pm25_wind_{ymd_}_KST_montage.png"


# ---------------- util ----------------
# --- AI: sections generation ---
def build_llm_input(date_str, recs):
    """입력 스키마(JSON) 최소 버전 생성 – 시계열만 포함(격자 파생값은 추후 추가 가능)."""
    # 도시 순서/그룹
    cities_order = [c for _, cs in GROUPS for c in cs]
    series = {c: [] for c in cities_order}
    for r in recs:
        if r["city"] in series:
            series[r["city"]].append({"time": r["timestamp_kst"], "pm25": round(float(r["pm25"]), 1)})

    # 피크(국내/동아시아)
    def pick_peak(cities):
        best = (-1, None, None)
        for r in recs:
            if r["city"] in cities:
                v = float(r["pm25"])
                if v > best[0]:
                    best = (v, r["timestamp_kst"], r["city"])
        return {"city": best[2], "time": best[1][-5:], "pm25": int(round(best[0]))}

    payload = {
        "meta": {
            "target_date": date_str,
            "cutoff_hour": 6,
            "timezone": "Asia/Seoul",
            "data_source": "ECMWF/CAMS(PM2.5), NOAA/GFS(10m wind)"
        },
        "cities": {
            "order": cities_order,
            "groups": {"domestic": DOMESTIC, "china": CHINA, "japan": JAPAN}
        },
        "thresholds": {"pm25_standard": 35},
        "timeseries": [{"city": c, "unit": "μg/m³", "values": series[c]} for c in cities_order],
        "derived": {
            "regional_extremes": {
                "domestic_peak":  pick_peak(DOMESTIC),
                "east_asia_peak": pick_peak(CHINA + JAPAN),
            }
        }
    }
    return payload

def call_llm_sections(input_json):
    """
    ENABLE_AI_SECTIONS=true 에서만 호출.
    실패 시 None 반환(렌더러는 기본문 사용).
    """
    import os, json, re
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] skipped: no OPENAI_API_KEY")
        return None

    # -------- 프롬프트 (부드러운 경어체, 굵게 의무, telegraphic bullets) --------
    system = (
        "You are a Korean environmental forecaster writing a public daily air-quality OUTLOOK (예보). "
        "Write in Korean. Use only <p> and <b> HTML tags. Tone: clear, calm, professional, for general public. "
        "HARD RULES:\n"
        "1) Tense: future only (…되겠습니다/…보이겠습니다/…넘기겠습니다). No past forms, no typos.\n"
        "2) Each of summary and impact is EXACTLY ONE paragraph (<p>…</p>) with 2–3 sentences. "
        "   • summary = daily FLOW (when/where/how strong). Include the 35 μg/m³ standard at least once and put "
        "     ONE key clause in <b>bold</b> (max 1–2 bold tags total).\n"
        "   • impact = CAUSE/MECHANISM (transport, W→E wind component, mixing, topography). Do not repeat sentences/phrases from summary. "
        "     Include ONE concrete anchor (a time, value, or named region) and put ONE key clause in <b>bold</b> (max 1–2 bold tags total).\n"
        "   Numbers per section should be concise (1–2 uses max).\n"
        "3) health = ONE paragraph with 2–4 sentences. Actionable advice (mask for sensitive groups, timing of ventilation/air-purifier, "
        "   check index before activity). Vary sentence endings; do not repeat the same ending twice in a row. Do NOT use awkward future-plan style "
        "   like '…할 예정입니다'. Keep it polite and practical.\n"
        "4) bullets = EXACTLY TWO items in TELEGRAPHIC STYLE (English: telegraphic style; Korean: 개조체). "
        "   That means concise clause-like sentences with a subject and verb, no honorific endings, no advice verbs, no trailing period. "
        "   Examples: '수도권 밤사이 기준 초과 가능성 높음', '산둥 고농도 동진, 서풍 성분 강화로 중부권 단기 영향 예상'. "
        "   Do NOT use labels like '서울 - …', no hyphen lists, no polite endings. "
        "   End each bullet as a telegraphic clause like '… 가능성 높음', '… 영향 예상', '… 일시 상승'. "
        "   No polite endings. No period.\n"

        "5) Use normal spacing; keep a single space after sentence-ending punctuation and after closing tags when followed by text. "
        "   Keep wording natural and not overly stiff.\n"
    )

    user = (
        "Return ONLY the following JSON schema using the input data (city time series + peaks). "
        "SCHEMA:"
        "{"
        "\"meta\":{\"target_date\":string,\"locale\":\"ko-KR\",\"pm25_standard\":number},"
        "\"sections\":{"
        "\"summary\":\"<p>2–3 sentences, daily flow; include 35 μg/m³ once; 1–2 <b>…</b> max</p>\","
        "\"impact\":\"<p>2–3 sentences, cause/mechanism; non-duplicative vs summary; 1–2 <b>…</b> max</p>\","
        "\"health\":{\"html\":\"<p>5–7 sentences; polite future tense; actionable; varied endings; no '…예정입니다'</p>\"},"
        "\"bullets\":[\"telegraphic style #1 (개조체, no honorific)\",\"telegraphic style #2 (개조체, no honorific)\"]"
        "}"
        "}\n"
        "QUALITY GATES:\n"
        "- summary/impact each: exactly one <p>; future tense; include at least one concrete anchor; have 1 bold clause.\n"
        "- bullets: two telegraphic sentences, no polite endings, no periods, no advice verbs.\n"
        "- health: 5–7 sentences, polite future tense, practical, varied endings.\n"
        "INPUT JSON FOLLOWS:\n"
    )

    # -------- SDK 호출 (chat.completions + JSON 모드) --------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
            ],
        )
        rid = getattr(resp, "id", "n/a")
        print(f"[LLM] ok id={rid}")
        text = resp.choices[0].message.content
        data = json.loads(text)
    except Exception as e:
        print(f"[LLM] error: {repr(e)}")
        return None

    # -------- 후처리 (최소화: bullets만 살짝 정리) --------
    def _fix_bullets(arr):
        """불릿을 '개조체(telegraphic)' 한 줄로 다듬되, 의미 손상 없이 최소 보정."""
        if not isinstance(arr, list):
            return arr
        endings = ("입니다.", "입니다", "합니다.", "합니다", "합니다요", "합니다요.",
                "겠습니다.", "겠습니다", "되겠습니다.", "되겠습니다", "됩니다.", "됩니다",
                "예상됩니다.", "예상됩니다", "예정입니다.", "예정입니다", "요.", "요")
        out = []
        for x in arr[:2]:
            if not isinstance(x, str):
                out.append("")
                continue
            t = x.strip()

            # HTML 태그 제거
            t = re.sub(r"<.*?>", "", t)

            # 라벨/하이픈 패턴만 정리(내용 단어는 건드리지 않음)
            t = t.replace(" - ", " ").replace("—", " ").replace("–", " ").replace(" -", " ").replace("- ", " ")

            # 끝 마침표/느낌표만 제거
            t = t.rstrip(" .!…")

            # 공손어/미래계획체 등 '정확히 일치하는' 끝 어미만 제거(어간은 절대 건드리지 않음)
            for suf in endings:
                if t.endswith(suf):
                    t = t[: -len(suf)].rstrip()
                    break

            out.append(t)
        # 정확히 2개 유지
        return (out + ["", ""])[:2]


    try:
        sec = data.get("sections", {})
        if "bullets" in sec:
            sec["bullets"] = _fix_bullets(sec["bullets"])
    except Exception as e:
        print(f"[LLM] post-fix warn: {repr(e)}")

    return data


    try:
        sec = data.get("sections", {})
        if "summary" in sec: sec["summary"] = _ensure_single_p(sec["summary"])
        if "impact"  in sec: sec["impact"]  = _ensure_single_p(sec["impact"])
        if isinstance(sec.get("health"), dict) and "html" in sec["health"]:
            sec["health"]["html"] = _ensure_single_p(sec["health"]["html"])
        if "bullets" in sec:
            sec["bullets"] = _fix_bullets(sec["bullets"])
    except Exception as e:
        print(f"[LLM] post-fix error: {repr(e)}")

    return data


def now_kst():
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST)

def decide_target_date():
    force = os.getenv("REPORT_FORCE_DATE")
    if force:
        return datetime.strptime(force, "%Y-%m-%d").date()
    t = now_kst()
    return t.date() if t.hour < 6 else (t + timedelta(days=1)).date()

def ymd(d): return d.strftime("%Y%m%d")

def metric_path_for(ymd_):     return os.path.join(METRICS_DIR, f"metrics_{ymd_}_KST.json")
def montage_path_for(ymd_):    return os.path.join(TILES_DIR, ymd_, f"pm25_wind_{ymd_}_KST_montage.png")
def timeseries_path_for(ymd_): return os.path.join(REPORTS_DIR, f"timeseries_{ymd_}_0600.png")


def load_metrics_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_records(arr):
    recs=[]
    for r in arr:
        city=r.get("city"); ts=r.get("timestamp_kst"); v=r.get("pm25")
        if city and ts and v is not None:
            recs.append({"city":city,"timestamp_kst":ts,"pm25":float(v)})
    recs.sort(key=lambda x: datetime.strptime(x["timestamp_kst"],"%Y-%m-%d %H:%M"))
    return recs

def collect_matrix(recs):
    uniq_ts=sorted({r["timestamp_kst"] for r in recs},
                   key=lambda s: datetime.strptime(s,"%Y-%m-%d %H:%M"))
    ordered=[c for _,cs in GROUPS for c in cs]
    by_ts_city=defaultdict(dict)
    for r in recs:
        by_ts_city[r["timestamp_kst"]][r["city"]]=r["pm25"]
    grid=[]; allv=[]
    for ts in uniq_ts:
        row={"time":ts[-5:]}  # HH:MM
        for c in ordered:
            v=by_ts_city.get(ts,{}).get(c)
            if v is None: row[c]=None
            else:
                iv=int(round(v)); row[c]=iv; allv.append(iv)
        grid.append(row)
    return grid, ordered, allv

def top3_threshold(vals):
    if not vals: return None
    s=sorted(vals, reverse=True)
    return s[min(3,len(s))-1]

def html_escape(s):
    return (s.replace("&","&amp;").replace("<","&lt;")
             .replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;"))

def update_manifest(latest_ymd: str, path=os.path.join(REPORTS_DIR, "manifest.json")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"latest": latest_ymd, "dates": [latest_ymd]}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f)
            dates = list(dict.fromkeys([latest_ymd] + (cur.get("dates") or [])))  # 중복 제거, 최신 맨 앞
            data = {"latest": latest_ymd, "dates": dates}
    except Exception as e:
        print(f"[manifest] read/merge skipped: {e}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest updated -> {path}")

# ---------------- HTML blocks ----------------
def build_table_html(grid, ordered, thr_top):
    head1=['<th class="stub time-col">시간</th>'] + [f"<th>{html_escape(c)}</th>" for c in ordered]
    group_cells=[f'<th class="group" colspan="{len(cs)}">{html_escape(lbl)}</th>' for lbl,cs in GROUPS]

    body=[]
    for r in grid:
        tds=[f'<td class="time-col"><b>{r["time"]}</b></td>']  # 시간열 볼드
        for c in ordered:
            v=r[c]
            if v is None:
                tds.append('<td class="na">—</td>')
                continue
            cls=[]
            if v>=35: cls.append("thresh")                   # 기준 이상 배경
            if thr_top and v>=thr_top: cls.append("top3")    # 상위 3개 진빨강+볼드
            cattr=f' class="{" ".join(cls)}"' if cls else ""
            tds.append(f"<td{cattr}>{v}</td>")
        body.append("<tr>"+"".join(tds)+"</tr>")

    return f"""
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr>{"".join(head1)}</tr>
      <tr class="group-row"><th></th>{"".join(group_cells)}</tr>
    </thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""

def pick_bullets_domestic_and_ea(grid):
    """개조체 2줄 생성 — 라벨 없이 간결 문장."""
    def hh_label(hhmm): return f"{int(hhmm.split(':')[0])}시"
    # 국내 최대값
    dom_best = None  # (v, time, city)
    for r in grid:
        for c in DOMESTIC:
            v=r.get(c)
            if isinstance(v,int):
                if (dom_best is None) or (v > dom_best[0]):
                    dom_best=(v, r["time"], c)
    # 동아시아(중국+일본) 최대값
    ea_best = None
    for r in grid:
        for c in CHINA+JAPAN:
            v=r.get(c)
            if isinstance(v,int):
                if (ea_best is None) or (v > ea_best[0]):
                    ea_best=(v, r["time"], c)
    # 문장(개조체, 라벨 없음)
    li1 = f"<li>{dom_best[2]} {hh_label(dom_best[1])} 약 <b>{dom_best[0]}</b> — 기준 대비 주의 구간</li>" if dom_best else ""
    li2 = f"<li>{ea_best[2]} {hh_label(ea_best[1])} 약 <b>{ea_best[0]}</b> — 동아시아 최고치</li>" if ea_best else ""
    return f"<ul class='bullets'>{li1}{li2}</ul>"


def build_page_html(date_str, table_html, montage_path=None, bullets_html="", timeseries_img=None,
                    summary_html=None, impact_html=None, health_html=None):
    date_kr=f"{int(date_str[0:4])}년 {int(date_str[5:7])}월 {int(date_str[8:10])}일"
    summary=(f"<p>{date_kr} 낮 동안 국내 대기질은 대체로 낮은 수준을 유지하겠으며, "
              "<b>밤(21시 전후) 수도권 중심으로 기준(35 μg/m³) 이상으로 상승</b>할 가능성이 큽니다. "
              "자정 무렵 정점 형성 후 익일 아침에는 기준 부근으로 내려올 전망입니다. "
              "국외(화북·발해만 일대)의 높은 농도가 밤~새벽에 <b>중부권에 단기 영향을 줄 수 있습니다.</b></p>")
    impact=("<p>낮에는 혼합·확산이 원활해 대체로 낮은 수준이겠습니다. "
            "다만 국외 고농도대가 서해 쪽에 머물 가능성이 있어, 해당 시간대에 수도권·충청에서 기준 이상 구간이 짧게 나타날 수 있습니다. "
            "이 상승은 <b>단기적</b>일 가능성이 높고, 새벽 이후 혼합 회복과 함께 빠르게 완화되겠습니다.</p>")
    health=("<p>오늘 밤부터 내일 아침 사이 수도권·충청은 기준 이상 구간이 일시적으로 나타날 수 있어 "
            "<b>민감군은 KF 마스크</b> 착용을 권합니다. 실내 환기는 짧고 자주가 좋습니다. "
            "영남·호남은 대체로 야외활동에 무리가 없으나, 장시간 외부 활동 전 <b>최신 지수</b> 확인을 권장합니다.</p>")
    if summary_html: summary = summary_html
    if impact_html:  impact  = impact_html
    if health_html:  health  = health_html
    montage = f'<figure class="montage"><img src="{montage_path}" alt="PM2.5 3×3 몽타주"></figure>' if montage_path else ""
    chart_block = (f'<figure class="chart"><img src="{os.path.basename(timeseries_img)}" '
                   f'alt="PM2.5 시계열 그래프"></figure>') if (timeseries_img and os.path.exists(timeseries_img)) else ""
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<style>
  :root {{
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }}
  body{{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }}
  header h1{{ margin:0; font-size:22px; font-weight:700 }}
  header .meta{{ margin:4px 0 24px; color:#555; font-size:13px }}
  section h2{{ font-size:18px; margin:24px 0 8px }}
  .bullets{{ margin:8px 0 12px 18px }}
  .bullets li{{ margin:2px 0 }}
  .chart img{{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }}
  .montage img{{ width:100%; display:block; border:none; margin:8px 0 12px }}
  .table-wrap{{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }}
  .unit-label{{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }}
  table.city-table{{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }}
  .city-table thead th, .city-table tbody td{{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }}
  .city-table thead th:last-child, .city-table tbody td:last-child{{ border-right:0 }}
  .city-table thead th{{ background:var(--soft); color:#222; font-weight:600 }}
  .city-table thead .group-row th.group{{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }}
  .city-table td.thresh{{ background:var(--warn-bg) }}
  .city-table td.top3{{ color:var(--top3); font-weight:700 }}
  footer{{ margin-top:28px; color:#666; font-size:12px; text-align:center }}
</style>
</head>
<body>
<header>
  <h1>{TITLE} ({date_kr})</h1>
  <div class="meta">데이터 출처: {DATA_SOURCE}</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    {summary}
    {montage}
  </section>

  <section id="impact">
    {bullets_html}
    <h2>2) 영향 해석</h2>
    {impact}
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    {chart_block}
    {table_html}
  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    {health}
  </section>
</main>

<footer>© {datetime.now().year} EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>"""

# ---------------- main ----------------
def main():
    target = decide_target_date()

    y = ymd(target)

    metrics_path   = metric_path_for(y)
    montage_fs     = montage_fs_for(y)     # 존재 확인
    montage_src    = montage_href_for(y)   # HTML src
    timeseries_png = timeseries_path_for(y)

    arr = load_metrics_json(metrics_path)
    recs = normalize_records(arr)
    grid, ordered, allv = collect_matrix(recs)
    thr = top3_threshold(allv)

    # (A) 기본 규칙문 준비
    table_html   = build_table_html(grid, ordered, thr)
    bullets_html = pick_bullets_domestic_and_ea(grid)
    summary_html = None
    impact_html  = None
    health_html  = None

    # (B) AI 사용 토글
    if os.getenv("ENABLE_AI_SECTIONS", "false").lower() in ("1","true","yes"):
        llm_in = build_llm_input(target.strftime("%Y-%m-%d"), recs)
        llm_out = call_llm_sections(llm_in)
        if llm_out and "sections" in llm_out:
            s = llm_out["sections"]
            # 필수 필드만 안전 반영(없으면 기존 규칙문 유지)
            summary_html = s.get("summary")
            impact_html  = s.get("impact")
            health_pack  = s.get("health") or {}
            health_html  = health_pack.get("html")
            # bullets 2개면 교체
            b = s.get("bullets") or []
            if isinstance(b, list) and len(b) == 2:
                bullets_html = "<ul class='bullets'><li>"+b[0]+"</li><li>"+b[1]+"</li></ul>"

    # (C) 최종 HTML 렌더(비어있으면 기본문 사용)
    page_html = build_page_html(
        target.strftime("%Y-%m-%d"),
        table_html,
        montage_path=montage_src if os.path.exists(montage_fs) else None,
        bullets_html=bullets_html,
        timeseries_img=timeseries_png,
        summary_html=summary_html,
        impact_html=impact_html,
        health_html=health_html
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, f"report_{y}_0600.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"[OK] {out}")

    update_manifest(y)

if __name__=="__main__":
    main()

```


---

## public/metrics_20251028_KST.json

```json
[
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 19.3208,
    "pm10": 20.3807,
    "u": 0.8097,
    "v": -1.1819,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 15.2187,
    "pm10": 17.0816,
    "u": 0.2026,
    "v": 0.0739,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 8.793,
    "pm10": 10.5888,
    "u": -0.4054,
    "v": -1.94,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 7.6093,
    "pm10": 8.5962,
    "u": 3.5887,
    "v": -4.5214,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 12.0937,
    "pm10": 13.4833,
    "u": 1.1097,
    "v": -1.0438,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 37.1855,
    "pm10": 39.7925,
    "u": -2.7459,
    "v": -2.4952,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 14.606,
    "pm10": 23.7024,
    "u": 0.8533,
    "v": 2.6064,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 52.1217,
    "pm10": 57.2422,
    "u": -0.4682,
    "v": 0.1377,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.743,
    "pm10": 9.8016,
    "u": -0.2362,
    "v": -2.9116,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 9.3704,
    "pm10": 11.447,
    "u": 1.7677,
    "v": -0.9628,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 8.8114,
    "pm10": 11.0375,
    "u": 1.7622,
    "v": -2.8316,
    "timestamp_kst": "2025-10-28 06:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 26.1826,
    "pm10": 28.1488,
    "u": 0.7785,
    "v": -1.4059,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 15.8124,
    "pm10": 17.115,
    "u": 0.0199,
    "v": -0.8102,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 10.218,
    "pm10": 11.8238,
    "u": -0.651,
    "v": -2.8493,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 8.5455,
    "pm10": 9.6081,
    "u": 1.4092,
    "v": -2.9504,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 12.0527,
    "pm10": 13.337,
    "u": 1.2295,
    "v": -1.3804,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 55.5276,
    "pm10": 58.0812,
    "u": -3.9454,
    "v": -2.3936,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.2281,
    "pm10": 24.4608,
    "u": 0.7447,
    "v": 3.435,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 91.6311,
    "pm10": 96.3872,
    "u": -0.8479,
    "v": -0.1772,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.7061,
    "pm10": 10.0168,
    "u": 0.0696,
    "v": -4.0761,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.7373,
    "pm10": 10.3036,
    "u": 2.675,
    "v": -3.3127,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 9.0779,
    "pm10": 12.2734,
    "u": 0.8386,
    "v": -4.1644,
    "timestamp_kst": "2025-10-28 09:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 11.9318,
    "pm10": 15.3867,
    "u": 2.0077,
    "v": -2.066,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 8.4133,
    "pm10": 10.8053,
    "u": 0.7055,
    "v": -2.4586,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 6.6771,
    "pm10": 8.1472,
    "u": -0.1886,
    "v": -3.5442,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 6.4333,
    "pm10": 7.185,
    "u": 0.7295,
    "v": -1.9679,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 6.1867,
    "pm10": 7.1976,
    "u": 1.7198,
    "v": -2.3939,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 21.8345,
    "pm10": 26.3343,
    "u": -4.2636,
    "v": -2.1588,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.8632,
    "pm10": 21.3336,
    "u": 1.0915,
    "v": 4.5986,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 68.8851,
    "pm10": 71.2797,
    "u": -0.4082,
    "v": 1.4654,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.165,
    "pm10": 8.1807,
    "u": 0.4027,
    "v": -3.8578,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.9727,
    "pm10": 10.7225,
    "u": 3.0849,
    "v": -4.0869,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.2635,
    "pm10": 10.0253,
    "u": -0.0903,
    "v": -3.8078,
    "timestamp_kst": "2025-10-28 12:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 12.0722,
    "pm10": 18.859,
    "u": 2.9835,
    "v": -1.9962,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 8.7592,
    "pm10": 15.7825,
    "u": 1.1435,
    "v": -2.475,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 6.0214,
    "pm10": 9.521,
    "u": 0.197,
    "v": -3.8447,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 5.854,
    "pm10": 7.1792,
    "u": -0.4704,
    "v": -0.8126,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 5.4876,
    "pm10": 9.7615,
    "u": 1.8248,
    "v": -3.0649,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 22.1679,
    "pm10": 27.4818,
    "u": -4.3559,
    "v": -2.0349,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.6781,
    "pm10": 20.9617,
    "u": 1.5931,
    "v": 5.016,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 59.6678,
    "pm10": 61.6506,
    "u": 0.1296,
    "v": 2.5971,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.8573,
    "pm10": 9.2277,
    "u": -0.3224,
    "v": -4.1818,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.0545,
    "pm10": 9.241,
    "u": 1.7099,
    "v": -5.5632,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.6254,
    "pm10": 9.0445,
    "u": -0.4127,
    "v": -5.2341,
    "timestamp_kst": "2025-10-28 15:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 23.5266,
    "pm10": 29.4898,
    "u": 2.0388,
    "v": -1.1948,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 17.7606,
    "pm10": 24.0667,
    "u": 0.5805,
    "v": -0.7928,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 11.1817,
    "pm10": 15.2364,
    "u": 0.1392,
    "v": -1.7751,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 11.3164,
    "pm10": 13.9493,
    "u": -0.2087,
    "v": -2.0589,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 14.2425,
    "pm10": 19.4258,
    "u": 0.488,
    "v": -1.2585,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 31.7574,
    "pm10": 36.0801,
    "u": -3.8626,
    "v": -1.1184,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.5906,
    "pm10": 20.3358,
    "u": 0.8963,
    "v": 5.2905,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 69.5723,
    "pm10": 71.3199,
    "u": -0.2518,
    "v": 2.5737,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.7443,
    "pm10": 9.7191,
    "u": -0.4095,
    "v": -2.8207,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 9.653,
    "pm10": 11.2077,
    "u": 0.2934,
    "v": -4.1656,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.7491,
    "pm10": 8.5104,
    "u": -0.5794,
    "v": -5.747,
    "timestamp_kst": "2025-10-28 18:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 44.1811,
    "pm10": 50.8238,
    "u": 0.1176,
    "v": -0.1825,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 27.1266,
    "pm10": 34.197,
    "u": -0.2606,
    "v": 0.3891,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 14.8535,
    "pm10": 19.9541,
    "u": -0.8674,
    "v": -1.1289,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 14.9574,
    "pm10": 19.4394,
    "u": -0.5843,
    "v": -2.8683,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 20.364,
    "pm10": 26.1375,
    "u": 0.0801,
    "v": -0.5126,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 48.2224,
    "pm10": 51.8103,
    "u": -3.1651,
    "v": -0.4747,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.1108,
    "pm10": 18.8174,
    "u": 0.2549,
    "v": 6.4149,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 77.1871,
    "pm10": 79.1898,
    "u": -0.0987,
    "v": 1.9729,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.329,
    "pm10": 10.1885,
    "u": -0.8445,
    "v": -1.5874,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.8083,
    "pm10": 13.6685,
    "u": 0.2846,
    "v": -1.8428,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.7919,
    "pm10": 9.6504,
    "u": -1.3763,
    "v": -4.5517,
    "timestamp_kst": "2025-10-28 21:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 50.4308,
    "pm10": 56.8128,
    "u": -0.621,
    "v": 0.1996,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 29.4242,
    "pm10": 36.9877,
    "u": -0.53,
    "v": 0.3833,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 17.5944,
    "pm10": 23.5975,
    "u": -1.0987,
    "v": -0.821,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 17.0719,
    "pm10": 22.0532,
    "u": -0.9714,
    "v": -4.0014,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 23.1785,
    "pm10": 28.9225,
    "u": -0.0737,
    "v": -0.7024,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 44.2383,
    "pm10": 47.361,
    "u": -2.6571,
    "v": -0.2313,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 10.3101,
    "pm10": 16.4571,
    "u": 0.8009,
    "v": 6.9666,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 63.9821,
    "pm10": 66.0838,
    "u": 0.2391,
    "v": 1.9853,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.5762,
    "pm10": 10.4899,
    "u": -0.6684,
    "v": -1.4025,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 16.3222,
    "pm10": 18.713,
    "u": 0.0526,
    "v": -0.8406,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 8.3925,
    "pm10": 10.4732,
    "u": -1.5487,
    "v": -4.1456,
    "timestamp_kst": "2025-10-29 00:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 40.8822,
    "pm10": 46.2125,
    "u": -0.9731,
    "v": 0.1707,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 27.1395,
    "pm10": 34.2034,
    "u": -0.7381,
    "v": 0.2837,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 19.4006,
    "pm10": 24.2043,
    "u": -1.2583,
    "v": -0.757,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 15.9199,
    "pm10": 20.5775,
    "u": -1.0944,
    "v": -4.6036,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 23.0963,
    "pm10": 28.5336,
    "u": -0.2647,
    "v": -0.9039,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 33.1274,
    "pm10": 35.801,
    "u": -2.6989,
    "v": -0.3802,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 9.3997,
    "pm10": 14.9373,
    "u": 1.0693,
    "v": 6.6679,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 51.2807,
    "pm10": 53.1603,
    "u": 1.1892,
    "v": 2.6789,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.738,
    "pm10": 8.5751,
    "u": -0.6461,
    "v": -1.5872,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 13.4451,
    "pm10": 16.3578,
    "u": -0.3426,
    "v": -0.1051,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.2696,
    "pm10": 9.5529,
    "u": -1.6756,
    "v": -4.2731,
    "timestamp_kst": "2025-10-29 03:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 32.3808,
    "pm10": 37.2931,
    "u": -1.2767,
    "v": 0.1702,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 26.3508,
    "pm10": 32.7219,
    "u": -0.9161,
    "v": 0.3801,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 21.0827,
    "pm10": 24.479,
    "u": -1.4022,
    "v": -0.4308,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 15.1041,
    "pm10": 19.3868,
    "u": -1.2703,
    "v": -4.7961,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 23.7013,
    "pm10": 28.8689,
    "u": -0.3012,
    "v": -0.9703,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 31.0015,
    "pm10": 33.5262,
    "u": -2.7812,
    "v": -0.2896,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 8.8166,
    "pm10": 13.603,
    "u": 0.7325,
    "v": 6.7588,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 45.4234,
    "pm10": 46.878,
    "u": 2.0368,
    "v": 3.3519,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 5.9938,
    "pm10": 7.9893,
    "u": -0.7359,
    "v": -1.5475,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 13.8287,
    "pm10": 17.2026,
    "u": -0.8737,
    "v": 0.3511,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.7679,
    "pm10": 9.5809,
    "u": -1.8098,
    "v": -4.4477,
    "timestamp_kst": "2025-10-29 06:00"
  }
]

```


---

## public/metrics_20251029_KST.json

```json
[
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 33.1995,
    "pm10": 38.5714,
    "u": -1.3596,
    "v": 0.2731,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 28.5459,
    "pm10": 36.2292,
    "u": -0.9125,
    "v": 0.3292,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 20.6083,
    "pm10": 25.1252,
    "u": -1.4257,
    "v": -0.4132,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 15.1119,
    "pm10": 19.6994,
    "u": -1.4646,
    "v": -5.0131,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 23.6177,
    "pm10": 29.6032,
    "u": -0.3529,
    "v": -0.9927,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 30.9959,
    "pm10": 33.7126,
    "u": -2.6602,
    "v": -0.224,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 9.4502,
    "pm10": 14.5745,
    "u": 0.6724,
    "v": 6.5766,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 50.2237,
    "pm10": 51.5132,
    "u": 1.9454,
    "v": 3.6827,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.2145,
    "pm10": 8.0277,
    "u": -0.7399,
    "v": -1.6157,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 13.5414,
    "pm10": 16.7943,
    "u": -0.9271,
    "v": 0.4065,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.75,
    "pm10": 9.1237,
    "u": -1.6726,
    "v": -4.3912,
    "timestamp_kst": "2025-10-29 06:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 39.8623,
    "pm10": 44.1156,
    "u": -1.5071,
    "v": 0.2697,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 26.7904,
    "pm10": 31.0016,
    "u": -1.1368,
    "v": 0.0185,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 20.5623,
    "pm10": 22.7445,
    "u": -1.1584,
    "v": 0.0424,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 16.3386,
    "pm10": 20.4562,
    "u": -1.8338,
    "v": -5.0566,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 25.0121,
    "pm10": 29.9618,
    "u": -0.2921,
    "v": -1.1101,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 47.9127,
    "pm10": 50.3631,
    "u": -3.7368,
    "v": 0.2215,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 10.5079,
    "pm10": 15.6322,
    "u": 0.1497,
    "v": 7.3195,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 57.1712,
    "pm10": 58.3536,
    "u": 2.6344,
    "v": 4.2738,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.0188,
    "pm10": 9.2809,
    "u": -1.5649,
    "v": -2.231,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.322,
    "pm10": 14.7911,
    "u": -1.2201,
    "v": -1.7174,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.54,
    "pm10": 9.0181,
    "u": -2.3949,
    "v": -4.0302,
    "timestamp_kst": "2025-10-29 09:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 19.6254,
    "pm10": 25.5669,
    "u": -1.1111,
    "v": 0.8129,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 7.7442,
    "pm10": 10.2338,
    "u": -1.652,
    "v": -0.4094,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 6.3653,
    "pm10": 8.8275,
    "u": -0.7125,
    "v": 0.8584,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 10.8041,
    "pm10": 15.8419,
    "u": -2.8377,
    "v": -4.6693,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 10.126,
    "pm10": 13.9524,
    "u": -0.5741,
    "v": -1.1264,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 17.9174,
    "pm10": 21.9292,
    "u": -4.639,
    "v": 1.118,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 10.3078,
    "pm10": 16.3406,
    "u": 0.2075,
    "v": 7.8883,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 48.9363,
    "pm10": 49.9153,
    "u": 4.734,
    "v": 4.6618,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 5.6192,
    "pm10": 7.759,
    "u": -2.2499,
    "v": -2.4297,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.8321,
    "pm10": 9.9491,
    "u": -0.7095,
    "v": -3.265,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 5.1441,
    "pm10": 6.981,
    "u": -4.0516,
    "v": -1.6072,
    "timestamp_kst": "2025-10-29 12:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 15.6018,
    "pm10": 19.8868,
    "u": -0.0766,
    "v": 0.8685,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 7.0273,
    "pm10": 9.3097,
    "u": -1.7134,
    "v": -0.5753,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 5.7335,
    "pm10": 7.7824,
    "u": -0.525,
    "v": 0.8846,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 9.9569,
    "pm10": 14.8494,
    "u": -4.2699,
    "v": -3.2879,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 8.0966,
    "pm10": 11.2221,
    "u": -0.5019,
    "v": -0.8552,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 22.3182,
    "pm10": 26.8775,
    "u": -4.772,
    "v": 1.7677,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.0603,
    "pm10": 18.9893,
    "u": 0.3892,
    "v": 7.8831,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 44.5544,
    "pm10": 45.5657,
    "u": 4.9714,
    "v": 4.2977,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 5.4639,
    "pm10": 7.4826,
    "u": -2.7208,
    "v": -2.4299,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 6.814,
    "pm10": 8.5289,
    "u": -0.4921,
    "v": -3.5214,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 4.982,
    "pm10": 6.6937,
    "u": -4.1961,
    "v": -0.2467,
    "timestamp_kst": "2025-10-29 15:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 44.1054,
    "pm10": 47.581,
    "u": -0.5353,
    "v": 0.5077,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 21.4412,
    "pm10": 23.5793,
    "u": -1.167,
    "v": -0.3272,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 16.6772,
    "pm10": 18.6685,
    "u": -0.8506,
    "v": 1.3978,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 12.2355,
    "pm10": 16.4971,
    "u": -4.3688,
    "v": -2.9751,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 18.7997,
    "pm10": 21.9021,
    "u": -0.5448,
    "v": -0.4839,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 35.0773,
    "pm10": 39.1235,
    "u": -3.613,
    "v": 1.7554,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 14.3902,
    "pm10": 21.8868,
    "u": -0.2241,
    "v": 7.8705,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 75.7674,
    "pm10": 76.6469,
    "u": 2.6576,
    "v": 2.4692,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.0876,
    "pm10": 7.3874,
    "u": -0.9548,
    "v": -1.6792,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 8.4158,
    "pm10": 9.6348,
    "u": -0.4589,
    "v": -2.9059,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 5.9655,
    "pm10": 7.326,
    "u": -3.9231,
    "v": -0.8198,
    "timestamp_kst": "2025-10-29 18:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 68.7559,
    "pm10": 72.3809,
    "u": -0.8257,
    "v": 0.236,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 27.6073,
    "pm10": 29.8143,
    "u": -1.1544,
    "v": 0.3845,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 23.6426,
    "pm10": 26.0379,
    "u": -1.4885,
    "v": 0.4592,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 13.7108,
    "pm10": 17.8314,
    "u": -3.7907,
    "v": -3.1532,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 25.3889,
    "pm10": 29.4971,
    "u": -0.7078,
    "v": -0.1158,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 50.6368,
    "pm10": 54.2963,
    "u": -3.578,
    "v": 2.0733,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 14.1527,
    "pm10": 20.9189,
    "u": -0.2641,
    "v": 7.7861,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 142.3436,
    "pm10": 144.4767,
    "u": 1.686,
    "v": 1.7165,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.3381,
    "pm10": 8.4611,
    "u": -1.0666,
    "v": -0.9379,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.9718,
    "pm10": 13.2154,
    "u": -1.2931,
    "v": -0.6406,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.3764,
    "pm10": 7.7986,
    "u": -2.5202,
    "v": -3.6427,
    "timestamp_kst": "2025-10-29 21:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 53.0017,
    "pm10": 55.6534,
    "u": -0.9507,
    "v": 0.2157,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 24.8734,
    "pm10": 27.0072,
    "u": -0.8865,
    "v": 0.5894,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 22.1115,
    "pm10": 24.5579,
    "u": -1.2915,
    "v": -0.0257,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 13.7271,
    "pm10": 17.6017,
    "u": -2.5015,
    "v": -3.4646,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 29.0175,
    "pm10": 33.7403,
    "u": -0.3432,
    "v": -0.1223,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 44.0525,
    "pm10": 47.3992,
    "u": -3.058,
    "v": 2.0563,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 13.2986,
    "pm10": 19.504,
    "u": 0.402,
    "v": 7.5367,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 131.8397,
    "pm10": 133.3431,
    "u": 1.3998,
    "v": 0.3778,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 7.3874,
    "pm10": 8.5253,
    "u": -0.8998,
    "v": -0.4266,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 14.9832,
    "pm10": 16.183,
    "u": -1.3685,
    "v": -0.0639,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.9601,
    "pm10": 8.7942,
    "u": -2.4704,
    "v": -3.5052,
    "timestamp_kst": "2025-10-30 00:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 36.3251,
    "pm10": 38.4347,
    "u": -0.9012,
    "v": 0.4727,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 22.9553,
    "pm10": 25.3081,
    "u": -0.7574,
    "v": 0.5929,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 18.7618,
    "pm10": 21.7179,
    "u": -1.2077,
    "v": -0.3289,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 12.3574,
    "pm10": 15.9741,
    "u": -0.6759,
    "v": -3.9395,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 27.5371,
    "pm10": 32.8006,
    "u": 0.0241,
    "v": -0.2427,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 37.7951,
    "pm10": 40.9577,
    "u": -2.5714,
    "v": 1.3143,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 12.2993,
    "pm10": 17.8555,
    "u": 0.5248,
    "v": 5.9004,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 112.6054,
    "pm10": 113.872,
    "u": 0.1648,
    "v": -1.1047,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.4027,
    "pm10": 7.6383,
    "u": -0.7029,
    "v": -0.0404,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 12.0322,
    "pm10": 12.9993,
    "u": -1.5035,
    "v": -0.332,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.4353,
    "pm10": 9.5933,
    "u": -1.6907,
    "v": -3.0834,
    "timestamp_kst": "2025-10-30 03:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 30.7626,
    "pm10": 32.7593,
    "u": -1.0713,
    "v": 0.5559,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 23.2061,
    "pm10": 25.9995,
    "u": -0.6418,
    "v": 0.579,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 17.7726,
    "pm10": 21.04,
    "u": -1.2612,
    "v": -0.3613,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 12.4939,
    "pm10": 16.1204,
    "u": -0.3129,
    "v": -3.5708,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 25.353,
    "pm10": 31.0355,
    "u": 0.0366,
    "v": -0.395,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 36.0566,
    "pm10": 38.9289,
    "u": -2.8,
    "v": 0.6621,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 11.7136,
    "pm10": 16.9039,
    "u": 0.5552,
    "v": 4.4309,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 94.4889,
    "pm10": 104.3963,
    "u": -1.3425,
    "v": -1.4038,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.7788,
    "pm10": 8.1699,
    "u": -0.7715,
    "v": -0.1238,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.7235,
    "pm10": 12.502,
    "u": -1.3871,
    "v": -0.363,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 8.6837,
    "pm10": 10.9447,
    "u": -1.5639,
    "v": -2.5197,
    "timestamp_kst": "2025-10-30 06:00"
  }
]

```


---

## public/metrics_20251030_KST.json

```json
[
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 30.7626,
    "pm10": 32.7593,
    "u": -1.0463,
    "v": 0.5324,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 23.2061,
    "pm10": 25.9995,
    "u": -0.65,
    "v": 0.6199,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 17.7726,
    "pm10": 21.04,
    "u": -1.2876,
    "v": -0.3921,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 12.4939,
    "pm10": 16.1204,
    "u": -0.2246,
    "v": -3.6599,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 25.353,
    "pm10": 31.0355,
    "u": 0.0645,
    "v": -0.3594,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 36.0566,
    "pm10": 38.9289,
    "u": -2.8416,
    "v": 0.6544,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 11.7136,
    "pm10": 16.9039,
    "u": 0.6431,
    "v": 4.5816,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 94.4889,
    "pm10": 104.3963,
    "u": -1.1361,
    "v": -1.2554,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.7788,
    "pm10": 8.1699,
    "u": -0.6823,
    "v": -0.1833,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.7235,
    "pm10": 12.502,
    "u": -1.3603,
    "v": -0.3526,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 8.6837,
    "pm10": 10.9447,
    "u": -1.5008,
    "v": -2.5837,
    "timestamp_kst": "2025-10-30 06:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 39.884,
    "pm10": 41.2503,
    "u": -1.0048,
    "v": 0.8294,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 28.0509,
    "pm10": 29.9577,
    "u": -0.6654,
    "v": 0.257,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 16.7445,
    "pm10": 18.9962,
    "u": -0.9804,
    "v": -0.3443,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 14.8444,
    "pm10": 18.3119,
    "u": -0.297,
    "v": -3.382,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 24.1243,
    "pm10": 28.4979,
    "u": 0.2471,
    "v": -0.4748,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 54.3628,
    "pm10": 56.6556,
    "u": -3.5017,
    "v": 0.7467,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.1941,
    "pm10": 21.0085,
    "u": -0.0587,
    "v": 3.5771,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 97.8449,
    "pm10": 111.7722,
    "u": -3.4332,
    "v": -1.6324,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 9.6277,
    "pm10": 11.2458,
    "u": -0.8184,
    "v": -0.379,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 13.6711,
    "pm10": 14.8004,
    "u": -1.4303,
    "v": -0.9875,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.089,
    "pm10": 9.3636,
    "u": -1.4875,
    "v": -1.733,
    "timestamp_kst": "2025-10-30 09:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 17.9401,
    "pm10": 20.0654,
    "u": 0.6051,
    "v": 1.1615,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 9.1901,
    "pm10": 12.4,
    "u": -0.4072,
    "v": -0.2932,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 8.1607,
    "pm10": 11.9878,
    "u": -0.7212,
    "v": -0.4686,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 10.0739,
    "pm10": 13.9695,
    "u": -0.3671,
    "v": -2.1027,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 12.3214,
    "pm10": 17.5551,
    "u": 0.4497,
    "v": -0.613,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 19.7193,
    "pm10": 23.3641,
    "u": -4.4144,
    "v": 1.5545,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.1929,
    "pm10": 22.1513,
    "u": -0.392,
    "v": 3.5641,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 46.8707,
    "pm10": 59.1276,
    "u": -6.3811,
    "v": -0.1087,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.4638,
    "pm10": 7.9313,
    "u": -0.6743,
    "v": -0.9318,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 10.1312,
    "pm10": 10.652,
    "u": -1.0443,
    "v": -1.0899,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 6.3219,
    "pm10": 8.5901,
    "u": -0.9681,
    "v": -0.322,
    "timestamp_kst": "2025-10-30 12:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 14.2131,
    "pm10": 16.9211,
    "u": 2.2288,
    "v": 0.3896,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 7.43,
    "pm10": 9.5439,
    "u": 0.2109,
    "v": -0.5333,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 7.2646,
    "pm10": 10.3235,
    "u": -0.2543,
    "v": -0.7072,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 9.7329,
    "pm10": 13.7207,
    "u": -0.8199,
    "v": -0.9935,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 10.6504,
    "pm10": 15.068,
    "u": 0.3953,
    "v": -0.4104,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 24.5279,
    "pm10": 28.7145,
    "u": -4.1566,
    "v": 0.8418,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 17.5889,
    "pm10": 24.0372,
    "u": -1.0324,
    "v": 3.2967,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 27.4633,
    "pm10": 32.3005,
    "u": -5.2076,
    "v": 0.4069,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 6.5439,
    "pm10": 7.7833,
    "u": -0.608,
    "v": -1.4439,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 10.076,
    "pm10": 10.5564,
    "u": -1.0675,
    "v": -0.7529,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 5.093,
    "pm10": 6.5996,
    "u": -1.6214,
    "v": 0.4406,
    "timestamp_kst": "2025-10-30 15:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 33.9048,
    "pm10": 36.6382,
    "u": 1.6937,
    "v": 0.3055,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 25.5135,
    "pm10": 27.4585,
    "u": 0.0121,
    "v": -0.4183,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 16.9894,
    "pm10": 19.7067,
    "u": -0.4533,
    "v": -0.2506,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 15.1116,
    "pm10": 18.5087,
    "u": -0.7731,
    "v": -1.4834,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 23.7305,
    "pm10": 27.9619,
    "u": 0.0552,
    "v": -0.1444,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 41.2269,
    "pm10": 44.3783,
    "u": -3.5339,
    "v": -0.1949,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 19.028,
    "pm10": 25.2924,
    "u": -2.5061,
    "v": 2.9958,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 40.4357,
    "pm10": 43.3294,
    "u": -3.9032,
    "v": 0.2858,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.3336,
    "pm10": 9.2317,
    "u": -0.6019,
    "v": -1.5993,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 14.4298,
    "pm10": 14.9765,
    "u": -1.4664,
    "v": -0.2676,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 8.9451,
    "pm10": 9.9854,
    "u": -2.6172,
    "v": -0.9382,
    "timestamp_kst": "2025-10-30 18:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 49.6011,
    "pm10": 52.8159,
    "u": 0.2994,
    "v": 0.4647,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 36.5574,
    "pm10": 38.8733,
    "u": -0.5679,
    "v": 0.0655,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 25.146,
    "pm10": 27.8829,
    "u": -0.845,
    "v": -0.4158,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 20.45,
    "pm10": 23.846,
    "u": -0.7631,
    "v": -2.376,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 35.4146,
    "pm10": 39.7487,
    "u": 0.0644,
    "v": -0.2979,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 49.8126,
    "pm10": 52.6591,
    "u": -3.6684,
    "v": -0.5657,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 22.9945,
    "pm10": 28.4252,
    "u": -2.2419,
    "v": 3.341,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 56.2854,
    "pm10": 59.3404,
    "u": -2.9083,
    "v": 0.2329,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 10.7001,
    "pm10": 11.5266,
    "u": -0.822,
    "v": -1.2139,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 14.4779,
    "pm10": 15.8832,
    "u": -1.9465,
    "v": -0.235,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 11.2835,
    "pm10": 12.3575,
    "u": -2.0398,
    "v": -2.4242,
    "timestamp_kst": "2025-10-30 21:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 54.4853,
    "pm10": 57.7213,
    "u": -0.2535,
    "v": 0.0774,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 37.4222,
    "pm10": 39.8571,
    "u": -0.5333,
    "v": -0.0118,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 25.3023,
    "pm10": 27.4462,
    "u": -0.7221,
    "v": -0.591,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 20.4583,
    "pm10": 23.7578,
    "u": -1.2142,
    "v": -3.0962,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 36.2647,
    "pm10": 41.131,
    "u": -0.0149,
    "v": -0.6788,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 41.6224,
    "pm10": 44.4093,
    "u": -2.9994,
    "v": 0.4597,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 23.4881,
    "pm10": 28.5322,
    "u": -1.289,
    "v": 3.3002,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 58.5226,
    "pm10": 61.9978,
    "u": -1.9774,
    "v": -0.3558,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 11.6744,
    "pm10": 12.444,
    "u": -0.7849,
    "v": -1.054,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 12.4097,
    "pm10": 14.2838,
    "u": -2.0797,
    "v": -0.3377,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 11.8156,
    "pm10": 12.4453,
    "u": -1.8796,
    "v": -2.8921,
    "timestamp_kst": "2025-10-31 00:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 46.3437,
    "pm10": 48.9891,
    "u": -0.6557,
    "v": -0.1359,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 33.9365,
    "pm10": 35.9604,
    "u": -0.6378,
    "v": 0.004,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 21.4533,
    "pm10": 22.9456,
    "u": -0.8514,
    "v": -0.8011,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 17.045,
    "pm10": 20.3171,
    "u": -1.023,
    "v": -3.9972,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 32.4142,
    "pm10": 36.8918,
    "u": -0.0091,
    "v": -0.8683,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 34.9535,
    "pm10": 37.8453,
    "u": -2.2092,
    "v": -0.2008,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 18.761,
    "pm10": 24.3285,
    "u": -0.2764,
    "v": 2.4854,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 47.8008,
    "pm10": 51.1896,
    "u": -1.2502,
    "v": -0.7811,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 10.0772,
    "pm10": 10.9367,
    "u": -1.0063,
    "v": -1.0905,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 10.5322,
    "pm10": 12.6884,
    "u": -2.3413,
    "v": -0.9419,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 10.9913,
    "pm10": 11.6597,
    "u": -2.2793,
    "v": -3.2097,
    "timestamp_kst": "2025-10-31 03:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 40.7167,
    "pm10": 43.0482,
    "u": -0.7506,
    "v": -0.3279,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 32.6132,
    "pm10": 34.9223,
    "u": -0.7307,
    "v": -0.0054,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 21.0393,
    "pm10": 22.6301,
    "u": -0.7445,
    "v": -0.7073,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 14.8763,
    "pm10": 18.0699,
    "u": -1.9343,
    "v": -4.5716,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 30.0047,
    "pm10": 33.3637,
    "u": -0.2305,
    "v": -0.97,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 36.7592,
    "pm10": 39.4801,
    "u": -1.8445,
    "v": -0.624,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.8126,
    "pm10": 23.306,
    "u": 0.0345,
    "v": 1.8502,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 41.6078,
    "pm10": 44.8053,
    "u": -0.6213,
    "v": -1.6351,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 9.7562,
    "pm10": 10.6346,
    "u": -1.1704,
    "v": -1.116,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 10.2001,
    "pm10": 12.4816,
    "u": -2.5766,
    "v": -0.9911,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 10.5688,
    "pm10": 11.3828,
    "u": -2.4373,
    "v": -3.7323,
    "timestamp_kst": "2025-10-31 06:00"
  }
]

```


---

## public/metrics_20251031_KST.json

```json
[
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 44.5509,
    "pm10": 46.8571,
    "u": -0.9277,
    "v": -0.3507,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 33.1334,
    "pm10": 35.6148,
    "u": -0.7495,
    "v": -0.0195,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 21.1657,
    "pm10": 22.6863,
    "u": -0.819,
    "v": -0.6233,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 14.423,
    "pm10": 17.589,
    "u": -1.7077,
    "v": -4.7585,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 30.6251,
    "pm10": 34.0484,
    "u": -0.2523,
    "v": -0.8917,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 33.3163,
    "pm10": 36.1506,
    "u": -1.0974,
    "v": -0.8214,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.1263,
    "pm10": 21.7983,
    "u": 1.8813,
    "v": 2.0383,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 47.2508,
    "pm10": 50.4323,
    "u": -0.8989,
    "v": -1.4911,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 10.6736,
    "pm10": 11.4436,
    "u": -1.2507,
    "v": -1.1627,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.6266,
    "pm10": 13.9236,
    "u": -2.1384,
    "v": -0.6039,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 10.6345,
    "pm10": 11.2947,
    "u": -2.3848,
    "v": -4.0013,
    "timestamp_kst": "2025-10-31 06:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 54.3921,
    "pm10": 56.8703,
    "u": -0.7451,
    "v": -0.4681,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 37.1246,
    "pm10": 39.1893,
    "u": -0.7228,
    "v": -0.2623,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 23.632,
    "pm10": 24.8056,
    "u": -0.5358,
    "v": -0.6156,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 16.9942,
    "pm10": 19.8734,
    "u": -3.3942,
    "v": -4.9377,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 32.0488,
    "pm10": 34.0185,
    "u": -0.6559,
    "v": -1.0832,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 57.5304,
    "pm10": 60.1126,
    "u": -0.1383,
    "v": -1.558,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 19.0152,
    "pm10": 23.1847,
    "u": 2.5534,
    "v": -0.6247,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 83.1936,
    "pm10": 86.3252,
    "u": -0.0667,
    "v": -1.1825,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 12.6837,
    "pm10": 13.284,
    "u": -1.2986,
    "v": -1.1174,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 13.1208,
    "pm10": 15.2206,
    "u": -1.5591,
    "v": -1.1954,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 10.1391,
    "pm10": 10.5352,
    "u": -4.7578,
    "v": -3.9557,
    "timestamp_kst": "2025-10-31 09:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 31.1694,
    "pm10": 34.925,
    "u": -0.337,
    "v": -0.6958,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 20.1042,
    "pm10": 22.7493,
    "u": -1.1393,
    "v": -0.8444,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 12.2375,
    "pm10": 14.1761,
    "u": -0.2053,
    "v": -0.4755,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 12.5566,
    "pm10": 15.8067,
    "u": -3.5293,
    "v": -4.8842,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 16.1594,
    "pm10": 18.841,
    "u": -0.967,
    "v": -1.2657,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 28.0641,
    "pm10": 32.6583,
    "u": 0.5509,
    "v": -2.8242,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 21.5872,
    "pm10": 23.741,
    "u": 2.8775,
    "v": -2.0431,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 50.2435,
    "pm10": 55.4517,
    "u": 2.0125,
    "v": -1.2048,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.3863,
    "pm10": 8.7795,
    "u": -1.8083,
    "v": -1.1689,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.4308,
    "pm10": 13.0639,
    "u": -2.1351,
    "v": -2.4832,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 7.1459,
    "pm10": 7.4643,
    "u": -6.4796,
    "v": -3.8519,
    "timestamp_kst": "2025-10-31 12:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 23.5852,
    "pm10": 26.46,
    "u": 1.0505,
    "v": -0.1713,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 15.7927,
    "pm10": 18.2024,
    "u": -1.4868,
    "v": -0.7044,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 12.312,
    "pm10": 14.1061,
    "u": 0.8366,
    "v": -0.532,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 11.4248,
    "pm10": 14.7281,
    "u": -3.7422,
    "v": -4.2831,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 11.943,
    "pm10": 14.5499,
    "u": -1.2852,
    "v": -1.1753,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 28.9976,
    "pm10": 34.0363,
    "u": 0.23,
    "v": -3.206,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 18.5796,
    "pm10": 19.492,
    "u": 3.7936,
    "v": -2.8126,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 36.7745,
    "pm10": 49.0249,
    "u": 3.3594,
    "v": -2.1253,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 5.7026,
    "pm10": 5.8562,
    "u": -1.7277,
    "v": -1.0741,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 11.2136,
    "pm10": 12.3816,
    "u": -2.0129,
    "v": -2.8271,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 5.6444,
    "pm10": 5.7469,
    "u": -4.5516,
    "v": -3.6985,
    "timestamp_kst": "2025-10-31 15:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 38.5133,
    "pm10": 40.6473,
    "u": 2.1144,
    "v": -0.6313,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 28.8099,
    "pm10": 30.7559,
    "u": 0.0389,
    "v": -0.5034,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 20.3396,
    "pm10": 21.5697,
    "u": 0.978,
    "v": -0.2663,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 13.4112,
    "pm10": 16.3054,
    "u": -2.962,
    "v": -3.969,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 22.6804,
    "pm10": 25.0204,
    "u": -0.5686,
    "v": -0.609,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 48.4639,
    "pm10": 52.3264,
    "u": -0.4242,
    "v": -2.1151,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 16.0986,
    "pm10": 17.3664,
    "u": 3.5401,
    "v": -5.8464,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 65.902,
    "pm10": 77.4214,
    "u": 2.284,
    "v": -1.2997,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.657,
    "pm10": 8.7556,
    "u": -0.9369,
    "v": -0.8724,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 9.1118,
    "pm10": 9.6441,
    "u": -1.9714,
    "v": -2.3914,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 4.3337,
    "pm10": 4.3899,
    "u": -2.9584,
    "v": -3.1806,
    "timestamp_kst": "2025-10-31 18:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 44.5737,
    "pm10": 46.8022,
    "u": 1.3627,
    "v": -0.5967,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 35.2819,
    "pm10": 37.2559,
    "u": 0.4776,
    "v": 0.3749,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 24.0749,
    "pm10": 25.3787,
    "u": 0.5388,
    "v": -0.3082,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 16.4258,
    "pm10": 19.2849,
    "u": -0.7397,
    "v": -3.2386,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 31.9701,
    "pm10": 34.5543,
    "u": -0.1393,
    "v": -0.2542,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 85.6793,
    "pm10": 89.1823,
    "u": 0.4074,
    "v": -1.1638,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 26.4767,
    "pm10": 29.7805,
    "u": 2.7796,
    "v": -6.1207,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 86.3608,
    "pm10": 96.8234,
    "u": 2.8454,
    "v": -1.3216,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 9.4109,
    "pm10": 9.5619,
    "u": -0.4724,
    "v": -0.5441,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 6.5573,
    "pm10": 6.8219,
    "u": -1.1674,
    "v": -1.8563,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 3.4805,
    "pm10": 3.584,
    "u": -0.8634,
    "v": -2.9234,
    "timestamp_kst": "2025-10-31 21:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 42.7181,
    "pm10": 45.1896,
    "u": 1.3415,
    "v": -0.3564,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 40.5462,
    "pm10": 42.4769,
    "u": 0.6817,
    "v": 0.601,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 28.4653,
    "pm10": 29.9581,
    "u": 0.6128,
    "v": -0.5255,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 16.3897,
    "pm10": 19.4253,
    "u": 1.3393,
    "v": -2.4574,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 33.7041,
    "pm10": 36.303,
    "u": 0.4755,
    "v": -0.0551,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 101.4878,
    "pm10": 105.2354,
    "u": 1.5179,
    "v": -1.3853,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 29.3743,
    "pm10": 34.7234,
    "u": 3.0297,
    "v": -5.394,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 69.8969,
    "pm10": 77.8092,
    "u": 2.6786,
    "v": -2.0464,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 8.0109,
    "pm10": 8.2873,
    "u": -0.0324,
    "v": -0.5489,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.5417,
    "pm10": 7.7156,
    "u": 0.119,
    "v": -1.0281,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 4.202,
    "pm10": 4.3573,
    "u": 0.2333,
    "v": -2.1298,
    "timestamp_kst": "2025-11-01 00:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 26.6363,
    "pm10": 28.3236,
    "u": 1.1273,
    "v": -0.2409,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 38.2724,
    "pm10": 40.1702,
    "u": 0.9772,
    "v": 0.6695,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 27.0746,
    "pm10": 28.7252,
    "u": 0.6599,
    "v": -0.7153,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 16.7687,
    "pm10": 19.8058,
    "u": 2.7937,
    "v": -2.1485,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 31.1342,
    "pm10": 33.4474,
    "u": 0.7151,
    "v": -0.251,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 77.814,
    "pm10": 80.7068,
    "u": 2.0402,
    "v": -2.3112,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 30.3634,
    "pm10": 36.1764,
    "u": 3.742,
    "v": -5.1762,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 64.9169,
    "pm10": 69.9579,
    "u": 2.816,
    "v": -1.6666,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 9.9952,
    "pm10": 10.7257,
    "u": 0.3939,
    "v": -0.0192,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.1781,
    "pm10": 7.3724,
    "u": 1.0765,
    "v": -0.0157,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 4.1178,
    "pm10": 4.3671,
    "u": 0.3204,
    "v": -1.6696,
    "timestamp_kst": "2025-11-01 03:00"
  },
  {
    "city": "서울",
    "lon": 126.978,
    "lat": 37.5665,
    "pm25": 19.9755,
    "pm10": 21.1486,
    "u": 1.6829,
    "v": -0.5225,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "대전",
    "lon": 127.3845,
    "lat": 36.3504,
    "pm25": 30.3918,
    "pm10": 32.1373,
    "u": 1.359,
    "v": 0.6575,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "광주",
    "lon": 126.8526,
    "lat": 35.1595,
    "pm25": 21.7446,
    "pm10": 23.4332,
    "u": 1.2376,
    "v": -0.9208,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "부산",
    "lon": 129.0756,
    "lat": 35.1796,
    "pm25": 18.7002,
    "pm10": 21.2863,
    "u": 3.254,
    "v": -1.905,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "대구",
    "lon": 128.6014,
    "lat": 35.8714,
    "pm25": 30.9423,
    "pm10": 32.9408,
    "u": 0.8655,
    "v": 0.0613,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "상하이",
    "lon": 121.4737,
    "lat": 31.2304,
    "pm25": 63.66,
    "pm10": 65.2492,
    "u": 2.036,
    "v": -2.9508,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "칭다오",
    "lon": 120.3826,
    "lat": 36.0671,
    "pm25": 32.5313,
    "pm10": 38.7107,
    "u": 4.1134,
    "v": -4.7821,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "톈진",
    "lon": 117.3616,
    "lat": 39.3434,
    "pm25": 64.1205,
    "pm10": 70.0838,
    "u": 2.6901,
    "v": -0.9161,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "후쿠오카",
    "lon": 130.4017,
    "lat": 33.5904,
    "pm25": 9.9506,
    "pm10": 11.462,
    "u": 0.3453,
    "v": 0.4156,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "오사카",
    "lon": 135.5023,
    "lat": 34.6937,
    "pm25": 7.8374,
    "pm10": 8.2758,
    "u": 1.5232,
    "v": 0.212,
    "timestamp_kst": "2025-11-01 06:00"
  },
  {
    "city": "히로시마",
    "lon": 132.4553,
    "lat": 34.3853,
    "pm25": 5.0421,
    "pm10": 5.4199,
    "u": 0.8975,
    "v": -1.5593,
    "timestamp_kst": "2025-11-01 06:00"
  }
]

```


---

## public/reports/index_daily_reports.html

```
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvData Insight Studio – Daily Reports</title>
<style>
  :root{--w:960px}
  body{margin:0 auto;max-width:var(--w);padding:24px 16px;font-family:-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;line-height:1.5}
  header{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  h1{font-size:20px;margin:0 8px 0 0}
  .controls{display:flex;align-items:center;gap:8px}
  select,button{font-size:14px;padding:6px 10px}
  .hint{color:#666;font-size:12px;margin:8px 0 12px}
  #viewer{width:100%;border:1px solid #e1e5ef;border-radius:10px}
  .error{background:#fff3f3;border:1px solid #ffd3d3;color:#a00;padding:10px;border-radius:8px}
</style>
</head>
<body>
<header>
  <h1>EnvData Insight Studio – Daily Reports</h1>
  <div class="controls">
    <button id="prevBtn" aria-label="이전 날짜">&lt;</button>
    <select id="dateSel"></select>
    <button id="nextBtn" aria-label="다음 날짜">&gt;</button>
    <button id="openBtn">새 탭으로 열기</button>
  </div>
</header>
<p class="hint">드롭다운에서 날짜를 선택하세요. 보고서는 아래 뷰어에서 바로 열립니다.</p>

<div id="status"></div>
<iframe id="viewer" title="Daily Report" src="about:blank" loading="lazy"></iframe>

<script>
(async function(){
  const manifestUrl = 'manifest.json';                  // this file sits in /public/reports/
  const reportHref  = (d)=> `report_${d}_0600.html`;    // same folder
  const q = new URLSearchParams(location.search);

  const $sel   = document.getElementById('dateSel');
  const $prev  = document.getElementById('prevBtn');
  const $next  = document.getElementById('nextBtn');
  const $open  = document.getElementById('openBtn');
  const $view  = document.getElementById('viewer');
  const $stat  = document.getElementById('status');

  function showError(msg){
    $stat.innerHTML = `<div class="error">${msg}</div>`;
  }

  let dates = [];
  let cur = null;

  try{
    const res = await fetch(manifestUrl, {cache:'no-cache'});
    if(!res.ok) throw new Error(`manifest ${res.status}`);
    const mf = await res.json();
    dates = (mf.dates||[]).slice();
    if(!dates.length) throw new Error('no dates in manifest');

    // fill dropdown (assume dates[0] is latest)
    $sel.innerHTML = '';
    dates.forEach(d=>{
      const o=document.createElement('option');
      o.value=d; o.textContent=d; $sel.appendChild(o);
    });

    const paramDate = q.get('date');
    cur = dates.includes(paramDate) ? paramDate : (mf.latest || dates[0]);
    $sel.value = cur;
    loadReport(cur, false);

  }catch(e){
    showError('목차(manifest.json)를 불러오지 못했습니다. 배포 경로를 확인하세요.');
    console.error(e);
    return;
  }

  function loadReport(d, pushUrl=true){
    cur = d;
    $sel.value = d;
    $view.src = reportHref(d);
    if(pushUrl){
      const url = new URL(location.href);
      url.searchParams.set('date', d);
      history.replaceState(null, '', url);
    }
  }

  function move(offset){
    const i = dates.indexOf(cur);
    if(i === -1) return;
    const j = Math.min(Math.max(i + offset, 0), dates.length-1);
    if(j !== i) loadReport(dates[j]);
  }

  // controls
  $sel.addEventListener('change', ()=> loadReport($sel.value));
  $prev.addEventListener('click', ()=> move(+1)); // dates[0]가 최신 → +1은 과거
  $next.addEventListener('click', ()=> move(-1));
  $open.addEventListener('click', ()=> window.open(reportHref(cur), '_blank'));

  // auto-resize iframe (same-origin)
  function resizeIframe(){
    try{
      const doc = $view.contentDocument || $view.contentWindow.document;
      if(!doc) return;
      const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight, 800);
      $view.style.height = h + 'px';
    }catch(_){}
  }
  $view.addEventListener('load', resizeIframe);
  window.addEventListener('resize', ()=> setTimeout(resizeIframe, 100));
})();
</script>
</body>
</html>

```


---

## public/reports/manifest.json

```json
{
  "latest": "20251031",
  "dates": [
    "20251031",
    "20251030",
    "20251029",
    "20251028"
  ]
}
```


---

## public/reports/report_20251028_0600.html

```
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvData Insight Studio – Daily Overview</title>
<style>
  :root {
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }
  body{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }
  header h1{ margin:0; font-size:22px; font-weight:700 }
  header .meta{ margin:4px 0 24px; color:#555; font-size:13px }
  section h2{ font-size:18px; margin:24px 0 8px }
  .bullets{ margin:8px 0 12px 18px }
  .bullets li{ margin:2px 0 }
  .chart img{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }
  .montage img{ width:100%; display:block; border:none; margin:8px 0 12px }
  .table-wrap{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }
  .unit-label{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }
  table.city-table{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }
  .city-table thead th, .city-table tbody td{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }
  .city-table thead th:last-child, .city-table tbody td:last-child{ border-right:0 }
  .city-table thead th{ background:var(--soft); color:#222; font-weight:600 }
  .city-table thead .group-row th.group{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }
  .city-table td.thresh{ background:var(--warn-bg) }
  .city-table td.top3{ color:var(--top3); font-weight:700 }
  footer{ margin-top:28px; color:#666; font-size:12px; text-align:center }
</style>
</head>
<body>
<header>
  <h1>EnvData Insight Studio – Daily Report (2025년 10월 28일)</h1>
  <div class="meta">데이터 출처: ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    <p>아침에는 전국 대부분 지역에서 대체로 낮은 농도를 보이겠으나 밤사이에는 수도권을 중심으로 농도가 상승하겠습니다. <b>서울은 00:00에 50 μg/m³ 안팎으로 35 μg/m³ 기준을 넘기겠습니다</b></p>
    <figure class="montage"><img src="../tiles/20251028/pm25_wind_20251028_KST_montage.png" alt="PM2.5 3×3 몽타주"></figure>
  </section>

  <section id="impact">
    <ul class='bullets'><li>수도권 밤사이 서울 00:00 50 μg/m³로 35 μg/m³ 기준 초과 가능성 높음</li><li>톈진 09:00 92 μg/m³ 고농도 발생, 서풍 성분으로 중부권 단기 영향 예상</li></ul>
    <h2>2) 영향 해석</h2>
    <p>중국 톈진의 오전 09:00 고농도(약 92 μg/m³)가 서쪽에서 동쪽으로 향하는 흐름을 타고 한반도로 일부 유입되겠고 이로 인해 중부권에 일시적 영향을 주겠습니다. <b>야간 약한 대기혼합과 역전층이 배경농도 축적을 촉진하겠으며</b> 저층 바람 약화와 분지 지형이 체류 시간을 늘리겠습니다</p>
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    <figure class="chart"><img src="timeseries_20251028_0600.png" alt="PM2.5 시계열 그래프"></figure>
    
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr><th class="stub time-col">시간</th><th>서울</th><th>대전</th><th>광주</th><th>부산</th><th>대구</th><th>상하이</th><th>칭다오</th><th>톈진</th><th>후쿠오카</th><th>오사카</th><th>히로시마</th></tr>
      <tr class="group-row"><th></th><th class="group" colspan="5">국내</th><th class="group" colspan="3">중국</th><th class="group" colspan="3">일본</th></tr>
    </thead>
    <tbody><tr><td class="time-col"><b>06:00</b></td><td>19</td><td>15</td><td>9</td><td>8</td><td>12</td><td class="thresh">37</td><td>15</td><td class="thresh">52</td><td>8</td><td>9</td><td>9</td></tr><tr><td class="time-col"><b>09:00</b></td><td>26</td><td>16</td><td>10</td><td>9</td><td>12</td><td class="thresh">56</td><td>16</td><td class="thresh top3">92</td><td>8</td><td>8</td><td>9</td></tr><tr><td class="time-col"><b>12:00</b></td><td>12</td><td>8</td><td>7</td><td>6</td><td>6</td><td>22</td><td>13</td><td class="thresh">69</td><td>6</td><td>8</td><td>7</td></tr><tr><td class="time-col"><b>15:00</b></td><td>12</td><td>9</td><td>6</td><td>6</td><td>5</td><td>22</td><td>13</td><td class="thresh">60</td><td>7</td><td>7</td><td>7</td></tr><tr><td class="time-col"><b>18:00</b></td><td>24</td><td>18</td><td>11</td><td>11</td><td>14</td><td>32</td><td>13</td><td class="thresh top3">70</td><td>8</td><td>10</td><td>7</td></tr><tr><td class="time-col"><b>21:00</b></td><td class="thresh">44</td><td>27</td><td>15</td><td>15</td><td>20</td><td class="thresh">48</td><td>12</td><td class="thresh top3">77</td><td>8</td><td>12</td><td>8</td></tr><tr><td class="time-col"><b>00:00</b></td><td class="thresh">50</td><td>29</td><td>18</td><td>17</td><td>23</td><td class="thresh">44</td><td>10</td><td class="thresh">64</td><td>9</td><td>16</td><td>8</td></tr><tr><td class="time-col"><b>03:00</b></td><td class="thresh">41</td><td>27</td><td>19</td><td>16</td><td>23</td><td>33</td><td>9</td><td class="thresh">51</td><td>7</td><td>13</td><td>7</td></tr><tr><td class="time-col"><b>06:00</b></td><td>32</td><td>26</td><td>21</td><td>15</td><td>24</td><td>31</td><td>9</td><td class="thresh">45</td><td>6</td><td>14</td><td>7</td></tr></tbody>
  </table>
</div>

  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    <p>민감군은 밤 시간대(00:00~03:00) 외출을 자제하고 보건용 마스크 착용을 권하겠습니다. 일반인은 낮 시간대(12:00 전후)에 실외 활동을 집중하시기를 권장드리겠습니다. 실내 환기는 오후 14:00~16:00에 실시하고 고농도 시간대에는 공기청정기 가동을 권고드리겠습니다. 야간에는 배경농도가 쌓이겠으니 창문을 닫고 실내 머무르기를 유의하겠습니다. 외출 전 지역별 예보와 실시간 지수를 확인하여 활동 시점을 조정하시기를 권하겠습니다</p>
  </section>
</main>

<footer>© 2025 EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>
```


---

## public/reports/report_20251029_0600.html

```
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvData Insight Studio – Daily Report</title>
<style>
  :root {
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }
  body{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }
  header h1{ margin:0; font-size:22px; font-weight:700 }
  header .meta{ margin:4px 0 24px; color:#555; font-size:13px }
  section h2{ font-size:18px; margin:24px 0 8px }
  .bullets{ margin:8px 0 12px 18px }
  .bullets li{ margin:2px 0 }
  .chart img{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }
  .montage img{ width:100%; display:block; border:none; margin:8px 0 12px }
  .table-wrap{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }
  .unit-label{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }
  table.city-table{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }
  .city-table thead th, .city-table tbody td{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }
  .city-table thead th:last-child, .city-table tbody td:last-child{ border-right:0 }
  .city-table thead th{ background:var(--soft); color:#222; font-weight:600 }
  .city-table thead .group-row th.group{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }
  .city-table td.thresh{ background:var(--warn-bg) }
  .city-table td.top3{ color:var(--top3); font-weight:700 }
  footer{ margin-top:28px; color:#666; font-size:12px; text-align:center }
</style>
</head>
<body>
<header>
  <h1>EnvData Insight Studio – Daily Report (2025년 10월 29일)</h1>
  <div class="meta">데이터 출처: ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    <p>오늘 대기 흐름은 서쪽에서 동쪽으로 이동하는 고농도 구간의 영향으로 저녁부터 야간에 중부와 서울권에서 농도가 상승하겠습니다. 특히 서울은 <b>21:00에 최고 약 69 μg/m³로 35 μg/m³ 기준을 넘어설 가능성 높겠습니다</b>. 낮 시간에는 혼합이 좋아 농도는 낮게 유지되겠습니다.</p>
    <figure class="montage"><img src="../tiles/20251029/pm25_wind_20251029_KST_montage.png" alt="PM2.5 3×3 몽타주"></figure>
  </section>

  <section id="impact">
    <ul class='bullets'><li>수도권 밤사이 기준 초과 가능성 높음</li><li>톈진 고농도(142 μg/m³) 서해상 통해 동진으로 중부권 야간 영향 예상</li></ul>
    <h2>2) 영향 해석</h2>
    <p>중국 톈진에서 21:00에 142 μg/m³ 수준의 초고농도가 나타나겠고 서풍을 타고 서해상을 통해 한반도로 유입되겠으며 중부와 서울권 야간 농도를 끌어올리겠겠습니다. 야간에는 지표부 풍속이 약해지겠고 <b>혼합층이 얕아져 오염물질이 오래 잔류하겠습니다</b>. 해안과 분지 지형에서는 농도 차이가 커지겠으니 단기 변동성에 유의하겠습니다.</p>
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    <figure class="chart"><img src="timeseries_20251029_0600.png" alt="PM2.5 시계열 그래프"></figure>
    
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr><th class="stub time-col">시간</th><th>서울</th><th>대전</th><th>광주</th><th>부산</th><th>대구</th><th>상하이</th><th>칭다오</th><th>톈진</th><th>후쿠오카</th><th>오사카</th><th>히로시마</th></tr>
      <tr class="group-row"><th></th><th class="group" colspan="5">국내</th><th class="group" colspan="3">중국</th><th class="group" colspan="3">일본</th></tr>
    </thead>
    <tbody><tr><td class="time-col"><b>06:00</b></td><td>33</td><td>29</td><td>21</td><td>15</td><td>24</td><td>31</td><td>9</td><td class="thresh">50</td><td>6</td><td>14</td><td>7</td></tr><tr><td class="time-col"><b>09:00</b></td><td class="thresh">40</td><td>27</td><td>21</td><td>16</td><td>25</td><td class="thresh">48</td><td>11</td><td class="thresh">57</td><td>7</td><td>11</td><td>7</td></tr><tr><td class="time-col"><b>12:00</b></td><td>20</td><td>8</td><td>6</td><td>11</td><td>10</td><td>18</td><td>10</td><td class="thresh">49</td><td>6</td><td>8</td><td>5</td></tr><tr><td class="time-col"><b>15:00</b></td><td>16</td><td>7</td><td>6</td><td>10</td><td>8</td><td>22</td><td>12</td><td class="thresh">45</td><td>5</td><td>7</td><td>5</td></tr><tr><td class="time-col"><b>18:00</b></td><td class="thresh">44</td><td>21</td><td>17</td><td>12</td><td>19</td><td class="thresh">35</td><td>14</td><td class="thresh">76</td><td>6</td><td>8</td><td>6</td></tr><tr><td class="time-col"><b>21:00</b></td><td class="thresh">69</td><td>28</td><td>24</td><td>14</td><td>25</td><td class="thresh">51</td><td>14</td><td class="thresh top3">142</td><td>7</td><td>12</td><td>6</td></tr><tr><td class="time-col"><b>00:00</b></td><td class="thresh">53</td><td>25</td><td>22</td><td>14</td><td>29</td><td class="thresh">44</td><td>13</td><td class="thresh top3">132</td><td>7</td><td>15</td><td>7</td></tr><tr><td class="time-col"><b>03:00</b></td><td class="thresh">36</td><td>23</td><td>19</td><td>12</td><td>28</td><td class="thresh">38</td><td>12</td><td class="thresh top3">113</td><td>6</td><td>12</td><td>7</td></tr><tr><td class="time-col"><b>06:00</b></td><td>31</td><td>23</td><td>18</td><td>12</td><td>25</td><td class="thresh">36</td><td>12</td><td class="thresh">94</td><td>7</td><td>12</td><td>9</td></tr></tbody>
  </table>
</div>

  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    <p>민감군은 야간 외출 시 마스크를 착용하시길 권하겠습니다. 실내 환기는 낮 시간대(예: 오전 10시–15시)에 실시하고 공기청정기는 야간 고농도 시간대에 가동하시면 효과적이겠습니다. 야외 활동 전에는 최신 지수를 확인해 주시길 부탁드리겠습니다.</p>
  </section>
</main>

<footer>© 2025 EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>
```


---

## public/reports/report_20251030_0600.html

```
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvData Insight Studio – Daily Report</title>
<style>
  :root {
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }
  body{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }
  header h1{ margin:0; font-size:22px; font-weight:700 }
  header .meta{ margin:4px 0 24px; color:#555; font-size:13px }
  section h2{ font-size:18px; margin:24px 0 8px }
  .bullets{ margin:8px 0 12px 18px }
  .bullets li{ margin:2px 0 }
  .chart img{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }
  .montage img{ width:100%; display:block; border:none; margin:8px 0 12px }
  .table-wrap{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }
  .unit-label{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }
  table.city-table{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }
  .city-table thead th, .city-table tbody td{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }
  .city-table thead th:last-child, .city-table tbody td:last-child{ border-right:0 }
  .city-table thead th{ background:var(--soft); color:#222; font-weight:600 }
  .city-table thead .group-row th.group{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }
  .city-table td.thresh{ background:var(--warn-bg) }
  .city-table td.top3{ color:var(--top3); font-weight:700 }
  footer{ margin-top:28px; color:#666; font-size:12px; text-align:center }
</style>
</head>
<body>
<header>
  <h1>EnvData Insight Studio – Daily Report (2025년 10월 30일)</h1>
  <div class="meta">데이터 출처: ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    <p>중부지방과 수도권에서 밤사이 미세먼지 농도가 상승하겠으며 서울은 밤 21시 이후 35 μg/m³ 기준을 넘기는 시점이 있겠습니다. <b>밤~초저녁에 중국발 고농도층과 국내 정체 대기가 겹쳐 영향이 커지겠습니다</b>. 낮에는 대기 확산으로 농도가 낮아지는 흐름이 보이겠습니다.</p>
    <figure class="montage"><img src="../tiles/20251030/pm25_wind_20251030_KST_montage.png" alt="PM2.5 3×3 몽타주"></figure>
  </section>

  <section id="impact">
    <ul class='bullets'><li>수도권 밤 21시 이후 35 μg/m³ 초과 가능성 높음</li><li>톈진 오전(09:00) 고농도 발생, 서풍 수송으로 중부권 단기 영향 예상</li></ul>
    <h2>2) 영향 해석</h2>
    <p>서풍 성분이 강화되겠으며 산둥권과 톈진에서 형성된 고농도 에어마스가 서쪽에서 동쪽으로 이동하겠습니다. <b>톈진의 오전 09:00 고농도(약 98 μg/m³)가 밤사이 잔류층으로 수도권에 유입되기 쉬워지겠습니다</b>. 야간 저층 역전으로 혼합층이 얕아져 농도가 일시 상승하겠습니다.</p>
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    <figure class="chart"><img src="timeseries_20251030_0600.png" alt="PM2.5 시계열 그래프"></figure>
    
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr><th class="stub time-col">시간</th><th>서울</th><th>대전</th><th>광주</th><th>부산</th><th>대구</th><th>상하이</th><th>칭다오</th><th>톈진</th><th>후쿠오카</th><th>오사카</th><th>히로시마</th></tr>
      <tr class="group-row"><th></th><th class="group" colspan="5">국내</th><th class="group" colspan="3">중국</th><th class="group" colspan="3">일본</th></tr>
    </thead>
    <tbody><tr><td class="time-col"><b>06:00</b></td><td>31</td><td>23</td><td>18</td><td>12</td><td>25</td><td class="thresh">36</td><td>12</td><td class="thresh top3">94</td><td>7</td><td>12</td><td>9</td></tr><tr><td class="time-col"><b>09:00</b></td><td class="thresh">40</td><td>28</td><td>17</td><td>15</td><td>24</td><td class="thresh">54</td><td>16</td><td class="thresh top3">98</td><td>10</td><td>14</td><td>7</td></tr><tr><td class="time-col"><b>12:00</b></td><td>18</td><td>9</td><td>8</td><td>10</td><td>12</td><td>20</td><td>16</td><td class="thresh">47</td><td>6</td><td>10</td><td>6</td></tr><tr><td class="time-col"><b>15:00</b></td><td>14</td><td>7</td><td>7</td><td>10</td><td>11</td><td>25</td><td>18</td><td>27</td><td>7</td><td>10</td><td>5</td></tr><tr><td class="time-col"><b>18:00</b></td><td>34</td><td>26</td><td>17</td><td>15</td><td>24</td><td class="thresh">41</td><td>19</td><td class="thresh">40</td><td>8</td><td>14</td><td>9</td></tr><tr><td class="time-col"><b>21:00</b></td><td class="thresh">50</td><td class="thresh">37</td><td>25</td><td>20</td><td class="thresh">35</td><td class="thresh">50</td><td>23</td><td class="thresh">56</td><td>11</td><td>14</td><td>11</td></tr><tr><td class="time-col"><b>00:00</b></td><td class="thresh">54</td><td class="thresh">37</td><td>25</td><td>20</td><td class="thresh">36</td><td class="thresh">42</td><td>23</td><td class="thresh top3">59</td><td>12</td><td>12</td><td>12</td></tr><tr><td class="time-col"><b>03:00</b></td><td class="thresh">46</td><td>34</td><td>21</td><td>17</td><td>32</td><td class="thresh">35</td><td>19</td><td class="thresh">48</td><td>10</td><td>11</td><td>11</td></tr><tr><td class="time-col"><b>06:00</b></td><td class="thresh">41</td><td>33</td><td>21</td><td>15</td><td>30</td><td class="thresh">37</td><td>17</td><td class="thresh">42</td><td>10</td><td>10</td><td>11</td></tr></tbody>
  </table>
</div>

  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    <p>민감군은 외출 시 보건용 마스크 착용을 권장하겠습니다. 실외 활동은 오전(09시 이전)에 집중하시면 노출을 줄이실 수 있겠습니다. 환기와 실내 공기청정은 낮 시간대 확산이 좋을 때 실시하시면 효과적이겠습니다. 야간에는 실내 창문을 닫고 공기청정기를 가동하시면 실내 농도 관리에 도움이 되겠습니다. 외출 전에는 실시간 지수를 확인하시면 안전한 활동 판단에 유용하겠습니다.</p>
  </section>
</main>

<footer>© 2025 EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>
```


---

## public/reports/report_20251031_0600.html

```
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvData Insight Studio – Daily Report</title>
<style>
  :root {
    --bg:#fff; --fg:#111; --muted:#666; --border:#d6dbe6; --grid:#e9edf5;
    --soft:#f5f7fb; --soft2:#f0f3fa; --warn-bg:#ffe9e9; --top3:#c00;
    --viz-width: 860px;
  }
  body{ margin:0 auto; max-width:960px; padding:32px 20px;
        font-family:-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Helvetica,Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
        line-height:1.6; color:var(--fg) }
  header h1{ margin:0; font-size:22px; font-weight:700 }
  header .meta{ margin:4px 0 24px; color:#555; font-size:13px }
  section h2{ font-size:18px; margin:24px 0 8px }
  .bullets{ margin:8px 0 12px 18px }
  .bullets li{ margin:2px 0 }
  .chart img{ width:var(--viz-width); max-width:100%; display:block; border:none; margin:0 auto 10px }
  .montage img{ width:100%; display:block; border:none; margin:8px 0 12px }
  .table-wrap{ position:relative; width:var(--viz-width); max-width:100%; margin:8px auto 18px }
  .unit-label{ position:absolute; right:0; top:-20px; font-size:12px; color:#666 }
  table.city-table{ width:100%; border-collapse:collapse; table-layout:fixed; text-align:center;
                     font-variant-numeric:tabular-nums; border:1px solid var(--border); background:#fff;
                     border-radius:10px; overflow:hidden }
  .city-table thead th, .city-table tbody td{ border-bottom:1px solid var(--grid); border-right:1px solid var(--grid);
                                               padding:6px 8px; font-size:13px; text-align:center }
  .city-table thead th:last-child, .city-table tbody td:last-child{ border-right:0 }
  .city-table thead th{ background:var(--soft); color:#222; font-weight:600 }
  .city-table thead .group-row th.group{ background:var(--soft2); color:#223; text-align:center; font-weight:700 }
  .city-table td.thresh{ background:var(--warn-bg) }
  .city-table td.top3{ color:var(--top3); font-weight:700 }
  footer{ margin-top:28px; color:#666; font-size:12px; text-align:center }
</style>
</head>
<body>
<header>
  <h1>EnvData Insight Studio – Daily Report (2025년 10월 31일)</h1>
  <div class="meta">데이터 출처: ECMWF/CAMS(PM2.5) · NOAA/GFS(풍향, 풍속)</div>
</header>

<main>
  <section id="summary">
    <h2>1) 종합 요약</h2>
    <p>31일 서울 등 중부권은 오전에 농도가 상승하여 09:00경 최고 54 μg/m³로 35 μg/m³을 넘어설 것으로 보이겠습니다. <b>상하이·톈진 등 동쪽 고농도가 밤과 새벽에 한반도로 유입되며 오전 영향이 강화되겠습니다</b></p>
    <figure class="montage"><img src="../tiles/20251031/pm25_wind_20251031_KST_montage.png" alt="PM2.5 3×3 몽타주"></figure>
  </section>

  <section id="impact">
    <ul class='bullets'><li>수도권 오전 09:00 주변 최고 54 μg/m³로 35 μg/m³ 초과 가능성 높음</li><li>상하이·톈진 고농도 동진으로 밤사이 중부권 일시 상승 영향 예상</li></ul>
    <h2>2) 영향 해석</h2>
    <p>중국 동부에서 발생한 고농도 물질은 약한 서풍 성분과 저층 기류의 정체로 서해를 건너 우리나라에 도달하겠고, 특히 밤 21:00~00:00 사이에 유입이 집중되겠습니다. <b>밤사이 기층 안정으로 혼합층이 얕아져 오염물질이 축적되기 쉬우며</b> 저녁부터 다음날 새벽까지 농도가 지속되겠습니다</p>
  </section>

  <section id="table">
    <h2>3) 도시별 예보</h2>
    <figure class="chart"><img src="timeseries_20251031_0600.png" alt="PM2.5 시계열 그래프"></figure>
    
<div class="table-wrap">
  <div class="unit-label">(단위: μg/m³)</div>
  <table class="city-table">
    <thead>
      <tr><th class="stub time-col">시간</th><th>서울</th><th>대전</th><th>광주</th><th>부산</th><th>대구</th><th>상하이</th><th>칭다오</th><th>톈진</th><th>후쿠오카</th><th>오사카</th><th>히로시마</th></tr>
      <tr class="group-row"><th></th><th class="group" colspan="5">국내</th><th class="group" colspan="3">중국</th><th class="group" colspan="3">일본</th></tr>
    </thead>
    <tbody><tr><td class="time-col"><b>06:00</b></td><td class="thresh">45</td><td>33</td><td>21</td><td>14</td><td>31</td><td>33</td><td>16</td><td class="thresh">47</td><td>11</td><td>12</td><td>11</td></tr><tr><td class="time-col"><b>09:00</b></td><td class="thresh">54</td><td class="thresh">37</td><td>24</td><td>17</td><td>32</td><td class="thresh">58</td><td>19</td><td class="thresh">83</td><td>13</td><td>13</td><td>10</td></tr><tr><td class="time-col"><b>12:00</b></td><td>31</td><td>20</td><td>12</td><td>13</td><td>16</td><td>28</td><td>22</td><td class="thresh">50</td><td>8</td><td>11</td><td>7</td></tr><tr><td class="time-col"><b>15:00</b></td><td>24</td><td>16</td><td>12</td><td>11</td><td>12</td><td>29</td><td>19</td><td class="thresh">37</td><td>6</td><td>11</td><td>6</td></tr><tr><td class="time-col"><b>18:00</b></td><td class="thresh">39</td><td>29</td><td>20</td><td>13</td><td>23</td><td class="thresh">48</td><td>16</td><td class="thresh">66</td><td>9</td><td>9</td><td>4</td></tr><tr><td class="time-col"><b>21:00</b></td><td class="thresh">45</td><td class="thresh">35</td><td>24</td><td>16</td><td>32</td><td class="thresh top3">86</td><td>26</td><td class="thresh top3">86</td><td>9</td><td>7</td><td>3</td></tr><tr><td class="time-col"><b>00:00</b></td><td class="thresh">43</td><td class="thresh">41</td><td>28</td><td>16</td><td>34</td><td class="thresh top3">101</td><td>29</td><td class="thresh">70</td><td>8</td><td>8</td><td>4</td></tr><tr><td class="time-col"><b>03:00</b></td><td>27</td><td class="thresh">38</td><td>27</td><td>17</td><td>31</td><td class="thresh">78</td><td>30</td><td class="thresh">65</td><td>10</td><td>7</td><td>4</td></tr><tr><td class="time-col"><b>06:00</b></td><td>20</td><td>30</td><td>22</td><td>19</td><td>31</td><td class="thresh">64</td><td>33</td><td class="thresh">64</td><td>10</td><td>8</td><td>5</td></tr></tbody>
  </table>
</div>

  </section>

  <section id="health">
    <h2>4) 생활·보건 안내</h2>
    <p>민감군은 외출 시 보건용 마스크 착용을 권하겠습니다. 실외 활동은 오전 고농도인 09:00 전후를 피하시는 것이 좋겠습니다. 환기는 오후 늦게 농도가 낮아진 시간에 시행하시는 것이 효과적이겠습니다. 공기청정기는 밤 21:00~00:00 사이 고농도 기간에 가동을 유지하시면 도움이 되겠습니다. 외출 전에는 실시간 지수를 확인하셔서 활동 강도를 조절하시면 안전하겠습니다.</p>
  </section>
</main>

<footer>© 2025 EnvData Insight Studio · Source: ECMWF/CAMS, NOAA/GFS</footer>
</body></html>
```


---

## scripts/publish_daily.sh

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] (${BASH_SOURCE[0]}:${LINENO}) 커맨드 실패"; exit 1' ERR

# ---- 경로 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLIC_DIR="${PROJECT_DIR}/public"

# venv 우선, 없다면 시스템 python
VENV_BIN="${PROJECT_DIR}/.venv/bin"
if [[ -x "${VENV_BIN}/python" ]]; then PYBIN="${VENV_BIN}/python"
elif command -v python3 >/dev/null 2>&1; then PYBIN="$(command -v python3)"
else PYBIN="$(command -v python)"; fi

# (선택) nvm
if [[ -n "${HOME:-}" && -s "${HOME}/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.nvm/nvm.sh"
  nvm use 20 >/dev/null || true
fi

# ---- 유틸 ----
require_file() { local f="$1"; local msg="${2:-missing file: $1}";
  [[ -f "$f" ]] || { echo "[FATAL] $msg" >&2; exit 1; }; }

# R 경고는 에러로 올리지 않음 (GDAL warning 무시)
run_r() { Rscript --vanilla "$1"; }

run_py() { "$PYBIN" "$@"; }

# ---- 파이프라인 (날짜 주입 금지: 각 스크립트가 내부 규칙대로 동작) ----

# 1) (선택) 원천 수집
if [[ "${DO_FETCH:-true}" == "true" ]]; then
  echo "[STEP] fetch GEE/CAMS/GFS"
  run_py "${PROJECT_DIR}/pipeline/adapters/fetch_gee_data.py"
fi

# 2) 타일/몽타주
echo "[STEP] build tiles & montage (R)"
run_r "${PROJECT_DIR}/r/make_pm25_wind_tiles.R"

# 3) 시계열
echo "[STEP] make timeseries (R)"
run_r "${PROJECT_DIR}/r/make_timeseries.R"

# 4) HTML 렌더
echo "[STEP] render HTML report"
run_py "${PROJECT_DIR}/pipeline/render_report_html.py"


# ---- 배포 ----
echo "[STEP] deploy to Cloudflare Pages (project=${CF_PROJECT:-envdata}, branch=${BRANCH:-production})"
wrangler pages deploy ./public --project-name envdata --branch main
echo "[OK] done at $(date '+%F %T')"

```


---

## tools/plot_bbox.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot East Asia BBOX with a simple background map (land/ocean/coast/borders)
Usage:
  python tools/plot_bbox.py --bbox 116.8,29.6,139.2,45.2 --out public/tiles/debug_bbox.png
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def parse_bbox(s):
    a = [float(x) for x in s.split(",")]
    if len(a) != 4: raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", default="116.8,29.6,139.2,45.2")
    p.add_argument("--out",  default="public/tiles/debug_bbox.png")
    p.add_argument("--margin", type=float, default=1.0, help="degree pad around bbox")
    args = p.parse_args()

    lon_min, lat_min, lon_max, lat_max = parse_bbox(args.bbox)

    # 대표 도시 (필요시 추가/수정)
    cities = [
        ("Seoul",    37.5665, 126.9780),
        ("Busan",    35.1796, 129.0756),
        ("Daejeon",  36.3504, 127.3845),
        ("Gwangju",  35.1595, 126.8526),
        ("Daegu",    35.8714, 128.6014),
        ("Shanghai", 31.2304, 121.4737),
        ("Qingdao",  36.0671, 120.3826),
        ("Tianjin",  39.3434, 117.3616),
        ("Fukuoka",  33.5904, 130.4017),
        ("Osaka",    34.6937, 135.5023),
        ("Hiroshima",34.3853, 132.4553),
    ]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    ax  = plt.axes(projection=proj)
    ax.set_extent(
        [lon_min - args.margin, lon_max + args.margin,
         lat_min - args.margin, lat_max + args.margin],
        crs=proj
    )

    # 배경 지도 피처
    ax.add_feature(cfeature.OCEAN,   facecolor="aliceblue")
    ax.add_feature(cfeature.LAND,    facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linestyle="--", linewidth=0.4)

    # BBOX
    rect = Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                     edgecolor="red", facecolor="none", linewidth=2,
                     transform=proj)
    ax.add_patch(rect)

    # 도시
    for name, lat, lon in cities:
        ax.plot(lon, lat, 'o', ms=4, transform=proj)
        ax.text(lon+0.2, lat+0.2, name, fontsize=8, transform=proj)

    ax.set_title(f"REGION_BBOX: {lon_min}–{lon_max}°E, {lat_min}–{lat_max}°N (background map)")
    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

```
