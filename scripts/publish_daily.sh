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
