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
