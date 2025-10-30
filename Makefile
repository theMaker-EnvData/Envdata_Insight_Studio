.PHONY: dev run install rdeps
install:
\tsource .venv/bin/activate && uv pip install -r requirements.txt || true
dev:
\tsource .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
run:
\tbash pipeline/run_all.sh 06:00 1
rdeps:
\tr -q -e "install.packages(c('tidyverse','lubridate','sf','terra','gstat','automap','viridis','jsonlite','httr'))"
