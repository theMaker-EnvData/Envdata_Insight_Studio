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
