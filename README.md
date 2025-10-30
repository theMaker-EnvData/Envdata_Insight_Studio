<h1 align="center">Envdata Insight Studio</h1>
<p align="center"><em>East Asia Air Quality — Daily Report Project</em></p>

<p align="center">
  <a href="https://envdata.pages.dev">
    <img alt="Live site" src="https://img.shields.io/badge/live-envdata.pages.dev-2ea44f?logo=icloud&logoColor=white">
  </a>
</p>

---

## What is this?
Envdata Insight Studio는 동아시아 권역의 대기질(ECMWF/CAMS)과 기상(NOAA/GFS) 데이터를 수집·전처리하여  
**일 단위 리포트 HTML**과 **시계열/타일 이미지**를 생성하고, Cloudflare Pages로 **자동 배포**하는 프로젝트입니다.

- 산출물: `public/` (정적 사이트 루트)
- 예시 산출물
  - `public/reports/report_YYYYMMDD_0600.html`
  - `public/tiles/YYYYMMDD/pm25_wind_YYYYMMDD_HH00KST.png`
  - `public/metrics_YYYYMMDD KST.json`

---
