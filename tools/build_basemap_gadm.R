#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(geodata)  # GADM downloader (returns SpatVector in current setup)
  library(terra)    # vector/raster + IO
})

# 경로 하드코딩
PROJECT_ROOT <- "/opt/EnvData_Insight_Studio"
BASEMAP_DIR  <- file.path(PROJECT_ROOT, "data/basemap")
dir.create(BASEMAP_DIR, recursive = TRUE, showWarnings = FALSE)

# 저장 파일명
OUT_FULL_GPKG <- file.path(BASEMAP_DIR, "gadm_l0_kr_cn_jp_ru_full.gpkg")
OUT_SIMP_GPKG <- file.path(BASEMAP_DIR, "gadm_l0_kr_cn_jp_ru_simplified.gpkg")

# 대상 국가 (ISO3)
countries <- c("KOR", "PRK", "CHN", "JPN", "RUS")

message("[basemap] downloading GADM L0: ", paste(countries, collapse = ", "))

# geodata::gadm() → SpatVector (terra)
vecs <- lapply(countries, function(cc) {
  v <- geodata::gadm(country = cc, level = 0, path = BASEMAP_DIR, version = "4.1")
  # 좌표계가 4326이 아닐 가능성 방지
  terra::project(v, "EPSG:4326")
})

# 병합 (속성 최소화)
g_all <- do.call(rbind, vecs)

# 속성 컬럼 정리 (국가명만 보존; 없으면 GID_0 보존)
keep <- intersect(c("NAME_0","GID_0","COUNTRY","CNTRY_NAME"), names(g_all))
if (length(keep) == 0) {
  # 없는 경우 대비: GID_0가 있으면 복사, 둘 다 없으면 더미
  if ("GID_0" %in% names(g_all)) {
    g_all$NAME_0 <- g_all$GID_0
  } else {
    g_all$NAME_0 <- NA_character_
  }
  keep <- "NAME_0"
}
g_all <- g_all[, keep, drop = FALSE]

# 원본 저장 (풀 해상도)
message("[basemap] writing full gpkg: ", OUT_FULL_GPKG)
if (file.exists(OUT_FULL_GPKG)) file.remove(OUT_FULL_GPKG)
terra::writeVector(g_all, OUT_FULL_GPKG, filetype = "GPKG", overwrite = TRUE)

# 지오메트리 단순화 (렌더링 성능 개선)
message("[basemap] simplifying geometry")
# tolerance는 도 단위(약 0.01 ~ 0.05 권장)
g_simp <- terra::simplifyGeom(g_all, tolerance = 0.01, preserveTopology = TRUE)

message("[basemap] writing simplified gpkg: ", OUT_SIMP_GPKG)
if (file.exists(OUT_SIMP_GPKG)) file.remove(OUT_SIMP_GPKG)
terra::writeVector(g_simp, OUT_SIMP_GPKG, filetype = "GPKG", overwrite = TRUE)

message("[basemap] done")
