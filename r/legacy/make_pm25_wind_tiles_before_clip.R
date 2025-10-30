# ================================================
# make_pm25_wind_tiles_hardcoded.R
# ================================================
# 사용법:
#   Rscript make_pm25_wind_tiles_hardcoded.R [YYYYMMDD]
# - 인자를 주면 해당 날짜를 "KST 컷오프일"로 사용
# - 인자를 생략하면 RAW_GEE_DIR 아래 가장 최근(내림차순) 폴더를 컷오프일로 사용
#
# 입력 파일 규칙 (KST 라벨, 컷오프일 폴더 고정):
#   RAW_GEE_DIR/<YYYYMMDD>/gee_cams_<YYYYMMDD_HH00>_native.tif
#   RAW_GEE_DIR/<YYYYMMDD>/gee_wind_<YYYYMMDD_HH00>_native.tif
#   (HH00는 06:00 KST부터 3시간 간격 총 9개)
#
# 출력:
#   - 타일 PNG 9장: OUT_TILE_DIR/<YYYYMMDD>/pm25_wind_<YYYYMMDD_HH00>KST.png
#   - 3x3 콜라주:   OUT_TILE_DIR/<YYYYMMDD>/pm25_wind_<YYYYMMDD>_KST_montage.png
#   - 도시별 시계열 JSON: OUT_JSON_DIR/metrics_<YYYYMMDD>_KST.json
# ================================================

suppressPackageStartupMessages({
  library(terra)
  library(sp)
  library(gstat)
  library(automap)
  library(lubridate)
  library(jsonlite)
  library(magick)
  library(showtext)
 library(sysfonts)
})
font_add(family = "NanumGothic", regular = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
showtext_opts(dpi = 200)
showtext_auto(enable = TRUE)

# -----------------------------
# 0) 경로 설정
# -----------------------------
PROJECT_ROOT <- "/opt/EnvData_Insight_Studio"
RAW_GEE_DIR  <- file.path(PROJECT_ROOT, "data/raw/gee")
OUT_TILE_DIR <- file.path(PROJECT_ROOT, "public/tiles")
OUT_JSON_DIR <- file.path(PROJECT_ROOT, "public")
TZ_KST       <- "Asia/Seoul"

# 베이스맵 경로
BASEMAP_SIMP <- "/opt/EnvData_Insight_Studio/data/basemap/gadm_l0_kr_cn_jp_ru_simplified.gpkg"
BMAP <- try(terra::vect(BASEMAP_SIMP), silent = TRUE)
if (inherits(BMAP, "try-error")) {
  warning("Basemap not found: ", BASEMAP_SIMP, " (skip basemap overlay)")
  BMAP <- NULL
}

# CAMS/WIND 그리드 해상도 (degree)
GRID_DEG_CAMS <- 0.02
GRID_DEG_WIND <- 0.02

# 바람장 화살표 상수
WIND_SPACING_DEG <- 0.5   # 화살표 간격(경위도 기준)
WIND_LEN_FRAC    <- 1.0   # 한 셀(=spacing) 대비 화살표 크기 배수
WIND_MIN_SPD     <- 0.0   # 거의 0인 벡터 제거(m/s)
WIND_WIDTH_FRAC  <- 0.25  # 두께 기준(간격 대비), spd 정규화 후 배수
WIND_SPD_REF_Q   <- 0.75  # 두께 정규화용 기준 속도 (90백분위)

# 도시 좌표 (이름, lon, lat) — 마커 on, 무마스크
cities <- data.frame(
  name = c(
    # KR
    "Seoul","Daejeon","Gwangju","Busan","Daegu",
    # CN
    "Shanghai","Qingdao","Tianjin",
    # JP
    "Fukuoka","Osaka","Hiroshima"
  ),
  lon = c(
    # KR
    126.9780, 127.3845, 126.8526, 129.0756, 128.6014,
    # CN
    121.4737, 120.3826, 117.3616,
    # JP
    130.4017, 135.5023, 132.4553
  ),
  lat = c(
    # KR
    37.5665, 36.3504, 35.1595, 35.1796, 35.8714,
    # CN
    31.2304, 36.0671, 39.3434,
    # JP
    33.5904, 34.6937, 34.3853
  ),
  kr_name = c(
    # KR
    "서울","대전","광주","부산","대구",
    # CN
    "상하이","칭다오","톈진",
    # JP
    "후쿠오카","오사카","히로시마"    
  ),
  stringsAsFactors = FALSE
)

# -----------------------------
# 1) 컷오프일 결정 & 타임스탬프 생성
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && grepl("^\\d{8}$", args[1])) {
  cutoff_day_str <- args[1]
} else {
  # 인자 생략 시: 현재 KST 시각 기준으로 컷오프일 결정
  now_kst <- with_tz(Sys.time(), tzone = TZ_KST)
  cutoff_date <- as_date(now_kst, tz = TZ_KST)
  if (hour(now_kst) >= 6) {
    cutoff_date <- cutoff_date + days(1)
  }
  cutoff_day_str <- format(cutoff_date, "%Y%m%d")
}

dir.create(file.path(OUT_TILE_DIR, cutoff_day_str), recursive = TRUE, showWarnings = FALSE)

# 컷오프 시작: 06:00 KST
start_kst <- force_tz(ymd_hm(paste0(cutoff_day_str, " 06:00")), tzone = TZ_KST)
times_kst <- start_kst + hours(seq(0, by = 3, length.out = 9))
fmt_kst   <- function(t) format(t, "%Y%m%d_%H00", tz = TZ_KST)

cams_path <- function(tk) file.path(RAW_GEE_DIR, cutoff_day_str, paste0("gee_cams_", fmt_kst(tk), "_native.tif"))
wind_path <- function(tk) file.path(RAW_GEE_DIR, cutoff_day_str, paste0("gee_wind_", fmt_kst(tk), "_native.tif"))

# -----------------------------
# 2) 유틸: kriging → IDW only
# -----------------------------
krige_to_grid <- function(src_r, res_deg, varname = "val", sample_cap = 25000L, idw_power = 2.0) {
  # src_r: single-layer SpatRaster (EPSG:4326)
  e <- ext(src_r)
  tgt <- rast(e, resolution = res_deg, crs = crs(src_r))

  vals <- values(src_r, mat = FALSE)
  ok   <- which(is.finite(vals))
  if (length(ok) == 0) stop("No finite points for interpolation: ", varname)

  if (length(ok) > sample_cap) {
    set.seed(42); ok <- sample(ok, sample_cap)
  }
  xy <- terra::xyFromCell(src_r, ok)
  df <- data.frame(lon = xy[,1], lat = xy[,2], val = vals[ok])
  sp::coordinates(df) <- ~ lon + lat
  sp::proj4string(df) <- sp::CRS("+init=epsg:4326")

  xy_t <- terra::xyFromCell(tgt, 1:ncell(tgt))
  nd <- data.frame(lon = xy_t[,1], lat = xy_t[,2])
  sp::coordinates(nd) <- ~ lon + lat
  sp::proj4string(nd) <- sp::CRS("+init=epsg:4326")

  # --- IDW only ---
  g <- gstat::gstat(id = "val", formula = val ~ 1, data = df, set = list(idp = idw_power))
  pred <- predict(g, newdata = nd)

  rr <- tgt
  values(rr) <- pred$val.pred
  return(rr)

  # --- (보관용) Ordinary Kriging 코드는 성능 문제로 비활성 ---
  # fit <- automap::autofitVariogram(val ~ 1, df)
  # pred <- gstat::krige(val ~ 1, locations = df, newdata = nd, model = fit$var_model)
  # rr <- tgt; values(rr) <- pred$var1.pred; return(rr)
}

# -----------------------------
# 3) 유틸: 도시 최근접셀 추출
# -----------------------------
extract_city_vals <- function(r_pm25, r_pm10, r_u, r_v) {
  res <- lapply(1:nrow(cities), function(i){
    p <- vect(data.frame(lon=cities$lon[i], lat=cities$lat[i]),
              geom=c("lon","lat"), crs="EPSG:4326")
    c(pm25=terra::extract(r_pm25, p, ID=FALSE)[1,1],
      pm10=terra::extract(r_pm10, p, ID=FALSE)[1,1],
      u   =terra::extract(r_u,    p, ID=FALSE)[1,1],
      v   =terra::extract(r_v,    p, ID=FALSE)[1,1])
  })
  out <- data.frame(
    city=cities$kr_name, lon=cities$lon, lat=cities$lat,
    pm25=sapply(res, `[[`, 1),
    pm10=sapply(res, `[[`, 2),
    u   =sapply(res, `[[`, 3),
    v   =sapply(res, `[[`, 4),
    stringsAsFactors = FALSE
  )
  out
}

# -----------------------------
# 4) 유틸: 바람장 벡터 처리
# -----------------------------

# 정규 그리드(경위도 spacing)에 맞춰 u,v 샘플 → 일정 길이 화살표 좌표 반환
build_quiver <- function(u_r, v_r, spacing_deg = WIND_SPACING_DEG,
                         len_frac = WIND_LEN_FRAC, min_spd = WIND_MIN_SPD) {
  ex  <- terra::ext(u_r)
  xs  <- seq(ex[1], ex[2], by = spacing_deg)
  ys  <- seq(ex[3], ex[4], by = spacing_deg)
  grd <- expand.grid(lon = xs, lat = ys)
  # bilinear로 u,v 샘플링
  uv  <- cbind(
    terra::extract(u_r, grd[, c("lon", "lat")], method = "bilinear")[,2],
    terra::extract(v_r, grd[, c("lon", "lat")], method = "bilinear")[,2]
  )
  colnames(uv) <- c("u", "v")
  df <- cbind(grd, uv)
  # 속도/정규화
  df$spd <- sqrt(df$u^2 + df$v^2)
  df <- df[is.finite(df$spd) & df$spd >= min_spd, , drop = FALSE]
  if (nrow(df) == 0) return(df)

  baseL <- spacing_deg * len_frac
  refL  <- stats::quantile(df$spd, probs = 0.90, na.rm = TRUE)
  if (!is.finite(refL) || refL <= 0) refL <- max(df$spd, na.rm = TRUE)
  sL <- pmin(df$spd / refL, 1.5)         # 너무 길어지는 것 방지(상한 1.5배)
  L  <- baseL * (0.2 + 0.8 * sL)         # 최소 길이=40%, 속도↑→최대 100%

  df$dx <- (df$u / df$spd) * L
  df$dy <- (df$v / df$spd) * L
  df
}

# 테이퍼(앞쪽 굵음) 폴리곤 그리기용 좌표 생성
# qv: build_quiver() 결과(df: lon,lat,dx,dy,spd)
taper_polys <- function(qv, spacing_deg = WIND_SPACING_DEG,
                        width_frac = WIND_WIDTH_FRAC,
                        spd_ref_q = WIND_SPD_REF_Q) {
  if (nrow(qv) == 0) return(list())

  # 속도 정규화 기준 (너무 두꺼워지는 것 방지)
  ref <- stats::quantile(qv$spd[is.finite(qv$spd)], probs = spd_ref_q, na.rm = TRUE)
  if (!is.finite(ref) || ref <= 0) ref <- max(qv$spd, na.rm = TRUE)
  if (!is.finite(ref) || ref <= 0) return(list())

  baseW <- spacing_deg * width_frac
  polys <- vector("list", nrow(qv))

  for (i in seq_len(nrow(qv))) {
    x0 <- qv$lon[i]; y0 <- qv$lat[i]
    x1 <- x0 + qv$dx[i]; y1 <- y0 + qv$dy[i]
    u  <- qv$dx[i]; v  <- qv$dy[i]
    L  <- sqrt(u*u + v*v)
    if (!is.finite(L) || L == 0) next

    # 단위 방향 및 수직 벡터
    ux <- u / L; uy <- v / L
    px <- -uy;   py <-  ux   # 좌측 수직

    # 두께: 뒤쪽은 얇게(0.35배), 앞쪽은 굵게(1배) — 풍속에 비례
    s   <- min( qv$spd[i] / ref, 1.5 )                 # 상한 클램프
    w_t <- baseW * 0.05 * s
    w_h <- baseW * 0.25 * s

    # 꼬리쪽(얇음) 중심: 시작점 살짝 앞으로(너무 뾰족 방지, 10%L)
    xt <- x0 + ux * (0.10 * L); yt <- y0 + uy * (0.10 * L)
    # 머리쪽(굵음) 중심: 끝점 바로 전(끝단 라벨 가림 방지, 5%L 뒤)
    xh <- x1 - ux * (0.05 * L); yh <- y1 - uy * (0.05 * L)

    # 테이퍼 사다리꼴(뒤 얇음 → 앞 굵음)
    # 순서: tail-left → head-left → head-right → tail-right
    poly_x <- c(xt + px*w_t,  xh + px*w_h,  xh - px*w_h,  xt - px*w_t)
    poly_y <- c(yt + py*w_t,  yh + py*w_h,  yh - py*w_h,  yt - py*w_t)

    polys[[i]] <- list(x = poly_x, y = poly_y)
  }
  polys
}


# -----------------------------
# 5) 유틸: 타일 시각화 (PM2.5 heat + wind arrows)
# -----------------------------
plot_tile <- function(r_pm25_ug, r_u, r_v, tk, outfile_png, zmax_all) {
  bb <- ext(r_pm25_ug)
  # 바람 샘플 포인트 (표시 밀도 조절)
  nx <- 20; ny <- 15
  xs <- seq(bb[1], bb[2], length.out = nx)
  ys <- seq(bb[3], bb[4], length.out = ny)
  grid_pts <- expand.grid(x = xs, y = ys)
  pv <- vect(grid_pts, geom = c("x","y"), crs = "EPSG:4326")
  u <- terra::extract(r_u, pv, ID=FALSE)[,1]
  v <- terra::extract(r_v, pv, ID=FALSE)[,1]

  pal_blue_red <- colorRampPalette(c("#00FFFF","#0099FF","#0044FF","#4400FF",
                            "#6600CC","#880088","#AA0044","#CC0000",
                            "#FF0000","#FF6600","#FFAA00","#FFFF00"))

  # Sequential Orange (어두운 주황 → 밝은 노랑)
  pal_orange <- colorRampPalette(c(
    "#3B0A0A","#5C1E0D","#7D3512","#9D4A12",
    "#B75E13","#CF7316","#E18622","#EDA23B",
    "#F4BE5A","#F9D67F","#FCE8A8","#FFF3C6"
  ))

  # Plasma (보라 → 노랑; 대비 강함)
  pal_plasma <- colorRampPalette(c(
    "#0D0887","#2A0592","#5C01A6","#7E03A8",
    "#9C179E","#B12A90","#CC4778","#E16462",
    "#F2844B","#FCA636","#FCCE25","#F0F921"
  ))

  # Viridis (보라/남청 → 노랑; 색맹 안전)
  pal_viridis <- colorRampPalette(c(
    "#440154","#471365","#482677","#3F4A89",
    "#31688E","#26828E","#1F9E89","#35B779",
    "#6CCE59","#A5DB36","#DCE319","#FDE725"
  ))

  pal <- pal_plasma

  SIZ <- list(
    main = 2.4,          # 제목
    axis = 1.8,          # 축 눈금/축제목
    legend_ticks = 1.6,  # 레전드 눈금
    legend_title = 1.6   # 레전드 제목
  )

  op <- par(
    cex.main = SIZ$main, 
    cex.axis = SIZ$axis, 
    cex.lab  = SIZ$axis,
    mar = c(4,4,5,6.2)   # 오른쪽(레전드) 자리 조금 넉넉히
  )

  png(outfile_png, width=1400, height=1200, res=150)
  par(mar=c(3,3,3,6))

  # 히트맵 범위(이상치 완화하고 싶으면 quantile 사용 가능)
  zlim <- c(0, zmax_all)   # min fixed at 0, max = shared daily max
  # 예: zlim <- as.numeric(quantile(values(r_pm25_ug), c(0.02, 0.98), na.rm=TRUE))
  breaks <- seq(0, zmax_all, length.out = 65)

  ts_str <- format(tk, "%Y-%m-%d %H:%M KST")
  #plot(r_pm25_ug, col = pal(64),
  #     main = paste0("PM2.5 (µg/m³) @ ", format(tk, "%Y년 %m월 %d일 %H시")),
  #     axes = TRUE, zlim = zlim, legend = FALSE)
  plot(r_pm25_ug, col = pal(64), breaks = breaks,
     main = paste0("PM2.5 (µg/m³) @ ", format(tk, "%Y년 %m월 %d일 %H시")),
     axes = TRUE, legend = FALSE)
  
  par(cex.main = SIZ$main, cex.axis = SIZ$axis)

  if (!is.null(BMAP)) {
    g_crop <- suppressWarnings(terra::crop(BMAP, terra::ext(r_pm25_ug)))
    terra::plot(g_crop, col = NA, border = "#4d4d4d", lwd = 0.6, add = TRUE)
    # 배경을 연한 색으로 채우고 싶다면: col = "#f2f2f2" 등으로 교체
  }

  # BMAP: build_basemap.R에서 저장해둔 GADM 폴리곤(SpatVector)이라고 가정
  b <- suppressWarnings(terra::crop(BMAP, terra::ext(r_pm25_ug)))
  # 폴리곤 내부는 채우지 않고, 경계선만 흰색으로
  terra::plot(b, add = TRUE, col = NA, border = "white", lwd = 0.6)

  # 바람 화살표 (스케일)
  #scl <- 0.15 * max(diff(c(bb[1],bb[2]))/nx, diff(c(bb[3],bb[4]))/ny)
  #arrows(grid_pts$x, grid_pts$y, grid_pts$x + u*scl, grid_pts$y + v*scl,
  #       length=0.06, angle=20, col="#222222AA", lwd=1.2)

  # --- wind quiver (0.5° 간격, 일정 길이) ---
  # --- wind quiver: 0.5° 간격 샘플 + 테이퍼 폴리곤 ---
  qv <- build_quiver(r_u_krg, r_v_krg, spacing_deg = WIND_SPACING_DEG)
  if (nrow(qv) > 0) {
    polys <- taper_polys(qv)
    if (length(polys) > 0) {
      # 바탕 윤곽
      # for (p in polys) polygon(p$x, p$y, border = rgb(0,0,0,0), col = NA, lwd = 2)
      # 본색
      for (p in polys) polygon(p$x, p$y, border = NA, col = rgb(1,1,1,1))
    }
  }

  # 도시 (무마스크)
  #points(cities$lon, cities$lat, pch=19, cex=0.6, col="#000000")
  #text(cities$lon, cities$lat, labels=cities$kr_name, pos=4, cex=0.7, col="#000000")

  ## ----- Cities (marker + label halo) -----
  # cities: data.frame(name, lon, lat) already prepared
  marker_cex_base <- 1.1   
  label_cex_base  <- 0.9

  # 픽셀단위 라벨 halo 도우미(추가 패키지 없이)
  .xy_off <- function(px = 1.5) {
    u  <- par("usr"); pin <- par("pin")  # usr=(xmin,xmax,ymin,ymax), pin=plot size (inches)
    dx <- (u[2]-u[1])/(pin[1]*72) * px   # 72 px/inch
    dy <- (u[4]-u[3])/(pin[2]*72) * px
    c(dx, dy)
  }
  halo_text <- function(x, y, labels, col = "black", bg = "white", cex = 1, px = 1.6, pos = 4, ...) {
    off <- .xy_off(px)
    ang <- seq(0, 2*pi, length.out = 9)[-9]
    for (a in ang) text(x + off[1]*cos(a), y + off[1.5]*sin(a), labels,
                        col = bg, cex = cex, xpd = NA, pos = 4, ...)
    text(x, y, labels, col = col, cex = cex, xpd = NA, pos = 4, ...)
  }

  # 마커: 검정 원 + 흰색 테두리 (서울은 내부에 작은 검정 점도 흰 테두리)
  # pch=21은 테두리(col)와 내부(bg)를 분리 지정 가능
  points(cities$lon, cities$lat,
         pch = 21, bg = "black", col = "white", lwd = 1.4,
         cex = marker_cex_base * 1)          # 2배

  # 라벨: 검정 글자 + 흰색 halo
  halo_text(cities$lon, cities$lat,
            labels = cities$kr_name,
            col = "white", bg = "black",
            cex = label_cex_base * 1, px = 1)

  # 색상바
#  plot(r_pm25_max, legend.only = TRUE, col = pal(64), zlim = zlim, axes = TRUE,
#     legend.width = 1.2, legend.shrink = 0.9, axis.args = list(cex.axis = 0.8),
#     smallplot = c(0.88, 0.93, 0.15, 0.85))

  plot(r_pm25_max, legend.only = TRUE, col = pal(64), zlim = c(0, zmax_all), axes = TRUE,
    legend.width = 1.2, legend.shrink = 0.9,
    axis.args  = list(cex.axis = 0.8), 
    smallplot  = c(0.88, 0.93, 0.15, 0.85))
     
  dev.off()
}

# -----------------------------
# 6) 메인 루프
# -----------------------------
message(sprintf("[start] cutoff(KST)=%s  times=%s..%s",
                cutoff_day_str,
                format(times_kst[1], "%Y-%m-%d %H:%M", tz=TZ_KST),
                format(tail(times_kst,1), "%Y-%m-%d %H:%M", tz=TZ_KST)))

records <- list()
tile_paths <- character(0)
r_pm25_krg_list <- vector("list", length(times_kst))
r_pm10_krg_list <- vector("list", length(times_kst))
r_u_krg_list    <- vector("list", length(times_kst))
r_v_krg_list    <- vector("list", length(times_kst))

# Pass A: interpolation + city extraction
for (i in seq_along(times_kst)) {
  tk <- times_kst[i]
  klabel <- fmt_kst(tk)
  cams_f <- cams_path(tk)
  wind_f <- wind_path(tk)

  message(sprintf("[t=%d/9] %s  cams=%s  wind=%s", i,
                  format(tk, "%Y-%m-%d %H:%M KST"), basename(cams_f), basename(wind_f)))

  if (!file.exists(cams_f) || !file.exists(wind_f)) {
    message("  -> missing input(s), skip")
    next
  }

  # ---- 읽기
  r_cams <- rast(cams_f)   # 6 bands (species)
  r_wind <- rast(wind_f)   # 2 bands (u, v)

  # ---- CAMS 밴드: 1=pm2.5, 2=pm10 (kg/m3) → µg/m3
  pm25_raw <- r_cams[[1]]
  pm10_raw <- r_cams[[2]]
  pm25_ug  <- pm25_raw * 1e9
  pm10_ug  <- pm10_raw * 1e9

  # ---- WIND 밴드: 1=u10, 2=v10
  u10 <- r_wind[[1]]
  v10 <- r_wind[[2]]

  # ---- 크리깅 (각 tif 커버의 extent로 그리드 생성; 해상도는 독립 제어)
  #r_pm25_krg <- krige_to_grid(pm25_ug, GRID_DEG_CAMS, varname="pm25")
  #r_pm10_krg <- krige_to_grid(pm10_ug, GRID_DEG_CAMS, varname="pm10")
  #r_u_krg    <- krige_to_grid(u10,     GRID_DEG_WIND, varname="u10")
  #r_v_krg    <- krige_to_grid(v10,     GRID_DEG_WIND, varname="v10")

  tmpl_cams <- rast(ext(r_cams), resolution = GRID_DEG_CAMS, crs = crs(r_cams))
  r_pm25_krg <- terra::resample(pm25_ug, tmpl_cams, method = "bilinear")
  r_pm10_krg <- terra::resample(pm10_ug, tmpl_cams, method = "bilinear")

  tmpl_wind <- rast(ext(r_wind), resolution = GRID_DEG_WIND, crs = crs(r_wind))
  r_u_krg <- terra::resample(u10, tmpl_wind, method = "bilinear")
  r_v_krg <- terra::resample(v10, tmpl_wind, method = "bilinear")

  r_pm25_krg_list[[i]] <- r_pm25_krg
  r_pm10_krg_list[[i]] <- r_pm10_krg
  r_u_krg_list[[i]]    <- r_u_krg
  r_v_krg_list[[i]]    <- r_v_krg

  # ---- 도시 최근접셀 값 추출
  city_df <- extract_city_vals(r_pm25_krg, r_pm10_krg, r_u_krg, r_v_krg)
  city_df$timestamp_kst <- format(tk, "%Y-%m-%d %H:%M", tz=TZ_KST)

  records[[length(records)+1]] <- city_df
}

pm25_max_all <- max(
  vapply(r_pm25_krg_list, function(r) {
    if (is.null(r)) return(NA_real_)
    terra::global(r, "max", na.rm = TRUE)[[1]]
  }, numeric(1)),
  na.rm = TRUE
)
if (!is.finite(pm25_max_all)) pm25_max_all <- 100

# r_pm25_krg -> r_pm25_mix 복사
r_pm25_max <- r_pm25_krg

# 유효(NA 아님) 셀에서 10곳 랜덤 선택 후 최대값 주입
set.seed(42)  # 재현성
vals <- values(r_pm25_max, mat = FALSE)
ok   <- which(is.finite(vals))
n    <- min(10, length(ok))     # 유효 셀이 10개보다 적을 경우 대비
pick <- sample(ok, n)

vals[pick] <- pm25_max_all
values(r_pm25_max) <- vals

#테스트
pm25_max_all <- 50

# Pass B: visualization only
for (i in seq_along(times_kst)) {
  tk <- times_kst[i]
  klabel <- fmt_kst(tk)
  out_png <- file.path(OUT_TILE_DIR, cutoff_day_str, paste0("pm25_wind_", klabel, "KST.png"))

  r_pm25_krg <- r_pm25_krg_list[[i]]
  r_u_krg    <- r_u_krg_list[[i]]
  r_v_krg    <- r_v_krg_list[[i]]

  if (!is.null(r_pm25_krg) && !is.null(r_u_krg) && !is.null(r_v_krg)) {
    plot_tile(r_pm25_krg, r_u_krg, r_v_krg, tk, out_png, pm25_max_all)
    message("  -> saved tile: ", out_png)
    tile_paths <- c(tile_paths, out_png)
    message(tile_paths) # 디버그용 임시
  }
}

# -----------------------------
# 7) 시계열 JSON 저장
# -----------------------------
# records: list of data.frame (도시별 row) x 9시각
if (length(records) > 0) {
  all_df <- do.call(rbind, records)
  # 도시별/시각별 레코드로 정리
  # 출력 스키마: [{timestamp_kst, city, lon, lat, pm25, pm10, u, v}, ...]
  js <- jsonlite::toJSON(all_df, dataframe = "rows", na = "null", auto_unbox = TRUE, pretty = TRUE)
  out_json <- file.path(OUT_JSON_DIR, paste0("metrics_", cutoff_day_str, "_KST.json"))
  write(js, out_json)
  message("saved JSON: ", out_json)
} else {
  message("no records to save (no inputs found?)")
}

# -----------------------------
# 8) 9개 타일 콜라주(3x3)
# -----------------------------
if (length(tile_paths) > 0) {
  # 누락이 있어도 있는 것만 합성
  imgs <- image_read(tile_paths[file.exists(tile_paths)])
  if (length(imgs) > 0) {
    # 3x3 그리드
    mont <- image_montage(imgs, tile = "3x3", geometry = "1400x1200+4+4")
    out_mont <- file.path(OUT_TILE_DIR, cutoff_day_str,
                          paste0("pm25_wind_", cutoff_day_str, "_KST_montage.png"))
    image_write(mont, out_mont, format = 'png')
    message("saved montage: ", out_mont)
  }
}

message("[done]")
