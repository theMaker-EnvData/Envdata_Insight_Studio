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
# 5) 타일 시각화 (√-scale + clip@100)
# -----------------------------
draw_colorbar <- function(pal, zmin, zmax, ticks_at, tick_labels,
                          smallplot = c(0.865, 0.945, 0.16, 0.86),  # 폭 ↑ (약 8%)
                          cex_axis = 0.9) {
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)

  # 컬러바 영역(작음)에는 마진을 극도로 줄여야 함
  par(new = TRUE,
      fig = c(smallplot[1], smallplot[2], smallplot[3], smallplot[4]),
      mar = c(0.2, 0.2, 0.2, 1.2),   # ← 핵심: tiny margins
      xaxs = "i", yaxs = "i")

  ny <- 256
  yy <- seq(zmin, zmax, length.out = ny)
  zz <- matrix(rep(yy, each = 2), nrow = 2)

  image(x = c(0, 1), y = yy, z = zz,
        col = pal(ny), axes = FALSE, xlab = "", ylab = "")

  axis(4, at = ticks_at, labels = tick_labels, las = 2, cex.axis = cex_axis, tck = -0.02)
  box()
}


plot_tile <- function(r_pm25_ug, r_u, r_v, tk, outfile_png, cap_ug = 100) {
  bb <- ext(r_pm25_ug)

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

  pal_magma <- colorRampPalette(c(
  "#000004","#180F3D","#3B0F70","#641A80",
  "#8C2981","#B5367A","#DE4968","#F66E5B",
  "#FE9F6D","#FEC574","#FEE08B","#FBFDBF"
))
  pal <- pal_magma

  # 표시용 값: 음수→0, 상한 클리핑(<= cap_ug), √변환
  r_pm25_clip <- clamp(r_pm25_ug, lower = 0, upper = cap_ug, values = TRUE)
  r_pm25_sqrt <- sqrt(r_pm25_clip)

  # √ 스케일에 맞춘 브레이크(색 구간)와 레전드 눈금(원 단위)
  brks  <- seq(0, sqrt(cap_ug), length.out = 65)
  ticks <- c(0, 10, 20, 30, 40, 60, 80, 100)  # 원 단위 눈금
  tick_at <- sqrt(ticks)                      # √축 위치

  # 바람(테이퍼 폴리곤) 준비
  qv <- build_quiver(r_u, r_v, spacing_deg = WIND_SPACING_DEG)

  png(outfile_png, width = 1600, height = 1200, res = 150)
  par(mar = c(0,0,0,0))

  # (1) 본체 레이어: breaks 제거, zlim으로 연속 스케일 지정
  plot(
    r_pm25_sqrt,
    col     = pal(64),
    zlim    = c(0, sqrt(cap_ug)),   # << 여기!
    main    = paste0("PM2.5 (µg/m³) @ ", format(tk, "%Y년 %m월 %d일 %H시")),
    axes    = TRUE,
    legend  = FALSE
  )


  # 베이스맵(경계선 흰색)
  if (!is.null(BMAP)) {
    b <- suppressWarnings(terra::crop(BMAP, terra::ext(r_pm25_sqrt)))
    terra::plot(b, add = TRUE, col = NA, border = "white", lwd = 0.6)
  }

  # 바람 테이퍼 폴리곤
  if (nrow(qv) > 0) {
    polys <- taper_polys(qv)
    if (length(polys) > 0) for (p in polys) polygon(p$x, p$y, border = NA, col = rgb(1,1,1,1))
  }

  # 도시 마커 + 라벨(현행 유지)
  points(cities$lon, cities$lat, pch = 21, bg = "black", col = "white", lwd = 1.4, cex = 1.1)
  .xy_off <- function(px = 1.5) { u<-par("usr"); pin<-par("pin"); c((u[2]-u[1])/(pin[1]*72)*px, (u[4]-u[3])/(pin[2]*72)*px) }
  halo_text <- function(x,y,labels,col="white",bg="black",cex=0.9,px=1,...) {
    off <- .xy_off(px); ang <- seq(0,2*pi,length.out=9)[-9]
    for (a in ang) text(x+off[1]*cos(a), y+off[1]*sin(a), labels, col=bg, cex=cex, xpd=NA, pos=4, ...)
    text(x,y,labels,col=col,cex=cex,xpd=NA,pos=4,...)
  }
  halo_text(cities$lon, cities$lat, labels = cities$kr_name)

  # (2) 범례: breaks 제거, zlim과 커스텀 눈금만 지정
  # √-스케일 범위
  zmin <- 0
  zmax <- sqrt(cap_ug)

  # 눈금 위치(√단위)와 라벨(원 단위)
  ticks_num <- c(0, 10, 20, 30, 40, 60, 80, 100)   # 위치 계산용 숫자
  ticks_at  <- sqrt(ticks_num)                      # √스케일 위치
  tick_lab  <- c("0","10","20","30","40","60","80", paste0(cap_ug, "+"))  # 맨 위만 "100+"

  draw_colorbar(
    pal         = pal,
    zmin        = zmin,
    zmax        = zmax,
    ticks_at    = ticks_at,
    tick_labels = tick_lab,                         # ← 여기만 문자 라벨
    smallplot   = c(0.88, 0.93, 0.15, 0.85),
    cex_axis    = 0.9 
  )

#  mtext(paste0("> ", cap_ug, " µg/m³ 포화"), side = 4, line = 3.3, cex = 0.8)



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


# Pass B: visualization only
for (i in seq_along(times_kst)) {
  tk <- times_kst[i]
  klabel <- fmt_kst(tk)
  out_png <- file.path(OUT_TILE_DIR, cutoff_day_str, paste0("pm25_wind_", klabel, "KST.png"))

  r_pm25_krg <- r_pm25_krg_list[[i]]
  r_u_krg    <- r_u_krg_list[[i]]
  r_v_krg    <- r_v_krg_list[[i]]

  if (!is.null(r_pm25_krg) && !is.null(r_u_krg) && !is.null(r_v_krg)) {
    plot_tile(r_pm25_krg, r_u_krg, r_v_krg, tk, out_png, cap_ug = 100)
    message("  -> saved tile: ", out_png)
    tile_paths <- c(tile_paths, out_png)
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
#    mont <- image_montage(imgs, tile = "3x3", geometry = "1400x1200")
    out_mont <- file.path(OUT_TILE_DIR, cutoff_day_str,
                          paste0("pm25_wind_", cutoff_day_str, "_KST_montage.png"))

    # 3장씩 잘라 행 만들기
    rows <- split(imgs, ceiling(seq_along(imgs)/3))

    # 가로로 붙여 각 행 생성, 그다음 행들을 세로로 쌓기
    row_imgs <- lapply(rows, function(im) image_append(image_join(im)))
    mont <- image_append(image_join(row_imgs), stack = TRUE)

    image_write(mont, out_mont, format = 'png')
    message("saved montage: ", out_mont)
  }
}

message("[done]")
