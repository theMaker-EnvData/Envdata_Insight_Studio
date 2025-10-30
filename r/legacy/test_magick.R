library(magick)

PROJECT_ROOT <- "/root/EnvData_Insight_Studio"
RAW_GEE_DIR  <- file.path(PROJECT_ROOT, "data/raw/gee")
OUT_TILE_DIR <- file.path(PROJECT_ROOT, "public/tiles")
OUT_JSON_DIR <- file.path(PROJECT_ROOT, "public")
TZ_KST       <- "Asia/Seoul"


tile_paths <- c( '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_0600KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_0900KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_1200KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_1500KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_1800KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_2100KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251027_0000KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251027_0300KST.png',
                 '/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251027_0600KST.png'
)

imgs <- image_read(tile_paths[file.exists(tile_paths)])

message(length(imgs))
mont <- image_montage(imgs, geometry = '1400x1200+4+4', tile = "3x3")
out_mont <- file.path("/root/EnvData_Insight_Studio/public/tiles/20251026/pm25_wind_20251026_KST_montage.png")
image_write(mont, out_mont, format = 'png')

#if (length(tile_paths) > 0) {
  # 누락이 있어도 있는 것만 합성
#  imgs <- image_read(tile_paths[file.exists(tile_paths)])
#  if (length(imgs) > 0) {
#    # 3x3 그리드
#    mont <- image_montage(imgs, tile = "3x3", geometry = "1400x1000+4+4")
#    out_mont <- file.path(OUT_TILE_DIR, cutoff_day_str,
#                          paste0("pm25_wind_", cutoff_day_str, "_KST_montage.png"))
#    image_write(mont, out_mont)
#    message("saved montage: ", out_mont)
#  }
#}