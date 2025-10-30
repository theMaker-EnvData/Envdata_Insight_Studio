# deps: jsonlite, dplyr, lubridate, ggplot2, scales, grid
library(jsonlite)
library(dplyr)
library(lubridate)
library(ggplot2)
library(scales)

# --------- 컷오프 규칙(06시)로 대상일 자동 결정 ---------
now_kst <- with_tz(Sys.time(), tzone = "Asia/Seoul")
report_date <- if (hour(now_kst) < 6) as_date(now_kst) else as_date(now_kst + days(1))
ymd_str <- format(report_date, "%Y%m%d")

# --------- 경로(고정) ---------
metrics_path <- file.path("public", sprintf("metrics_%s_KST.json", ymd_str))
out_dir  <- file.path("public", "reports")
out_file <- file.path(out_dir, sprintf("timeseries_%s_0600.png", ymd_str))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --------- 도시 순서(범례 고정) ---------
cities_domestic <- c("서울","대전","광주","부산","대구")
cities_china    <- c("상하이","칭다오","톈진")
cities_japan    <- c("후쿠오카","오사카","히로시마")
city_levels <- c(cities_domestic, cities_china, cities_japan)

# --------- 데이터 ---------
dat_raw <- fromJSON(metrics_path)
message(metrics_path)
dat <- dat_raw %>%
  transmute(
    city = factor(as.character(city), levels = city_levels),
    ts   = ymd_hm(timestamp_kst, tz = "Asia/Seoul"),
    pm25 = as.numeric(pm25)
  ) %>% filter(!is.na(city), !is.na(ts), !is.na(pm25))

# 시간 범위: 06시 ~ 익일 06시
t_start <- ymd_hm(paste0(format(report_date, "%Y-%m-%d"), " 06:00"), tz = "Asia/Seoul")
t_end   <- t_start + hours(24)

# 1) region 순서 고정 + √값 계산 (dat_f 만드는 곳)
dat_f <- dat %>%
  filter(ts >= t_start, ts <= t_end) %>%
  mutate(
    pm25_sqrt = sqrt(pmax(pm25, 0)),
    region = case_when(
      city %in% cities_domestic ~ "국내",
      city %in% cities_china    ~ "중국",
      TRUE                      ~ "일본"
    ),
    region = factor(region, levels = c("국내","중국","일본")) # 범례 순서: 국내→중국→일본
  )

# 2) 국가별로만 확실히 구분되도록 '도시별 색상' 직접 매핑
city_colors <- c(
  # 국내
  "서울"="#D55E00", "대전"="#0072B2", "광주"="#009E73",
  "부산"="#CC79A7", "대구"="#E69F00",
  # 중국
  "상하이"="#1F78B4", "칭다오"="#E31A1C", "톈진"="#33A02C",
  # 일본
  "후쿠오카"="#000000",  # Black
  "오사카"   ="#00BCD4", # Cyan
  "히로시마"="#F0027F"  # Magenta
)

# y축: √값으로 그리되 라벨은 원래값(제곱) 표시
y_max_orig <- max(dat_f$pm25, na.rm = TRUE)
base_breaks <- c(0, 10, 20, 35, 50, 100, 150)    # 40, 75 제외
y_breaks_orig <- base_breaks[base_breaks <= max(100, y_max_orig)]
y_breaks_sqrt <- sqrt(y_breaks_orig)

# x축: 06시부터 3시간 간격 ~ 익일 06시 (오른쪽 여백 0)
x_breaks <- seq(t_start, t_end, by = "3 hours")
x_labels <- paste0(hour(x_breaks), "시")

# 기준선 주석 위치: 오른쪽 여백을 0으로 했으므로, 끝점에서 60초 왼쪽
annot_x <- t_end - seconds(60)
annot_y <- sqrt(35) - 0.3

# --------- 플롯 ---------
# 3) 플롯 블록 (기준선 얇게, 라벨 작게·가깝게, 중국=대시/일본=점선, 색상 매핑 적용)
p <- ggplot(dat_f, aes(x = ts, y = pm25_sqrt, color = city, group = city, linetype = region)) +
  geom_line(linewidth = 0.6, alpha = 0.95) +
  # 기준선: 실선 + 더 얇게(0.2)
  geom_hline(yintercept = sqrt(35), linetype = "solid", linewidth = 0.2, color = "firebrick") +
  # 기준선 라벨: 글씨 더 작게(3.0), 선에 더 바짝(y - 0.12)
  annotate("text", x = annot_x, y = sqrt(35) - 0.12,
           label = "(PM2.5 대기환경기준: 35 μg/m³)",
           hjust = 1, vjust = 1, size = 3.0, color = "firebrick") +
  scale_x_datetime(limits = c(t_start, t_end),
                   breaks = x_breaks, labels = x_labels,
                   expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(breaks = y_breaks_sqrt, labels = y_breaks_orig,
                     expand = expansion(mult = c(0.05, 0.02))) +
  # 도시별 색상 + 범례 순서(도시 레벨 순) 고정
  scale_color_manual(values = city_colors, breaks = city_levels) +
  # 권역별 선모양: 국내=실선, 중국=대시, 일본=점선
  scale_linetype_manual(values = c("국내" = "solid", "중국" = "dashed", "일본" = "dotted")) +
  labs(x = NULL, y = "PM2.5 (μg/m³)", color = NULL, linetype = NULL) +
  guides(
    linetype = guide_legend(order = 1, override.aes = list(color = "grey30", linewidth = 0.9)),
    color    = guide_legend(ncol = 1, order = 2)  # 도시 목록(서울부터) 세로
  ) +
  theme_minimal(base_family = "Apple SD Gothic Neo") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "#bbbbbb", fill = NA, linewidth = 0.4),
    legend.position = "right",
    legend.key.width = unit(14, "pt"),
    legend.text = element_text(size = 9),
    axis.text.x = element_text(margin = margin(t = 4)),
    axis.title.y = element_text(margin = margin(r = 8))
  )



ggsave(out_file, p, width = 1000/96, height = 360/96, dpi = 96, units = "in", bg = "white")
message("[OK] ", out_file)
