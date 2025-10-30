#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(yaml)
  library(sf)
  library(automap)
  library(gstat)
})

options(stringsAsFactors = FALSE)

`%||%` <- function(a, b) {
  if (is.null(a)) b else a
}

to_posixct_kst <- function(x, tz = "Asia/Seoul") {
  if (inherits(x, "POSIXt")) return(as.POSIXct(x, tz = tz))
  if (is.null(x) || length(x) == 0) return(as.POSIXct(NA, origin = "1970-01-01", tz = tz))
  if (is.numeric(x)) {
    return(as.POSIXct(x, origin = "1970-01-01", tz = tz))
  }
  if (is.character(x)) {
    x2 <- gsub("([+-]\\d{2}):(\\d{2})$", "\\1\\2", x)
    ts <- suppressWarnings(as.POSIXct(x2, format = "%Y-%m-%dT%H:%M:%S%z", tz = tz))
    na_idx <- is.na(ts)
    if (any(na_idx)) {
      ts[na_idx] <- suppressWarnings(as.POSIXct(x2[na_idx], tz = tz))
    }
    return(ts)
  }
  return(suppressWarnings(as.POSIXct(x, tz = tz)))
}

format_iso_kst <- function(ts, tz = "Asia/Seoul") {
  if (is.null(ts) || is.na(ts)) {
    return(NA_character_)
  }
  out <- format(ts, tz = tz, usetz = FALSE, format = "%Y-%m-%dT%H:%M:%S%z")
  sub("([+-]\\d{2})(\\d{2})$", "\\1:\\2", out)
}

parse_args <- function(args) {
  parsed <- list(
    norm_json = NULL,
    bbox = NULL,
    cities_yaml = NULL,
    out = NULL,
    grid_deg = NULL,
    ts = NULL,
    out_date = NULL,
    tol_min = 1440L,
    tol_hours_legacy = NULL,
    tol_h_legacy = NULL
  )
  idx <- 1
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument '%s'", key), call. = FALSE)
    }
    if (key %in% c(
      "--norm_json",
      "--bbox",
      "--cities_yaml",
      "--out",
      "--grid_deg",
      "--ts",
      "--out_date",
      "--tol_min",
      "--tol_hours",
      "--tol_h"
    )) {
      if (idx == length(args)) {
        stop(sprintf("Missing value for %s", key), call. = FALSE)
      }
      value <- args[[idx + 1]]
      idx <- idx + 2
      if (key == "--norm_json") {
        parsed$norm_json <- value
      } else if (key == "--bbox") {
        parsed$bbox <- value
      } else if (key == "--cities_yaml") {
        parsed$cities_yaml <- value
      } else if (key == "--out") {
        parsed$out <- value
      } else if (key == "--grid_deg") {
        parsed$grid_deg <- as.numeric(value)
      } else if (key == "--ts") {
        parsed$ts <- value
      } else if (key == "--out_date") {
        parsed$out_date <- value
      } else if (key == "--tol_min") {
        parsed$tol_min <- as.integer(value)
      } else if (key == "--tol_hours") {
        parsed$tol_hours_legacy <- as.integer(value)
      } else if (key == "--tol_h") {
        parsed$tol_h_legacy <- as.integer(value)
      }
    } else {
      stop(sprintf("Unknown flag %s", key), call. = FALSE)
    }
  }

  required <- c("norm_json", "bbox", "cities_yaml", "out", "grid_deg", "ts")
  missing <- vapply(required, function(field) is.null(parsed[[field]]), logical(1))
  if (any(missing)) {
    stop(
      sprintf(
        "Missing required arguments: %s",
        paste(required[missing], collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (!is.null(parsed$grid_deg)) {
    if (!is.finite(parsed$grid_deg) || parsed$grid_deg <= 0) {
      stop("grid_deg must be positive numeric.", call. = FALSE)
    }
  }

  if (!is.null(parsed$tol_min) && (!is.finite(parsed$tol_min) || parsed$tol_min < 0)) {
    stop("tol_min must be a non-negative integer.", call. = FALSE)
  }

  if (!is.null(parsed$tol_hours_legacy) && (!is.finite(parsed$tol_hours_legacy) || parsed$tol_hours_legacy < 0)) {
    stop("tol_hours must be a non-negative integer.", call. = FALSE)
  }

  if (!is.null(parsed$tol_h_legacy) && (!is.finite(parsed$tol_h_legacy) || parsed$tol_h_legacy < 0)) {
    stop("tol_h must be a non-negative integer.", call. = FALSE)
  }

  if (is.null(parsed$tol_min)) {
    parsed$tol_min <- 1440L
  }

  parsed
}

parse_bbox <- function(value) {
  clean <- gsub("[<>\\s]", "", value)
  parts <- strsplit(clean, ",", fixed = TRUE)[[1]]
  if (length(parts) != 4) {
    stop("Bounding box must contain four comma-separated values.", call. = FALSE)
  }
  nums <- as.numeric(parts)
  if (any(!is.finite(nums))) {
    stop("Bounding box values must be numeric.", call. = FALSE)
  }
  if (!(nums[1] < nums[3] && nums[2] < nums[4])) {
    stop("Bounding box must satisfy lon_min < lon_max and lat_min < lat_max.", call. = FALSE)
  }
  list(lon_min = nums[1], lat_min = nums[2], lon_max = nums[3], lat_max = nums[4])
}

build_axis <- function(min_val, max_val, step) {
  span <- max_val - min_val
  if (!is.finite(span) || span < 0) {
    stop("Invalid bounding box span.", call. = FALSE)
  }
  count <- floor(span / step + 1e-9)
  if (count < 1) {
    return(round(min_val, 6))
  }
  vals <- min_val + step * seq(0, count - 1)
  round(vals, 6)
}

flatten_grid <- function(lat_axis, lon_axis) {
  # Flatten order: lat-major (south→north outer loop, west→east inner loop)
  lat_vec <- rep(lat_axis, each = length(lon_axis))
  lon_vec <- rep(lon_axis, times = length(lat_axis))
  list(lats = as.numeric(lat_vec), lons = as.numeric(lon_vec))
}

load_station_rows <- function(source) {
  raw <- source
  if (is.character(source)) {
    raw <- jsonlite::fromJSON(source, simplifyVector = FALSE)
  }
  if (!is.list(raw)) {
    stop("Malformed AQICN JSON: expected list of stations.", call. = FALSE)
  }
  rows <- lapply(raw, function(entry) {
    if (!is.list(entry) || (!is.null(names(entry)) && is.null(entry$station) && is.null(entry$uid))) {
      return(NULL)
    }
    station <- entry$station %||% list()
    geo <- station$geo %||% list(NA_real_, NA_real_)
    lat <- as.numeric(geo[[1]])
    lon <- as.numeric(geo[[2]])
    data.frame(
      uid = as.character(entry$uid %||% NA_character_),
      name = station$name %||% NA_character_,
      lat = ifelse(is.finite(lat), lat, NA_real_),
      lon = ifelse(is.finite(lon), lon, NA_real_),
      pm25 = if (is.null(entry$pm25)) NA_real_ else as.numeric(entry$pm25),
      time_kst = entry$time_kst %||% NA_character_,
      delta_min = if (is.null(entry$delta_min)) NA_real_ else as.numeric(entry$delta_min),
      status = entry$status %||% NA_character_,
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0) {
    stop("AQICN dataset empty.", call. = FALSE)
  }
  do.call(rbind, rows)
}

normalize_iso_string <- function(value, default_offset = "+09:00") {
  val <- trimws(value %||% "")
  val <- sub("Z$", "+00:00", val, perl = TRUE)
  if (!grepl("T", val, fixed = TRUE)) {
    return(val)
  }
  time_part <- sub(".*T", "", val)
  if (!grepl(":[0-9]{2}:[0-9]{2}", time_part)) {
    val <- sub("([0-9]{2}:[0-9]{2})([+-]|$)", "\\1:00\\2", val, perl = TRUE)
  }
  if (!grepl("(Z|[+-][0-9]{2}:?[0-9]{2})$", val)) {
    val <- paste0(val, default_offset)
  }
  sub("([+-][0-9]{2})(:)([0-9]{2})$", "\\1\\3", val, perl = TRUE)
}

parse_iso_timestamp <- function(value, default_offset = "+09:00", tz = "Asia/Seoul") {
  if (length(value) > 1) {
    prototype <- to_posixct_kst(NA_real_, tz = tz)
    return(vapply(
      value,
      function(elem) parse_iso_timestamp(elem, default_offset, tz),
      prototype
    ))
  }
  if (is.null(value) || is.na(value) || !nzchar(value)) {
    return(to_posixct_kst(NA_real_, tz = tz))
  }
  normalized <- normalize_iso_string(value, default_offset)
  to_posixct_kst(normalized, tz = tz)
}

nearest_value <- function(grid_lats, grid_lons, values, lat, lon) {
  valid <- which(!is.na(values))
  if (length(valid) == 0) {
    return(NA_real_)
  }
  diffs_lat <- grid_lats[valid] - lat
  diffs_lon <- grid_lons[valid] - lon
  dist2 <- diffs_lat * diffs_lat + diffs_lon * diffs_lon
  pick <- valid[which.min(dist2)]
  values[pick]
}

compute_region_means <- function(city_map, city_values) {
  result <- list()
  for (code in names(city_map)) {
    vals <- unname(unlist(city_values[[code]] %||% list()))
    finite_vals <- vals[is.finite(vals)]
    result[[code]] <- if (length(finite_vals) == 0) NA_real_ else mean(finite_vals)
  }
  result
}

infer_out_date_from_filename <- function(path) {
  fname <- basename(path)
  m <- regexec("aqicn_(\\d{8})_.*\\.json$", fname)
  mat <- regmatches(fname, m)
  if (length(mat) == 1 && length(mat[[1]]) == 2) {
    date_str <- mat[[1]][2]
    if (grepl("^\\d{8}$", date_str)) {
      message(sprintf("Inferred out_date=%s from norm_json filename=%s", date_str, fname))
      return(date_str)
    }
  }
  NULL
}

resolve_target_timestamp <- function(ts_input, out_date, norm_json, tz_kst) {
  ts_trim <- trimws(ts_input %||% "")
  hhmm_pattern <- "^[0-9]{2}:[0-9]{2}$"
  iso_base_pattern <- "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"

  if (grepl(hhmm_pattern, ts_trim)) {
    parts <- as.integer(strsplit(ts_trim, ":", fixed = TRUE)[[1]])
    if (length(parts) != 2 || any(is.na(parts)) ||
      parts[1] < 0 || parts[1] > 23 ||
      parts[2] < 0 || parts[2] > 59) {
      stop("Invalid --ts; expected HH:MM with HH in 00-23 and MM in 00-59.", call. = FALSE)
    }
    if (!is.null(out_date)) {
      if (!grepl("^\\d{8}$", out_date)) {
        stop("Invalid --out_date; expected YYYYMMDD.", call. = FALSE)
      }
      resolved_date <- out_date
    } else {
      resolved_date <- infer_out_date_from_filename(norm_json)
      if (is.null(resolved_date)) {
        stop(
          sprintf("Cannot infer date for --ts=%s. Provide --out_date or use RFC3339.", ts_trim),
          call. = FALSE
        )
      }
    }
    iso_candidate <- sprintf(
      "%s-%s-%sT%s:00",
      substr(resolved_date, 1, 4),
      substr(resolved_date, 5, 6),
      substr(resolved_date, 7, 8),
      ts_trim
    )
    parsed <- parse_iso_timestamp(iso_candidate, tz = tz_kst)
    if (is.na(parsed)) {
      stop("Failed to parse constructed timestamp from --ts and --out_date.", call. = FALSE)
    }
    return(list(
      posix = parsed,
      iso = format_iso_kst(parsed, tz_kst),
      out_date = resolved_date,
      source = "hhmm"
    ))
  }

  if (grepl(iso_base_pattern, ts_trim)) {
    parsed <- parse_iso_timestamp(ts_trim, tz = tz_kst)
    if (is.na(parsed)) {
      stop("Failed to parse --ts as RFC3339/ISO timestamp.", call. = FALSE)
    }
    resolved_date <- format(parsed, tz = tz_kst, format = "%Y%m%d")
    return(list(
      posix = parsed,
      iso = format_iso_kst(parsed, tz_kst),
      out_date = resolved_date,
      source = "iso"
    ))
  }

  stop("Invalid --ts format. Use HH:MM or RFC3339/ISO8601.", call. = FALSE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  bbox <- parse_bbox(args$bbox)

  tz_kst <- "Asia/Seoul"

  tol_min <- as.integer(args$tol_min %||% 1440L)
  if (!is.null(args$tol_hours_legacy)) {
    warning("--tol_hours is deprecated; use --tol_min instead.")
    tol_min <- as.integer(args$tol_hours_legacy) * 60L
  }
  if (!is.null(args$tol_h_legacy)) {
    warning("--tol_h is deprecated; use --tol_min instead.")
    tol_min <- as.integer(args$tol_h_legacy) * 60L
  }
  if (!is.finite(tol_min) || tol_min <= 0) {
    stop("tol_min must be a positive integer (minutes).", call. = FALSE)
  }

  target_info <- resolve_target_timestamp(args$ts, args$out_date, args$norm_json, tz_kst)
  if (!is.null(args$out_date) && target_info$source != "hhmm" && args$out_date != target_info$out_date) {
    warning("--out_date ignored because --ts already includes an explicit date.")
  }

  message(
    sprintf(
      "CLI args: norm_json=%s bbox=%s grid_deg=%.3f ts=%s out_date=%s tol_min=%d out=%s",
      args$norm_json,
      args$bbox,
      args$grid_deg,
      args$ts,
      args$out_date %||% "NULL",
      tol_min,
      args$out
    )
  )

  message(sprintf("Resolved target timestamp (KST)=%s", target_info$iso))

  raw_payload <- jsonlite::fromJSON(args$norm_json, simplifyVector = FALSE)
  stations <- load_station_rows(raw_payload)

  target_ts <- to_posixct_kst(target_info$posix, tz = tz_kst)
  target_ts_iso <- format_iso_kst(target_ts, tz_kst)
  window_start <- to_posixct_kst(target_ts - as.difftime(tol_min, units = "mins"), tz = tz_kst)
  window_end <- to_posixct_kst(target_ts, tz = tz_kst)

  message(sprintf(
    "Resolved target_ts=%s, tol_min=%d, window=[%s, %s]",
    format(target_ts, tz = tz_kst, format = "%Y-%m-%dT%H:%M:%S%z"),
    tol_min,
    format(window_start, tz = tz_kst, format = "%Y-%m-%dT%H:%M:%S%z"),
    format(window_end, tz = tz_kst, format = "%Y-%m-%dT%H:%M:%S%z")
  ))

  stations$ts_posix <- to_posixct_kst(stations$time_kst, tz = tz_kst)
  stations$uid <- stations$uid %||% NA_character_
  stations$uid <- ifelse(stations$uid == "NA", NA_character_, stations$uid)
  delta_mins <- abs(as.numeric(difftime(stations$ts_posix, target_ts, units = "mins")))
  valid_idx <- !is.na(stations$ts_posix) &
    is.finite(stations$pm25) &
    is.finite(stations$lat) &
    is.finite(stations$lon) &
    stations$ts_posix >= window_start &
    stations$ts_posix <= window_end &
    delta_mins <= tol_min
  window_rows <- stations[valid_idx, , drop = FALSE]

  if (nrow(window_rows) > 0) {
    window_rows$station_key <- window_rows$uid
    missing_key <- is.na(window_rows$station_key) | window_rows$station_key == ""
    window_rows$station_key[missing_key] <- window_rows$name[missing_key]
    missing_key <- is.na(window_rows$station_key) | window_rows$station_key == ""
    if (any(missing_key)) {
      window_rows$station_key[missing_key] <- sprintf("anon_%s", seq_len(sum(missing_key)))
    }
    groups <- split(window_rows, window_rows$station_key)
    selected_list <- lapply(groups, function(df_group) {
      df_group$delta_mins <- abs(as.numeric(difftime(df_group$ts_posix, target_ts, units = "mins")))
      order_idx <- order(df_group$delta_mins, -as.numeric(df_group$ts_posix))
      df_group[order_idx[1], , drop = FALSE]
    })
    target_rows <- do.call(rbind, selected_list)
    rownames(target_rows) <- NULL
    target_rows$delta_mins <- NULL
    target_rows$station_key <- NULL
  } else {
    target_rows <- window_rows
  }

  coverage_n <- nrow(target_rows)
  if (coverage_n < 3) {
    stop(
      sprintf("Insufficient station coverage (need >=3). Found %d.", coverage_n),
      call. = FALSE
    )
  }

  window_start_iso <- format_iso_kst(window_start, tz_kst)
  window_end_iso <- format_iso_kst(window_end, tz_kst)

  message(sprintf(
    "target_ts_kst=%s, out_date=%s, tol_min=%d, window=[%s, %s], coverage_n=%d",
    target_ts_iso,
    target_info$out_date,
    tol_min,
    window_start_iso,
    window_end_iso,
    coverage_n
  ))
  tol_hours <- tol_min / 60

  force_idw <- coverage_n < 3
  if (force_idw) {
    warning("Low coverage (<3); using IDW fallback")
  }

  lat_axis <- build_axis(bbox$lat_min, bbox$lat_max, args$grid_deg)
  lon_axis <- build_axis(bbox$lon_min, bbox$lon_max, args$grid_deg)
  flatten <- flatten_grid(lat_axis, lon_axis)

  stations_sf <- st_as_sf(
    target_rows,
    coords = c("lon", "lat"),
    crs = 4326,
    agr = "constant"
  )
  stations_merc <- st_transform(stations_sf, 3857)
  target_sf <- st_as_sf(
    data.frame(lon = flatten$lons, lat = flatten$lats),
    coords = c("lon", "lat"),
    crs = 4326,
    agr = "constant"
  )
  target_merc <- st_transform(target_sf, 3857)

  stations_sp <- as(stations_merc, "Spatial")
  target_sp <- as(target_merc, "Spatial")

  bbox_merc <- st_bbox(stations_merc)
  dx <- bbox_merc$xmax - bbox_merc$xmin
  dy <- bbox_merc$ymax - bbox_merc$ymin
  maxdist <- sqrt(dx * dx + dy * dy)
  if (!is.finite(maxdist) || maxdist <= 0) {
    maxdist <- NULL
  }

  status_flag <- "OK"
  method_used <- "autoKrige"
  predictions <- NULL

  if (!force_idw) {
    krige_result <- tryCatch(
      automap::autoKrige(
        pm25 ~ 1,
        input_data = stations_sp,
        new_data = target_sp,
        nmax = 20,
        maxdist = maxdist
      ),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    if (!is.null(krige_result) && !is.null(krige_result$krige_output@data$var1.pred)) {
      predictions <- krige_result$krige_output@data$var1.pred
    } else {
      warning("autoKrige failed or returned no predictions; falling back to IDW")
      status_flag <- "ESTIMATED"
      method_used <- "IDW"
    }
  } else {
    status_flag <- "ESTIMATED"
    method_used <- "IDW"
  }

  if (is.null(predictions)) {
    idw_result <- gstat::idw(
      pm25 ~ 1,
      locations = stations_sp,
      newdata = target_sp,
      idp = 2,
      nmax = 8
    )
    predictions <- idw_result@data$var1.pred
    if (is.null(predictions)) {
      stop("IDW interpolation failed.", call. = FALSE)
    }
  }

  if (identical(method_used, "IDW")) {
    status_flag <- "ESTIMATED"
  }

  predictions <- as.numeric(predictions)
  if (length(predictions) != length(flatten$lats)) {
    stop("Prediction count mismatch with target grid.", call. = FALSE)
  }
  predictions[!is.finite(predictions)] <- NA_real_
  predictions[predictions < 0] <- 0

  message(sprintf("AQICN kriging method=%s", method_used))

  cities <- yaml::read_yaml(args$cities_yaml)
  if (!is.list(cities)) {
    stop("Failed to parse cities YAML.", call. = FALSE)
  }

  city_values <- list()
  for (code in names(cities)) {
    entries <- cities[[code]] %||% list()
    city_values[[code]] <- lapply(entries, function(city) {
      nearest_value(
        flatten$lats,
        flatten$lons,
        predictions,
        as.numeric(city$lat),
        as.numeric(city$lon)
      )
    })
    names(city_values[[code]]) <- vapply(entries, function(city) city$name %||% "", character(1))
  }

  city_pm25 <- list()
  for (code in names(city_values)) {
    for (name in names(city_values[[code]])) {
      city_pm25[[name]] <- city_values[[code]][[name]]
    }
  }

  region_means_raw <- compute_region_means(cities, city_values)
  region_mean <- list(pm2_5 = region_means_raw)

  max_city_name <- NULL
  max_city_value <- NA_real_
  if (length(city_pm25) > 0) {
    all_vals <- unlist(city_pm25, use.names = TRUE)
    valid_vals <- all_vals[is.finite(all_vals)]
    if (length(valid_vals) > 0) {
      max_idx <- which.max(valid_vals)
      max_city_value <- valid_vals[[max_idx]]
      max_city_name <- names(valid_vals)[[max_idx]]
    }
  }

  output <- list(
    grid = list(
      lats = flatten$lats,
      lons = flatten$lons
    ),
    values = list(
      pm2_5 = as.numeric(predictions)
    ),
    region_mean = region_mean,
    city_pm25 = city_pm25,
    max_city = list(
      name = max_city_name,
      value = if (is.null(max_city_name)) NA_real_ else max_city_value
    ),
    meta = list(
      method = method_used,
      status = status_flag,
      bbox = c(bbox$lat_min, bbox$lon_min, bbox$lat_max, bbox$lon_max),
      grid_deg = args$grid_deg,
      timestamp_kst = target_ts_iso,
      tol_hours = tol_hours,
      coverage_n = coverage_n,
      target_ts_kst = target_ts_iso
    )
  )

  dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(
    output,
    args$out,
    auto_unbox = TRUE,
    pretty = TRUE,
    na = "null"
  )

  message(sprintf("AQICN kriging method=%s status=%s coverage_n=%d", method_used, status_flag, coverage_n))
}

tryCatch(
  {
    main()
  },
  error = function(e) {
    message(e$message)
    quit(status = 1)
  }
)
