#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot East Asia BBOX with a simple background map (land/ocean/coast/borders)
Usage:
  python tools/plot_bbox.py --bbox 116.8,29.6,139.2,45.2 --out public/tiles/debug_bbox.png
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def parse_bbox(s):
    a = [float(x) for x in s.split(",")]
    if len(a) != 4: raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", default="116.8,29.6,139.2,45.2")
    p.add_argument("--out",  default="public/tiles/debug_bbox.png")
    p.add_argument("--margin", type=float, default=1.0, help="degree pad around bbox")
    args = p.parse_args()

    lon_min, lat_min, lon_max, lat_max = parse_bbox(args.bbox)

    # 대표 도시 (필요시 추가/수정)
    cities = [
        ("Seoul",    37.5665, 126.9780),
        ("Busan",    35.1796, 129.0756),
        ("Daejeon",  36.3504, 127.3845),
        ("Gwangju",  35.1595, 126.8526),
        ("Daegu",    35.8714, 128.6014),
        ("Shanghai", 31.2304, 121.4737),
        ("Qingdao",  36.0671, 120.3826),
        ("Tianjin",  39.3434, 117.3616),
        ("Fukuoka",  33.5904, 130.4017),
        ("Osaka",    34.6937, 135.5023),
        ("Hiroshima",34.3853, 132.4553),
    ]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    ax  = plt.axes(projection=proj)
    ax.set_extent(
        [lon_min - args.margin, lon_max + args.margin,
         lat_min - args.margin, lat_max + args.margin],
        crs=proj
    )

    # 배경 지도 피처
    ax.add_feature(cfeature.OCEAN,   facecolor="aliceblue")
    ax.add_feature(cfeature.LAND,    facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linestyle="--", linewidth=0.4)

    # BBOX
    rect = Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                     edgecolor="red", facecolor="none", linewidth=2,
                     transform=proj)
    ax.add_patch(rect)

    # 도시
    for name, lat, lon in cities:
        ax.plot(lon, lat, 'o', ms=4, transform=proj)
        ax.text(lon+0.2, lat+0.2, name, fontsize=8, transform=proj)

    ax.set_title(f"REGION_BBOX: {lon_min}–{lon_max}°E, {lat_min}–{lat_max}°N (background map)")
    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
