#!/usr/bin/env python3
"""Preprocess web_data_latest.nc into web-ready assets."""

import h5py
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime, timedelta
from scipy.spatial import KDTree

NC_FILE = 'web_data_latest.nc'
OUT_DIR = 'data'
MAP_DIR = os.path.join(OUT_DIR, 'map')

os.makedirs(MAP_DIR, exist_ok=True)

SITE_NAMES = [
    'Playas de Tijuana',
    'Imperial Beach Pier',
    'Silver Strand',
    'Coronado Avenida Lunar',
]

# Color ramp keyed on log10(dye): [log10_val, R, G, B, A]
# Palette: transparent → yellow → orange → purple → dark purple
COLOR_RAMP = np.array([
    [-8.0, 255, 220,   0,   0],   # fully transparent
    [-6.0, 255, 220,   0,  20],   # barely visible (below low threshold)
    [-5.0, 255, 210,   0, 110],   # visible yellow at low/medium boundary (0.001%)
    [-4.0, 255, 160,   0, 185],   # yellow-orange
    [-3.0, 255,  80,   0, 225],   # orange at medium/high boundary (0.1%)
    [-1.3,  70,   0, 110, 240],   # dark purple starts at ~5%
    [ 0.0,  40,   0,  70, 255],   # darkest purple (high)
], dtype=float)


def l10_to_rgba(l10_frame: np.ndarray) -> np.ndarray:
    """Convert a 2D log10-dye array to RGBA uint8."""
    h, w = l10_frame.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = ~np.isnan(l10_frame)
    v = np.clip(l10_frame[valid], COLOR_RAMP[0, 0], COLOR_RAMP[-1, 0])
    for ch in range(4):
        rgba[valid, ch] = np.interp(v, COLOR_RAMP[:, 0], COLOR_RAMP[:, ch + 1]).astype(np.uint8)
    return rgba


print("Loading NetCDF…")
with h5py.File(NC_FILE, 'r') as f:
    raw_time     = f['time'][:]
    map_lat      = f['map_lat'][:]          # (1141, 486)
    map_lon      = f['map_lon'][:]
    map_l10      = f['map_l10_dye_tot'][:]  # (121, 1141, 486)
    shore_lat    = f['shoreline_lat'][:]    # (1060,)
    shore_lon    = f['shoreline_lon'][:]
    shore_risk   = f['shoreline_risk'][:]   # (121, 1060)
    sites_lat    = f['sites_lat'][:]        # (4,)
    sites_lon    = f['sites_lon'][:]
    sites_risk   = f['sites_risk'][:]       # (121, 4)
    sites_dye    = f['sites_dye_tot'][:]    # (121, 4)
    sites_l10    = f['sites_l10_dye_tot'][:] # (121, 4)
    thresh       = f['thresh_holds'][:]     # [-5, -3]

base = datetime(1999, 1, 1)
timestamps = [(base + timedelta(days=float(t))).strftime('%Y-%m-%dT%H:%M:%S') for t in raw_time]

# ── Map frames ───────────────────────────────────────────────────────────────
# The model uses a curvilinear grid (rows/cols are NOT aligned with lat/lon).
# The northern edge is shifted ~0.06° west relative to the southern edge, so
# naively stretching the data array into a rectangular bounding box would place
# ocean pixels over land.  Fix: re-project onto a regular lat/lon raster via a
# pre-computed KD-tree nearest-neighbour lookup.

print(f"Generating {len(raw_time)} map frames…")

# Bounding box for the output image
bounds = {
    'south': float(map_lat.min()),
    'north': float(map_lat.max()),
    'west':  float(map_lon.min()),
    'east':  float(map_lon.max()),
}

# Output raster size (~300 px wide)
OUT_W = 300
dy = bounds['north'] - bounds['south']
dx = bounds['east']  - bounds['west']
OUT_H = round(OUT_W * dy / dx)

# Build KD-tree from all curvilinear (lat, lon) grid points
print("  Building KD-tree for re-projection…")
flat_lat = map_lat.ravel()      # (1141×486,)
flat_lon = map_lon.ravel()
tree = KDTree(np.column_stack([flat_lat, flat_lon]))

# Regular output grid — rows run N→S so row 0 is the northern edge
lat_reg = np.linspace(bounds['north'], bounds['south'], OUT_H)  # N→S
lon_reg = np.linspace(bounds['west'],  bounds['east'],  OUT_W)
lon_out, lat_out = np.meshgrid(lon_reg, lat_reg)
query_pts = np.column_stack([lat_out.ravel(), lon_out.ravel()])

dist, src_idx = tree.query(query_pts, k=1, workers=-1)
dist     = dist.reshape(OUT_H, OUT_W)
src_idx  = src_idx.reshape(OUT_H, OUT_W)

# Mask output pixels that are further than ~1.5 grid-cell diagonals from any
# model point — these are land / outside the domain.
# Approximate grid-cell diagonal ≈ 0.0006° → threshold ~ 0.001°
DIST_THRESH = 0.001
land_mask = dist > DIST_THRESH   # True = land / outside model domain

print(f"  Re-projection map ready. Land coverage: {land_mask.mean()*100:.1f}%")

for i in range(len(raw_time)):
    flat_data = map_l10[i].ravel()          # (554526,)
    out_data  = flat_data[src_idx]          # (OUT_H, OUT_W)
    out_data[land_mask] = np.nan            # blank out land
    rgba = l10_to_rgba(out_data)
    img  = Image.fromarray(rgba, 'RGBA')
    img.save(os.path.join(MAP_DIR, f'frame_{i:03d}.png'), optimize=True, compress_level=6)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(raw_time)}")

print("Map frames done.")

# ── times.json ───────────────────────────────────────────────────────────────
# Four corners of the curvilinear model grid (angled box), in (lat, lon) order
# suitable for Leaflet polygons.
domain_corners = [
    [float(map_lat[ 0,  0]), float(map_lon[ 0,  0])],
    [float(map_lat[ 0, -1]), float(map_lon[ 0, -1])],
    [float(map_lat[-1, -1]), float(map_lon[-1, -1])],
    [float(map_lat[-1,  0]), float(map_lon[-1,  0])],
]

with open(os.path.join(OUT_DIR, 'times.json'), 'w') as fh:
    json.dump({
        'times': timestamps,
        'bounds': bounds,
        'domain': domain_corners,
        'thresholds': thresh.tolist(),
    }, fh)

# ── sites.json ───────────────────────────────────────────────────────────────
sites_payload = {
    'names': SITE_NAMES,
    'lats':  sites_lat.tolist(),
    'lons':  sites_lon.tolist(),
    'risk':  [[int(r) for r in row] for row in sites_risk],   # (121,4)
    'dye':   [[float(f'{ v:.6e}') for v in row] for row in sites_dye],
    'l10':   [[round(float(v), 3) for v in row] for row in sites_l10],
}
with open(os.path.join(OUT_DIR, 'sites.json'), 'w') as fh:
    json.dump(sites_payload, fh)

# ── shoreline.json ────────────────────────────────────────────────────────────
# risk is 0/1/2 integers → very compact
shore_payload = {
    'lats': [round(float(v), 5) for v in shore_lat],
    'lons': [round(float(v), 5) for v in shore_lon],
    'risk': [[int(r) for r in row] for row in shore_risk],  # (121,1060)
}
with open(os.path.join(OUT_DIR, 'shoreline.json'), 'w') as fh:
    json.dump(shore_payload, fh)

print("JSON files written.")
print(f"\nDone. Output in: {OUT_DIR}/")
print(f"  times.json, sites.json, shoreline.json")
print(f"  map/frame_000.png … frame_{len(raw_time)-1:03d}.png")
