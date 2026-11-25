"""
Raster Processor Module — Drubo Live Benchmark Platform
========================================================
Handles GeoTIFF reading, metadata extraction, and pixel sampling at plot
coordinates.  Works with two backends:
  1. rasterio (preferred, full CRS/affine support)
  2. PIL (Pillow) fallback using simple linear geotransform

Authors: Drubo Paul
"""

import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Try importing yaml for metadata
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ===========================================================================
# Core classes
# ===========================================================================

class RasterInfo:
    """Container for raster metadata."""
    def __init__(self, path, width, height, crs, bounds, nodata, dtype, band_count):
        self.path = path
        self.filename = Path(path).name
        self.width = width
        self.height = height
        self.crs = crs
        self.bounds = bounds  # dict: left, bottom, right, top
        self.nodata = nodata
        self.dtype = dtype
        self.band_count = band_count

    def to_dict(self):
        return {
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "crs": str(self.crs),
            "bounds": self.bounds,
            "nodata": self.nodata,
            "dtype": str(self.dtype),
            "band_count": self.band_count,
        }


# ===========================================================================
# Rasterio backend
# ===========================================================================

def _read_info_rasterio(path):
    with rasterio.open(path) as src:
        bounds = src.bounds
        return RasterInfo(
            path=str(path),
            width=src.width,
            height=src.height,
            crs=str(src.crs) if src.crs else "EPSG:4326 (assumed)",
            bounds={"left": bounds.left, "bottom": bounds.bottom,
                    "right": bounds.right, "top": bounds.top},
            nodata=src.nodata,
            dtype=str(src.dtypes[0]),
            band_count=src.count,
        )


def _sample_rasterio(path, lat, lon, window_size=3):
    """Sample pixel values at (lat, lon) using rasterio.  Returns mean within window."""
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)  # rasterio uses (x, y) → (col, row)
        half = window_size // 2
        row_start = max(0, row - half)
        col_start = max(0, col - half)
        row_end = min(src.height, row + half + 1)
        col_end = min(src.width, col + half + 1)
        window = Window(col_start, row_start,
                        col_end - col_start, row_end - row_start)
        data = src.read(1, window=window).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return float(np.nanmean(data))


# ===========================================================================
# PIL fallback (for simple georeferenced TIFFs using linear extents)
# ===========================================================================

def _read_info_pil(path, extent=None):
    img = Image.open(path)
    width, height = img.size
    if extent is None:
        # Default Dhaka extent from metadata.yaml or hard-coded
        extent = {"lat_min": 23.700, "lat_max": 23.880,
                  "lon_min": 90.340, "lon_max": 90.460}
    return RasterInfo(
        path=str(path),
        width=width,
        height=height,
        crs="EPSG:4326 (assumed from extent)",
        bounds={"left": extent["lon_min"], "bottom": extent["lat_min"],
                "right": extent["lon_max"], "top": extent["lat_max"]},
        nodata=None,
        dtype=str(img.mode),
        band_count=1 if img.mode in ("L", "I", "F", "I;16") else len(img.getbands()),
    )


def _sample_pil(path, lat, lon, extent=None, window_size=3):
    if extent is None:
        extent = {"lat_min": 23.700, "lat_max": 23.880,
                  "lon_min": 90.340, "lon_max": 90.460}
    img = np.array(Image.open(path), dtype=float)
    h, w = img.shape[:2]

    py = int(np.clip(
        (extent["lat_max"] - lat) / (extent["lat_max"] - extent["lat_min"]) * (h - 1),
        0, h - 1))
    px = int(np.clip(
        (lon - extent["lon_min"]) / (extent["lon_max"] - extent["lon_min"]) * (w - 1),
        0, w - 1))

    half = window_size // 2
    y0 = max(0, py - half)
    y1 = min(h, py + half + 1)
    x0 = max(0, px - half)
    x1 = min(w, px + half + 1)

    if img.ndim == 2:
        return float(np.mean(img[y0:y1, x0:x1]))
    else:
        return float(np.mean(img[y0:y1, x0:x1, 0]))


# ===========================================================================
# Public API
# ===========================================================================

def read_raster_info(path, extent=None):
    """Read metadata from a GeoTIFF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    if HAS_RASTERIO:
        return _read_info_rasterio(path)
    elif HAS_PIL:
        return _read_info_pil(path, extent)
    else:
        raise ImportError("Neither rasterio nor Pillow is installed.")


def sample_raster_at_point(path, lat, lon, extent=None, window_size=3):
    """Sample pixel value at a geographic point, with optional window average."""
    path = Path(path)
    if HAS_RASTERIO:
        return _sample_rasterio(path, lat, lon, window_size)
    elif HAS_PIL:
        return _sample_pil(path, lat, lon, extent, window_size)
    else:
        raise ImportError("Neither rasterio nor Pillow is installed.")


def extract_features_from_rasters(df, rasters_dir, extent=None, window_size=3):
    """
    Extract remote-sensing features from all rasters in a directory for every
    plot (row) in the DataFrame.  Expects columns `lat` and `long`.

    Returns a DataFrame with columns: s2_ndvi, canopy_height_m, sar_vh_db
    (and raw band values as bonus columns).
    """
    rasters_dir = Path(rasters_dir)
    result = {
        "s2_red": [], "s2_nir": [],
        "s2_ndvi": [], "canopy_height_m": [], "sar_vh_db": []
    }

    # Identify raster files
    red_path = nir_path = vh_path = ch_path = None
    for f in rasters_dir.glob("*.tif"):
        name_lower = f.stem.lower()
        if "red" in name_lower or "b4" in name_lower:
            red_path = f
        elif "nir" in name_lower or "b8" in name_lower:
            nir_path = f
        elif "vh" in name_lower or "sar" in name_lower or "sentinel1" in name_lower:
            vh_path = f
        elif "canopy" in name_lower or "height" in name_lower or "chm" in name_lower:
            ch_path = f

    found = {
        "Sentinel-2 Red (B4)": red_path,
        "Sentinel-2 NIR (B8)": nir_path,
        "Sentinel-1 VH": vh_path,
        "Canopy Height": ch_path,
    }
    missing = [k for k, v in found.items() if v is None]

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["long"]

        # Sentinel-2 bands
        if red_path and nir_path:
            red_val = sample_raster_at_point(red_path, lat, lon, extent, window_size)
            nir_val = sample_raster_at_point(nir_path, lat, lon, extent, window_size)
            denom = nir_val + red_val
            ndvi = (nir_val - red_val) / denom if denom > 0 else 0.0
        else:
            red_val = nir_val = ndvi = 0.0

        # Canopy height
        if ch_path:
            ch_raw = sample_raster_at_point(ch_path, lat, lon, extent, window_size)
            ch_m = ch_raw / 10.0  # Raw stored as decimetres
        else:
            ch_m = 0.0

        # SAR backscatter
        if vh_path:
            vh_raw = sample_raster_at_point(vh_path, lat, lon, extent, window_size)
            vh_db = (vh_raw - 4000.0) / 100.0  # Reconstruct dB
        else:
            vh_db = 0.0

        result["s2_red"].append(red_val)
        result["s2_nir"].append(nir_val)
        result["s2_ndvi"].append(ndvi)
        result["canopy_height_m"].append(ch_m)
        result["sar_vh_db"].append(vh_db)

    import pandas as pd
    return pd.DataFrame(result), found, missing


def scan_raster_directory(rasters_dir, extent=None):
    """Scan a directory and return metadata for all GeoTIFF files found."""
    rasters_dir = Path(rasters_dir)
    infos = []
    for f in sorted(rasters_dir.glob("*.tif")):
        try:
            info = read_raster_info(f, extent)
            infos.append(info)
        except Exception as e:
            infos.append({"filename": f.name, "error": str(e)})
    return infos


def read_metadata_yaml(rasters_dir):
    """Read metadata.yaml from a rasters directory if it exists."""
    yaml_path = Path(rasters_dir) / "metadata.yaml"
    if not yaml_path.exists():
        return None
    if HAS_YAML:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    else:
        with open(yaml_path) as f:
            return {"raw_text": f.read()}
