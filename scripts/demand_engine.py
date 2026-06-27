import json
import os
import numpy as np
from typing import Dict, Tuple
import rasterio
from rasterio.mask import mask as raster_mask
import geopandas as gpd
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from shapely.geometry import box, mapping
import pyproj

CONFIG_DIR = "config"
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"

WEIGHTS_PATH = os.path.join(CONFIG_DIR, "weights.json")


def _load_weights() -> Dict:
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_geometry(bbox: list, raster_path: str) -> dict:
    """
        Returns a geometry in the raster's native CRS.
        bbox = [south, west, north, east] in WGS84 decimal degrees.
        """
    south, west, north, east = bbox
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if raster_crs.to_epsg() == 4326:
        # Already WGS84 — use bbox directly
        return mapping(box(west, south, east, north))

    # Reproject bbox bounds from WGS84 to raster CRS
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", raster_crs.to_string(), always_xy=True
    )
    x_min, y_min = transformer.transform(west, south)
    x_max, y_max = transformer.transform(east, north)
    return mapping(box(x_min, y_min, x_max, y_max))


def _compute_ndvi_density(
    b04_path: str, b08_path: str, bbox: list, threshold: float = 0.2
) -> float:
    geom = _bbox_geometry(bbox, b08_path)  # ← use raster-aware version
    try:
        with rasterio.open(b08_path) as nir_src:
            nir_data, _ = raster_mask(nir_src, [geom], crop=True, nodata=0)
            nir = nir_data[0].astype(float)
        with rasterio.open(b04_path) as red_src:
            red_data, _ = raster_mask(red_src, [geom], crop=True, nodata=0)
            red = red_data[0].astype(float)

        valid = (nir + red) > 0
        ndvi = np.where(valid, (nir - red) / (nir + red + 1e-10), -1)
        valid_pixels = np.sum(valid)
        if valid_pixels == 0:
            return 0.0
        above_threshold = np.sum((ndvi > threshold) & valid)
        return float(above_threshold / valid_pixels)
    except Exception as e:
        print(f"[demand_engine] NDVI computation error: {e}")
        return 0.3


def _compute_viirs_proxy(viirs_path: str, bbox: list) -> float:
    geom = _bbox_geometry(bbox, viirs_path)  # ← fixed
    try:
        with rasterio.open(viirs_path) as src:
            data, _ = raster_mask(src, [geom], crop=True, nodata=0)
            arr = data[0].astype(float)
            valid = arr > 0
            if not np.any(valid):
                return 0.1
            mean_val = float(np.mean(arr[valid]))
            regional_max = float(np.percentile(arr[valid], 95)) if np.any(valid) else 1.0
            # This normalizes against the 95th percentile of the actual data,
            # so the output is always meaningful regardless of the unit scaling.
            return float(np.clip(mean_val / regional_max, 0.0, 1.0))
    except Exception as e:
        print(f"[demand_engine] VIIRS computation error: {e}")
        return 0.2


def _compute_resource_suitability(
    b04_path: str, b08_path: str, bbox: list, subcategory: str
) -> float:
    NDVI_RANGES = {
        "1.1": (0.20, 1.00),
        "1.2": (0.05, 0.50),
        "1.3": (0.00, 0.20),
        "2.1": (0.00, 0.30),
        "2.2": (0.00, 0.40),
        "2.3": (0.00, 0.40),
        "2.4": (0.00, 0.30),
        "3.1": (0.00, 0.30),
        "3.2": (0.00, 0.50),
        "3.3": (0.00, 0.15),
    }
    lo, hi = NDVI_RANGES.get(subcategory, (0.0, 1.0))
    geom = _bbox_geometry(bbox, b08_path)  # ← fixed
    try:
        with rasterio.open(b08_path) as nir_src:
            nir_data, _ = raster_mask(nir_src, [geom], crop=True, nodata=0)
            nir = nir_data[0].astype(float)
        with rasterio.open(b04_path) as red_src:
            red_data, _ = raster_mask(red_src, [geom], crop=True, nodata=0)
            red = red_data[0].astype(float)

        valid = (nir + red) > 0
        ndvi = np.where(valid, (nir - red) / (nir + red + 1e-10), -1)
        valid_pixels = np.sum(valid)
        if valid_pixels == 0:
            return 0.5
        in_range = np.sum((ndvi >= lo) & (ndvi <= hi) & valid)
        return float(in_range / valid_pixels)
    except Exception as e:
        print(f"[demand_engine] Resource suitability error: {e}")
        return 0.5


def _compute_road_accessibility(
    dem_path: str, osm_path: str, center: list, bbox: list
) -> float:
    try:
        from scipy.ndimage import distance_transform_edt
        gdf = gpd.read_file(osm_path)
        road_types = {"motorway", "trunk", "primary", "secondary", "tertiary", "road", "residential"}
        road_rows = gdf[gdf.get("highway", gdf.get("type", "")).isin(road_types)] if "highway" in gdf.columns else gpd.GeoDataFrame()

        if road_rows.empty:
            return 0.5

        with rasterio.open(dem_path) as dem_src:
            geom = _bbox_geometry(bbox, dem_path)
            dem_data, dem_transform = raster_mask(dem_src, [geom], crop=True, nodata=0)
            dem_arr = dem_data[0].astype(float)
            rows, cols = dem_arr.shape

        road_mask = np.zeros((rows, cols), dtype=bool)
        for _, row in road_rows.iterrows():
            if row.geometry is None:
                continue
            try:
                from rasterio.features import rasterize
                burned = rasterize(
                    [(row.geometry, 1)],
                    out_shape=(rows, cols),
                    transform=dem_transform,
                    fill=0,
                    dtype=np.uint8
                )
                road_mask |= burned.astype(bool)
            except Exception:
                pass

        if not np.any(road_mask):
            return 0.5

        dist = distance_transform_edt(~road_mask)
        max_dist = max(dist.max(), 1)
        accessibility_raw = 1.0 - float(dist.mean() / max_dist)
        return float(np.clip(accessibility_raw, 0.0, 1.0))

    except Exception as e:
        print(f"[demand_engine] Road accessibility error: {e}")
        return 0.5


def _compute_competition_penalty(osm_path: str, subcategory: str) -> float:
    CATEGORY_TAGS = {
        "1.1": {"landuse": ["farmland", "farm", "orchard", "meadow", "greenhouse"]},
        "1.2": {"shop": ["craft", "bakery", "pastry", "clothes"]},
        "1.3": {"industrial": ["workshop", "fabrication"], "craft": ["yes"]},
        "2.1": {"shop": ["hardware", "tools"], "office": ["cleaning", "maintenance"]},
        "2.2": {"amenity": ["taxi", "delivery"], "shop": ["logistics"]},
        "2.3": {"amenity": ["school", "kindergarten", "nursing_home", "clinic"]},
        "2.4": {"office": ["yes", "consulting", "accountant", "lawyer"]},
        "3.1": {"shop": ["yes", "supermarket", "convenience", "general"], "amenity": ["marketplace"]},
        "3.2": {"amenity": ["community_centre", "events_venue", "social_facility"]},
        "3.3": {"tourism": ["camp_site", "viewpoint", "attraction", "information"]},
    }
    BASELINE = {"1.1": 5, "1.2": 3, "1.3": 2, "2.1": 4, "2.2": 3, "2.3": 4,
                "2.4": 3, "3.1": 6, "3.2": 2, "3.3": 2}

    try:
        gdf = gpd.read_file(osm_path)
        tags = CATEGORY_TAGS.get(subcategory, {})
        count = 0
        for col, vals in tags.items():
            if col in gdf.columns:
                count += int(gdf[col].isin(vals).sum())
        baseline = BASELINE.get(subcategory, 4)
        # Invert: more competition = lower score
        penalty = float(np.clip(1.0 - (count / (baseline * 3)), 0.0, 1.0))
        return penalty
    except Exception as e:
        print(f"[demand_engine] Competition error: {e}")
        return 0.6


def compute_demand_score(signals: Dict, weights: Dict) -> int:
    raw = (
        weights["agricultural_density"] * signals["agricultural_density"]
        + weights["population_proxy"]       * signals["population_proxy"]
        + weights["road_accessibility"]     * signals["road_accessibility"]
        + weights["competition_penalty"]    * signals["competition_penalty"]
        + weights["resource_suitability"]   * signals["resource_suitability"]
    )
    return int(np.clip(raw * 100, 0, 100))


def compute_all_demand_scores(community_config: Dict) -> Dict:
    bbox = community_config["bbox"]
    b04 = community_config["sentinel2_b04"]
    b08 = community_config["sentinel2_b08"]
    dem = community_config["dem_tile"]
    viirs = community_config["viirs_tile"]
    osm = community_config["osm_geojson"]
    center = community_config["center"]
    weights_all = _load_weights()

    # Pre-compute shared signals
    print("[demand_engine] Computing NDVI agricultural density...")
    agri_density = _compute_ndvi_density(b04, b08, bbox, threshold=0.08)
    print(f"  → agricultural_density = {agri_density:.3f}")

    print("[demand_engine] Computing VIIRS population proxy...")
    pop_proxy = _compute_viirs_proxy(viirs, bbox)
    print(f"  → population_proxy = {pop_proxy:.3f}")

    print("[demand_engine] Computing road accessibility...")
    road_acc = _compute_road_accessibility(dem, osm, center, bbox)
    print(f"  → road_accessibility = {road_acc:.3f}")

    results = {}
    subcategories = list(weights_all.keys())

    for cat_id in subcategories:
        print(f"[demand_engine] Scoring {cat_id}...")
        cat_weights = weights_all[cat_id]

        comp = _compute_competition_penalty(osm, cat_id)
        rsrc = _compute_resource_suitability(b04, b08, bbox, cat_id)

        signals = {
            "agricultural_density": agri_density,
            "population_proxy": pop_proxy,
            "road_accessibility": road_acc,
            "competition_penalty": comp,
            "resource_suitability": rsrc,
        }
        score = compute_demand_score(signals, cat_weights)
        results[cat_id] = {
            "demand_score": score,
            "signals": signals,
        }
        print(f"  → score = {score}")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, f"demand_scores_{community_config['community_id']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[demand_engine] Saved → {out_path}")
    return results


def load_demand_scores(community_id: str) -> Dict:
    path = os.path.join(OUTPUTS_DIR, f"demand_scores_{community_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pre-computed demand scores found at {path}. Run compute_demand.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)