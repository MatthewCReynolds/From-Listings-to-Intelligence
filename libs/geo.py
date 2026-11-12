"""Geospatial features from OpenStreetMap points of interest (POIs).

Moved from notebooks/real/phase_01_market_rent/04_geospatial_features.ipynb (originally by Lady Infante
Acosta). The notebook's globals are now keyword arguments whose defaults are the notebook's values.
"""
from pathlib import Path

import numpy as np
import requests

from libs.paths import DATA

SEARCH_RADII = [500, 1000, 1500]  # meters
EARTH_R = 6_371_000               # meters
OSM_DIR = DATA / "osm"
BASE_URL = "https://download.geofabrik.de/north-america/us"
AREAS = {
    "chicago": {"file": "illinois-latest.osm.pbf", "bbox": [-88.5, 41.4, -87.2, 42.5]},
    "dfw": {"file": "texas-latest.osm.pbf", "bbox": [-98.0, 32.2, -96.3, 33.4]},
    "dc": {"file": "district-of-columbia-latest.osm.pbf", "bbox": [-77.6, 38.6, -76.8, 39.2]},
}
CATEGORY_TAGS = {
    "hospitals": {"amenity": ["hospital"]},
    "pharmacies": {"amenity": ["pharmacy"]},
    "supermarkets": {"shop": ["supermarket"]},
    "convenience_stores": {"shop": ["convenience"]},
    "restaurants": {"amenity": ["restaurant", "fast_food"]},
    "cafes": {"amenity": ["cafe"]},
    "banks": {"amenity": ["bank"]},
    "atms": {"amenity": ["atm"]},
    "parks": {"leisure": ["park", "playground"]},
    "public_transport": {"highway": ["bus_stop"], "railway": ["station", "halt"], "amenity": ["bus_station"]},
}


def download_pbf(filename, osm_dir=OSM_DIR, base_url=BASE_URL):
    """Download one OpenStreetMap .pbf extract from Geofabrik, skipping files already on disk."""
    # Define the local file path
    osm_dir = Path(osm_dir)
    osm_dir.mkdir(parents=True, exist_ok=True)
    path = osm_dir / filename

    # Skip download if the file already exists
    if path.exists():
        print(f"File already exists: {filename}")
        return path

    # Build the download URL
    url = f"{base_url}/{filename}"

    print(f"Downloading {url} ...")

    # Use a temporary file to avoid keeping incomplete downloads
    tmp = path.with_suffix(".part")

    with requests.get(url, stream=True, timeout=60) as response:

        # Raise an error if the download request fails
        response.raise_for_status()

        # Download the file in chunks
        with open(tmp, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    # Rename the temporary file after a successful download
    tmp.rename(path)

    print(f"Saved to: {path}")

    return path


def distance(lat1, lon1, lat2, lon2, earth_radius=EARTH_R):
    """Haversine distance in meters; works on scalars or NumPy arrays."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    return earth_radius * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def generate_features(lat, lon, category_pois, category_tags=CATEGORY_TAGS, search_radii=SEARCH_RADII):
    """POI counts within each search radius and the distance to the nearest POI, per category."""
    features = {}

    for category in category_tags:
        poi_coords = category_pois[category]

        if len(poi_coords) > 0:
            distances = distance(
                lat,
                lon,
                poi_coords[:, 0],
                poi_coords[:, 1]
            )
        else:
            distances = np.array([])

        for radius in search_radii:
            features[f"{category}_{radius}m"] = int(
                np.sum(distances <= radius)
            )

        features[f"nearest_{category}_m"] = (
            distances.min() if len(distances) > 0 else np.nan
        )

    return features
