# ==============================================================================
# Autor: Teilprojekt Naturgefahren 
# Projekt: DeepAlpine
# Modul: Instanziierung der Raster-Makrostruktur (build_template_grid.py)
# Beschreibung: Generiert syntaktisch leere Referenz-Arrays (500m Rasterweite) 
#               zur einheitlichen Ko-Registrierung von ERA5 und 
#               Geomorphologie-Derivaten.
# ==============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

# Pfade & Parameter
# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")

# Eingangsdaten der Rutschungen zur Bestimmung der geografischen Ausdehnung
csv_path = base_dir / "data" / "raw" / "vectors" / "landslides_beta.csv"

# Ausgabeverzeichnis für zwischengespeicherte Raster
out_dir = base_dir / "data" / "intermediate" / "rasters" / "500"
out_dir.mkdir(parents=True, exist_ok=True)

out_template = out_dir / "template_grid_500m_epsg25832.tif"

# Ziel-Projektion (UTM / ETRS89) und Auflösung
target_crs = "EPSG:25832"
pixel_size = 500 # 500 Meter Maßstabsebene für Downscaling

# 1. Punktdaten laden und auf Ziel-KRS projizieren
df = pd.read_csv(csv_path)

if not {"Latitude", "Longitude"}.issubset(df.columns):
    raise ValueError("Attributspalten (Latitude/Longitude) fehlen in der CSV-Quelldatei.")

# Erstellung des Geodatenrahmens über sphärische GPS-Daten (WGS84)
gdf_wgs = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326"
)

print(f"Quell-Referenzsystem: {gdf_wgs.crs}")
print(f"Obergrenzen WGS84 (Bounding Box): {gdf_wgs.total_bounds}")

# Transformation ins UTM-Koordinatensystem für kartesische Flächendistanzen
gdf_utm = gdf_wgs.to_crs(target_crs)
print(f"Ziel-Referenzsystem (Metrisch): {gdf_utm.crs}")
print(f"Obergrenzen UTM (Bounding Box): {gdf_utm.total_bounds}")

minx, miny, maxx, maxy = gdf_utm.total_bounds

# Aufbringung eines topologischen Buffers (+20 km) zur Makro-Abdeckung 
buffer = 20000
minx -= buffer
miny -= buffer
maxx += buffer
maxy += buffer

# 2. Raster-Dimensionen berechnen
# Ableitung der Matrix-Zellstruktur passend zum 500m-Rastermaß
width = int(np.ceil((maxx - minx) / pixel_size))
height = int(np.ceil((maxy - miny) / pixel_size))

print(f"Abgeleitete Array-Matrix: Breite={width}, Höhe={height}, Zellweite={pixel_size}m")
print(f"Puffer-Zonen Extent (UTM): MinX={minx}, MinY={miny}, MaxX={maxx}, MaxY={maxy}")

# Deklaration der affinen Transformation (Ursprungsreferenz 'Oben-Links')
transform = from_origin(minx, maxy, pixel_size, pixel_size)

# 3. Das Template GeoTIFF erstellen
# Aggregation zu TIF-Metadaten für nachfolgende Convolution Operationen
meta = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": -9999.0,
    "width": width,
    "height": height,
    "count": 1,
    "crs": target_crs,
    "transform": transform,
}

# Aufbau der leeren NoData-Tensorfläche
data = np.full((height, width), -9999.0, dtype="float32")

with rasterio.open(out_template, "w", **meta) as dst:
    dst.write(data, 1)

print(f"\nStruktur-Array (Template Grid) fehlerfrei kompiliert unter: {out_template}")
