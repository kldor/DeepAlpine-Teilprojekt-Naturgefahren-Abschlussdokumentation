# ==============================================================================
# Autor: Teilprojekt Naturgefahren (Kilian Dorn)
# Projekt: DeepAlpine
# Modul: Orografische Downscaling-Korrektur (create_coarse_dem.py)
# Beschreibung: Skript zur Ableitung der ERA5-Gitter Referenzhöhen (Orography)
#               durch Aggregation des 10m-Modells zur Gewährleistung physikalisch 
#               korrekter Lapse-Rate-Anpassungen (Temperatur/Niederschlag).
# ==============================================================================
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
import json

# --- GLOBALE PROJ KORREKTUR: Erzwingt die Nutzung der systemkonsistenten PROJ-Datenbank ---
# Ausführung zwingend vor Import von rasterio/pyproj erforderlich
proj_path = "/Users/kiliandorn/miniconda3/envs/deepalpine_env/share/proj"
if os.path.exists(proj_path):
    os.environ["PROJ_LIB"] = proj_path
    os.environ["PROJ_DATA"] = proj_path
    os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")
dem_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "dem_10m_UTM.tif"
out_json = base_dir / "data" / "intermediate" / "era5_orography.json"

def main():
    print(f"Ladevorgang des Digitalen Geländemodells (DEM): {dem_path}")
    
    # Definition des ERA5-Klimagitters (0.25 Grad physikalische Auflösung)
    # Geometrisches Matching mit den Mittelpunkten der ERA5-Zellen
    
    # Definition der Projektion via PROJ-String zur Umgehung von Datenbankinkonsistenzen
    dst_crs = '+proj=longlat +datum=WGS84 +no_defs' 
    res = 0.25
    
    # Begrenzungen (Bounding Box) und affine Transformation des Zielrasters festlegen
    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=res
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': np.nan
        })

        print(f"Reprojektion und räumliche Aggregation (Resampling, 0.25 Grad)... ({width}x{height})")
        # Einsatz des Resampling.average Algorithmus zur Extrahierung der volumetrischen Zell-Mittelhöhe
        dst_data = np.zeros((height, width), dtype='float32')
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.average
        )

    # Serialisierung in ein assoziatives Dictionary (LAT_LON Indexierung)
    print("Konvertierung der Matrix in eine JSON-kompatible Datenstruktur...")
    orography = {}
    for r in range(height):
        for c in range(width):
            val = dst_data[r, c]
            if np.isfinite(val) and val != 0:
                lon, lat = rasterio.transform.xy(transform, r, c)
                # Rundung zur exakten Übereinstimmung der ERA5-Rasterzentren (0.25, 0.50, 0.75, 0.00)
                lat_r = round(lat * 4) / 4
                lon_r = round(lon * 4) / 4
                key = f"{lat_r:.2f}_{lon_r:.2f}"
                orography[key] = float(val)
                print(f"  Zentrum {key}: {val:.1f} m (M.ü.A.)")

    print(f"Serialisierung abgeschlossen. Export nach: {out_json}")
    with open(out_json, 'w') as f:
        json.dump(orography, f)

if __name__ == "__main__":
    main()
