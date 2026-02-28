# ==============================================================================
# Autor: Teilprojekt Naturgefahren
# Projekt: DeepAlpine
# Modul: Terrain-Prozessierung (process_terrain_features.py)
# Beschreibung: Generiert hydrologische und geomorphologische Raster-Features
#               (TWI, Profile Curvature, Plan Curvature, Aspect) aus dem
#               digitalen Höhenmodell (DEM).
# ==============================================================================
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.ndimage import convolve
from pathlib import Path
import os

# Paths
# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")
raw_dem_path = base_dir / "data" / "raw" / "dem" / "output_hh.tif"
slope_ref_path = base_dir / "data" / "raw" / "slope" / "slope_10m_UTM.tif"

output_dir = base_dir / "data" / "intermediate" / "rasters" / "10m"
output_dir.mkdir(parents=True, exist_ok=True)

reprojected_dem_path = output_dir / "dem_10m_UTM.tif"
plan_curv_path = output_dir / "plan_curvature_10m.tif"
profile_curv_path = output_dir / "profile_curvature_10m.tif"
aspect_path = output_dir / "aspect_10m.tif"
twi_path = output_dir / "twi_10m.tif"

def reproject_dem():
    print(f"Transformation des Referenzrasters {raw_dem_path.name} (Koordinatenreferenz EPSG:25832)...")
    
    with rasterio.open(slope_ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_nodata = ref.nodata

    with rasterio.open(raw_dem_path) as src:
        # Geometrisches Matching der Raum- und Auflösungsverhältnisse für kohärente Arrays
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': -9999.0 # Zuweisung eines validierten Null-Wertes (-9999.0) für Vektorlücken
        })

        with rasterio.open(reprojected_dem_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    print(f"Erfolgreiche Speicherung des referenzierten Modells: {reprojected_dem_path}")

def calculate_curvatures():
    print("Numerische Approximation von Oberflächenverkrümmungen (Plan/Profile Curvature)...")
    with rasterio.open(reprojected_dem_path) as src:
        dem = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        res = src.res[0] # assuming square pixels (10m)

    print(f"  DEM-Dimension (Shape): {dem.shape}")
    # Algorithmische Stützung auf Faltungsmatrizen iterativ zu Zevenbergen & Thorne (1987)
    # Kalkulation zweidimensionaler partikularer Differenzen (Neun-Punkt-Fenster)
    
    # Kernel for first derivatives
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * res)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / (8 * res)
    
    # Kernel for second derivatives
    Kxx = np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]]) / (4 * res**2)
    Kyy = np.array([[1, 2, 1], [-2, -4, -2], [1, 2, 1]]) / (4 * res**2)
    Kxy = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * res**2)

    print("  Ausführung der Bildfaltung (Convolution Filter - O(N) Komplexität)... ")
    # Berechnung linearer Terrain-Komponenten (Erste/Zweite Ableitung)
    p = convolve(dem, Kx) # dz/dx
    q = convolve(dem, Ky) # dz/dy
    r = convolve(dem, Kxx) # d2z/dx2
    s = convolve(dem, Kxy) # d2z/dxdy
    t = convolve(dem, Kyy) # d2z/dy2
    
    # Limitierung zur Prävention von Null-Divisions-Instabilitäten
    slope_sq = p**2 + q**2
    slope_sq[slope_sq == 0] = 1e-10
    
    # Profile Curvature (vertical)
    # ProC = -2 * (z_xx * z_x^2 + 2 * z_xy * z_x * z_y + z_yy * z_y^2) / (p^2 + q^2)
    print("  Herleitung der Profile Curvature (Vertikaler Beschleunigungsgradient)...")
    prof_c = -2.0 * (r * p**2 + 2 * s * p * q + t * q**2) / slope_sq
    
    # Plan Curvature (horizontal)
    # PlaC = 2 * (z_xx * z_y^2 - 2 * z_xy * z_x * z_y + z_yy * z_x^2) / (p^2 + q^2)
    print("  Herleitung der Plan Curvature (Horizontaler Divergenzgradient)...")
    plan_c = 2.0 * (r * q**2 - 2 * s * p * q + t * p**2) / slope_sq

    # Aspect (Hangausrichtung)
    # 0 = North, 90 = East, 180 = South, 270 = West
    print("  Bestimmung der azimutalen Hangexposition (Aspect/Trigonometrie)...")
    aspect = np.degrees(np.arctan2(-p, q))
    aspect[aspect < 0] += 360.0

    # Theoretische Kalkulation des Hydrologischen Indices (Topographic Wetness Index)
    # TWI = ln(a / tan(beta))
    # Applikation eines logarithmischen Fluss-Proxys basierend auf lokaler Topographie: (1 + plan_curvature_normalized) / tan(slope)
    # Concave areas (high plan curv) collect water.
    print("  Synthetisierung morphologischer Wassersättigungs-Wahrscheinlichkeiten (TWI)...")
    slope_rad = np.arctan(np.sqrt(slope_sq))
    tan_slope = np.tan(slope_rad)
    tan_slope[tan_slope <= 0] = 0.001 # avoid division by zero
    
    # Normierte Plankrümmung als lokaler Einzugsgebiets-Proxy (konkav = positiv)
    # Plankrümmungswerte sind in der Regel klein: Verstärken sie, damit Flächenfaktor wirken.
    catchment_proxy = np.maximum(0.1, 1.0 + plan_c * 100.0) 
    twi = np.log(catchment_proxy / tan_slope)
    
    # Maskierung undefinierter Terraindaten in Derivat-Rastern
    mask = (dem == nodata)
    prof_c[mask] = -9999.0
    plan_c[mask] = -9999.0
    aspect[mask] = -9999.0
    twi[mask] = -9999.0
    
    # Save outputs
    meta.update(dtype='float32', nodata=-9999.0)
    
    with rasterio.open(profile_curv_path, 'w', **meta) as dst:
        dst.write(prof_c.astype('float32'), 1)
    print(f"Artefakt exportiert (Profile Curvature): {profile_curv_path}")
    
    with rasterio.open(plan_curv_path, 'w', **meta) as dst:
        dst.write(plan_c.astype('float32'), 1)
    print(f"Artefakt exportiert (Plan Curvature): {plan_curv_path}")

    with rasterio.open(aspect_path, 'w', **meta) as dst:
        dst.write(aspect.astype('float32'), 1)
    print(f"Artefakt exportiert (Aspect): {aspect_path}")

    with rasterio.open(twi_path, 'w', **meta) as dst:
        dst.write(twi.astype('float32'), 1)
    print(f"Artefakt exportiert (Topographic Wetness Index): {twi_path}")

if __name__ == "__main__":
    if not reprojected_dem_path.exists():
        reproject_dem()
    else:
        print("Überspringung: Zieldatensatz bereits referenziert vorliegend.")
    
    calculate_curvatures()
    print("\nGeomorphologischer Berechnungsworkflow terminiert.")
