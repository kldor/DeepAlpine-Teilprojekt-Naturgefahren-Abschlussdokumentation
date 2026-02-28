# ==============================================================================
# Autor: Teilprojekt Naturgefahren
# Projekt: DeepAlpine
# Modul: Modellvorhersage & Geo-Visualisierung (predict_rf_era5.py)
# Beschreibung: Dieses Skript nutzt das trainierte Random-Forest-Modell, lädt
#               die topografischen Basisdaten sowie die ERA5-Klimadaten für 
#               einen spezifischen Tag, interpoliert diese auf ein 500m-Raster,
#               berechnet die Vorhersagen und speichert die Gefahrenkarte.
# ==============================================================================
import os
import sys

# --- GLOBALE PROJ KORREKTUR: Zwingt die Nutzung der konsistenten PROJ-Datenbank ---
# Dies MUSS vor dem Import von rasterio oder pyproj geschehen
proj_path = "/Users/kiliandorn/miniconda3/envs/deepalpine_env/share/proj"
if os.path.exists(proj_path):
    os.environ["PROJ_LIB"] = proj_path
    os.environ["PROJ_DATA"] = proj_path
    # Verhindert die Nutzung korrupter Systemkontexte in pyproj
    os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import rasterio
import pyproj
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
from rasterio.fill import fillnodata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json

# Konfiguration der Basisverzeichnisse
# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")

# Pfade der Eingangsdaten (Topografie, ERA5, Raster-Template)
slope_path = base_dir / "data" / "raw" / "slope" / "slope_10m_UTM.tif"
nc_single_dir = base_dir / "data" / "raw" / "era5_single" / "nc_singel"
nc_soil_dir = base_dir / "data" / "raw" / "era5_single" / "nc_soil"
template_path = base_dir / "data" / "intermediate" / "rasters" / "500" / "template_grid_500m_epsg25832.tif"
dem_raster_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "dem_10m_UTM.tif"
era5_orog_path = base_dir / "data" / "intermediate" / "era5_orography.json"
plan_curv_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "plan_curvature_10m.tif"
profile_curv_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "profile_curvature_10m.tif"
aspect_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "aspect_10m.tif"
twi_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "twi_10m.tif"

# Fallback-Logik für Pfade, um verschiedene Projektversionen zu unterstützen
if not template_path.exists():
    template_path = base_dir / "data" / "intermediate" / "rasters" / "500" / "template_grid_500m_epsg25832.tif"

models_dir = base_dir / "models"
model_path = models_dir / "rf_era5.joblib"

outputs_dir = base_dir / "outputs" / "rasters" / "era5_predictions"
outputs_dir.mkdir(parents=True, exist_ok=True)

# Auswahl des Zieldatums für die Rutschungsvorhersage
import sys
if len(sys.argv) > 1:
    target_date_str = sys.argv[1]
    print(f"Initialisiere Datum für Modell-Vorhersage: {target_date_str}")
else:
    target_date_str = "2021-05-08"
target_date = pd.to_datetime(target_date_str)

# Konsistenzsicherung: Definition der Zeitfenster (identisch zur Trainingsphase)
WINDOWS = {
    "1d": 1,
    "2d": 2,
    "7d": 7,
    "14d": 14,
    "21d": 21
}

# 1. Trainiertes Modell und Raster-Struktur laden
print(f"Lade Modell: {model_path}")
bundle = joblib.load(model_path)
rf = bundle["model"]
feature_names = bundle["feature_names"]
metrics = bundle.get("metrics", {}) # Load scores if they exist
# Verwendung des optimierten Schwellenwerts aus dem Training (Fallback: 0.5)
model_threshold = bundle.get("best_threshold", 0.5)
print(f"Modell geladen. Verwende trainierten Schwellenwert: {model_threshold:.2f}")

# ERA5 Höhen-Basislinie via JSON laden (für Lapse-Rate-Korrektur)
if era5_orog_path.exists():
    with open(era5_orog_path, 'r') as f:
        era5_orog = json.load(f)
    print(f"ERA5-Höhen-Basislinie für {len(era5_orog)} Zellen geladen.")
else:
    era5_orog = {}
    print("Warnung: ERA5-Höhen-Basislinie nicht gefunden.")
print("Erwartete Features:", feature_names)
print("Modell Metriken:", metrics)

print(f"Lade Slope & Template...")
with rasterio.open(template_path) as tmp:
    H_tpl = tmp.height
    W_tpl = tmp.width
    dst_transform = tmp.transform
    dst_crs = tmp.crs
    template_meta = tmp.meta.copy()

# 2. Resampling der Terrain-Features auf das 500m Vorhersage-Raster
terrain_features = {
    "Slope": slope_path,
    "DEM": dem_raster_path,
    "Plan_Curvature": plan_curv_path,
    "Profile_Curvature": profile_curv_path,
    "Aspect": aspect_path,
    "TWI": twi_path
}

resampled_terrain = {}

for name, path in terrain_features.items():
    print(f"  Resampling {name}...")
    with rasterio.open(path) as src:
        arr = np.zeros((H_tpl, W_tpl), dtype='float32')
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        resampled_terrain[name] = arr
        
        # Spezielle Handhabung der Gültigkeitsmaske (Mask/Validity) basierend auf der Slope-Ebene
        if name == "Slope":
            # Präzise Maskierung: Südtirol-Grenze entspricht validen Datenpixeln
            # Ausschluss von NoData-Werten (-3.4e+38) aus der Hangneigungsberechnung
            valid_slope_mask = (arr > -1e30) & (np.isfinite(arr))
            slope_clean = np.where(valid_slope_mask, arr, np.nan)

dem_resampled = resampled_terrain["DEM"]
feature_rasters = {k: v for k, v in resampled_terrain.items() if k not in ["Slope", "DEM", "Aspect"]}
if "Aspect" in resampled_terrain:
    aspect_rad = np.radians(np.nan_to_num(resampled_terrain["Aspect"], nan=0.0))
    feature_rasters["Aspect_North"] = np.where(valid_slope_mask, np.cos(aspect_rad).astype("float32"), np.nan)
    feature_rasters["Aspect_East"] = np.where(valid_slope_mask, np.sin(aspect_rad).astype("float32"), np.nan)
H, W = H_tpl, W_tpl

print("Terrain-Features erfolgreich resampelt.")
# Konsistenzprüfung: Alle erforderlichen Prädiktoren müssen vorliegen
# Die Hangneigung (Slope) wird für die Interaktionsterme (z.B. Slope_x_Soil) persistiert
# Beibehaltung der bereinigten Hangneigung für algebraische Interaktionen

# Variablen-Mapping zur Harmonisierung der NetCDF-Bezeichner
VAR_MAP = {
    "total_precipitation": "tp",
    "snowfall": "sf",
    "2m_temperature": "t2m",
    "snow_depth": "sd",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
}

def get_var(dataset, long_name):
    if long_name in dataset:
        return long_name
    short = VAR_MAP.get(long_name)
    if short in dataset:
        return short
    return None

# 3. Erstellung der täglichen Wetter-Feature-Raster
print(f"Lade ERA5 NetCDF für das Datum {target_date}...")
try:
    # Einlesen aller NetCDF-Archive aus den Einzel- und Boden-Verzeichnissen
    datasets = []
    for p in list(nc_single_dir.glob("*.nc")) + list(nc_soil_dir.glob("*.nc")):
        datasets.append(xr.open_dataset(p))
    
    # Zusammenführung inkrementeller Datensätze zu einem kohärenten XArray-Dataset
    # Argument compat='override' löst Konflikte experimenteller ERA5-Versionen (expver)
    print("  Führe Datensätze zusammen (Merging)...")
    ds = xr.merge(datasets, compat='override')
    
    # Eliminierung der experimentellen Dimension ('expver') falls präsent 
    if 'expver' in ds.coords or 'expver' in ds.dims:
        print("  Behandle 'expver'-Dimension (Experimentelle vs. Archivierte Daten)...")
        try:
            # Fusionierung von archivierten und echtzeitnahen (NRT) Reanalysedaten
            ds_archived = ds.sel(expver=1) if 1 in ds.expver else None
            ds_nrt = ds.sel(expver=5) if 5 in ds.expver else None
            
            if ds_archived is not None and ds_nrt is not None:
                ds = ds_archived.combine_first(ds_nrt)
            elif ds_archived is not None:
                ds = ds_archived
            elif ds_nrt is not None:
                ds = ds_nrt
        except Exception as e:
            print(f"  Hinweis: expver-Handhabung fehlgeschlagen ({e}), versuche dennoch fortzufahren...")

    # Robust time dimension detection
    global time_dim
    time_dim = 'time' if 'time' in ds.dims or 'time' in ds.coords else 'valid_time'
    if time_dim not in ds.variables and time_dim not in ds.dims:
        time_dim = next((c for c in ds.coords if 'time' in c.lower()), 'time')
    
    print(f"  Erkannte Zeit-Dimension: {time_dim}")
except Exception as e:
    print(f"\nTECHNISCHER FEHLER BEIM LADEN DER NETCDF: {e}")
    print("-" * 40)
    print("This error usually means the 'netcdf4' library is missing from your conda environment.")
    print("Please run: conda install -c conda-forge netcdf4")
    print("-" * 40 + "\n")
    raise ImportError(f"Could not load ERA5 NetCDF data: {e}")

# Berechnung der räumlichen Raster-Koordinaten für die Interpolation (UTM zu WGS84)
ds_lon_max = float(ds.longitude.max())

print("Generiere Koordinatengitter (UTM auf WGS84 Mapping)...")
X_utm, Y_utm = np.meshgrid(
    np.linspace(dst_transform.c, dst_transform.c + W*dst_transform.a, W),
    np.linspace(dst_transform.f, dst_transform.f + H*dst_transform.e, H)
)
from pyproj import Transformer
transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
Lon_grid, Lat_grid = transformer.transform(X_utm, Y_utm)

# Handhabung der ERA5 0-360 Longitude, falls erforderlich
if ds_lon_max > 180:
    Lon_grid[Lon_grid < 0] += 360

print("Interpoliere ERA5-Features auf das 500m Vorhersage-Raster...")

# Extraktion der Zustandvariablen (State Variables) exakt für das anvisierte Datum
ds_day = ds.sel({time_dim: target_date}, method="nearest")
ds_interp = ds_day.interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest")

v_sd = get_var(ds_interp, "snow_depth")
if v_sd:
    feature_rasters["Snow_depth"] = ds_interp[v_sd].values * 1000.0 # Umrechnung von m auf mm
    
for l in range(1, 5):
    v_soil = get_var(ds_interp, f"volumetric_soil_water_layer_{l}")
    if v_soil:
        feature_rasters[f"Soil_L{l}"] = ds_interp[v_soil].values

# Erfassung der Bodensättigungsdifferenz der vergangenen sieben Tage
target_date_7d = target_date - pd.Timedelta(days=7)
ds_7d = ds.sel({time_dim: target_date_7d}, method="nearest").interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest")

for l in [1, 4]:
    v_soil = get_var(ds_7d, f"volumetric_soil_water_layer_{l}")
    if v_soil and f"Soil_L{l}" in feature_rasters:
        feature_rasters[f"Soil_Change_7d_L{l}"] = feature_rasters[f"Soil_L{l}"] - ds_7d[v_soil].values

# Extraktion von Wetter-Fenstern (kumulatives Wetter der vorherigen Tage)
for label, days in WINDOWS.items():
    start = target_date - pd.Timedelta(days=days)
    ds_win = ds.sel({time_dim: slice(start, target_date)})
    
    # Summer des Niederschlags im gewählten Fenster berechnen
    v_tp = get_var(ds_win, "total_precipitation")
    if v_tp:
        agg = ds_win[v_tp].sum(dim=time_dim)
        interp = agg.interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest")
        feature_rasters[f"Rainfall_{label}_tp"] = interp.values * 1000.0 # Convert m to mm
        
    # Berechnung der kumulativen Schneefallmengen innerhalb des Temporal-Fensters
    v_sf = get_var(ds_win, "snowfall")
    if v_sf:
        agg = ds_win[v_sf].sum(dim=time_dim)
        interp = agg.interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest")
        feature_rasters[f"Snowfall_{label}"] = interp.values * 1000.0 # Convert m to mm
        
    # Aggregation der Durchschnittstemperatur (Transformation von Kelvin in Celsius)
    v_t2m = get_var(ds_win, "2m_temperature")
    if v_t2m:
        agg = ds_win[v_t2m].mean(dim=time_dim)
        interp = agg.interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest")
        feature_rasters[f"T2m_mean_{label}"] = interp.values - 273.15

# Berechnung der Schneeschmelzrate (Schneehöhe heute - Schneehöhe vor 3 Tagen)
print("Berechne Schneeschmelzrate (Änderung in 3 Tagen)...")
v_sd = get_var(ds, "snow_depth")
if v_sd:
    target_date_3d = target_date - pd.Timedelta(days=3)
    ds_3d = ds.sel({time_dim: target_date_3d}, method="nearest")
    # Re-Interpolation der tagesaktuellen Schneehöhen zur Datenvalidierung
    sd_now = ds_day[v_sd].interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest").values * 1000.0
    sd_3d = ds_3d[v_sd].interp(latitude=xr.DataArray(Lat_grid), longitude=xr.DataArray(Lon_grid), method="nearest").values * 1000.0
    feature_rasters["Snowmelt_rate_3d"] = sd_now - sd_3d  # Negative = melting

# Einbindung der spatial aggregierten Terrain-Derivate
# 'feature_rasters' already contains Plan_Curvature, Profile_Curvature, Aspect, TWI.
# Explizite Injektion der Hangneigung für Interaktions-Prädiktoren
feature_rasters["Slope"] = slope_clean

# Berechnung des kritischen Interaktions-Features (Hangneigung x Bodenfeuchte)
if "Slope" in feature_rasters and "Soil_L1" in feature_rasters:
    print("Berechne Interaktions-Feature: Slope_x_Soil_L1...")
    s = np.nan_to_num(feature_rasters["Slope"], nan=0.0)
    sw = np.nan_to_num(feature_rasters["Soil_L1"], nan=0.0)
    feature_rasters["Slope_x_Soil_L1"] = (s * sw).astype("float32")

if "Slope" in feature_rasters and "Soil_L4" in feature_rasters:
    print("Berechne Interaktions-Feature: Slope_x_Soil_L4...")
    s = np.nan_to_num(feature_rasters["Slope"], nan=0.0)
    sw = np.nan_to_num(feature_rasters["Soil_L4"], nan=0.0)
    feature_rasters["Slope_x_Soil_L4"] = (s * sw).astype("float32")

# Interaktion (Hang x Schneeschmelze): Erhöhtes Risiko an steilen Schmelzzonen
if "Slope" in feature_rasters and "Snowmelt_rate_3d" in feature_rasters:
    print("Berechne Interaktions-Feature: Slope_x_Snowmelt...")
    # Numerisches Clipping zur Prävention arithmetischer Overflows
    s = np.clip(np.nan_to_num(feature_rasters["Slope"], nan=0.0), 0, 90)
    sm = np.nan_to_num(feature_rasters["Snowmelt_rate_3d"], nan=0.0)
    # Vorzeichenwechsel der Schmelzrate und Ausreißer-Limitierung (Clipping)
    melt = np.clip(-sm, 0, 1000) # Max 1 meter of melt in 3 days is plenty
    feature_rasters["Slope_x_Snowmelt"] = (s * melt).astype("float32")
elif "Slope" in feature_rasters:
    feature_rasters["Slope_x_Snowmelt"] = np.zeros_like(feature_rasters["Slope"])
    
if "TWI" in feature_rasters and "Rainfall_2d_tp" in feature_rasters:
    print("Berechne Interaktions-Feature: TWI_x_Rainfall_2d...")
    twi = np.nan_to_num(feature_rasters["TWI"], nan=0.0)
    rain2d = np.nan_to_num(feature_rasters["Rainfall_2d_tp"], nan=0.0)
    feature_rasters["TWI_x_Rainfall_2d"] = (twi * rain2d).astype("float32")


# 4. Finales Stacking der Features und Modellvorhersage
print("Stapele Features (Stacking) für das Modell...")
X_stack = []

# Vorbereitung der Korrekturfaktoren (Temperatur durch Lapse Rate & Niederschlag orografisch)
lapse_rate = -0.0065
temp_delta_grid = np.zeros_like(dem_resampled)
rain_factor_grid = np.ones_like(dem_resampled)

# Geografische Rückprojektion (Lon/Lat) für den orografischen Höhenabgleich
for r in range(H_tpl):
    for c in range(W_tpl):
        h_pt = dem_resampled[r, c]
        if h_pt > 0:
            lo, la = Lon_grid[r, c], Lat_grid[r, c]
            lat_r = round(la * 4) / 4
            lon_r = round(lo * 4) / 4
            key = f"{lat_r:.2f}_{lon_r:.2f}"
            h_cell = era5_orog.get(key)
            if h_cell is not None and h_cell != -9999.0:
                elev_diff = h_pt - h_cell
                temp_delta_grid[r, c] = elev_diff * lapse_rate
                # Orografischer Skalierungsfaktor: +8% pro 100m (Grenzwerte: 0.5x bis 2.0x)
                rain_factor_grid[r, c] = np.clip(1 + (elev_diff / 100.0) * 0.08, 0.5, 2.0)

for f_name in feature_names:
    if f_name in feature_rasters:
        arr = feature_rasters[f_name]
        
        # Thermodynamische Höhenkorrektur (Lapse-Rate) der Temperaturvektoren
        if "T2m_mean" in f_name:
            arr = arr + temp_delta_grid
        
        # Apply Orographic Rainfall Correction
        if "Rainfall" in f_name or "Snowfall" in f_name:
            arr = arr * rain_factor_grid
            
        X_stack.append(arr.flatten())
    else:
        print(f"WARNUNG: Feature {f_name} fehlt! Werde mit 0 füllen.")
        X_stack.append(np.zeros(H*W))

X_in = np.column_stack(X_stack)

# Definition einer Gültigkeitsmaske für die gesamte Karte (Pixel mit vollständigen Daten)
valid_mask = np.isfinite(X_in).all(axis=1)

# --- STRICT SPATIAL FILTERING ---
# Die Wahrscheinlichkeits-Inferenz erfolgt exklusiv für Pixel mit:
# 1. Vollständigem Prädiktoren-Satz (Valid Mask)
# 2. Räumlicher Verortung innerhalb des Geltungsbereichs (Valid Slope Mask)
mask_1d = valid_slope_mask.flatten()
prediction_mask = valid_mask & mask_1d

print(f"Berechnung der Inferenz erfolgt für {np.sum(prediction_mask)} Pixel (Schnittmenge aus Daten & Südtirol-Grenze)...")

# Initialisieren des Raster-Arrays: -9999 repräsentiert maskierte Zonen (Transparenz)
prob_flat = np.full(H*W, -9999.0, dtype="float32")

# Repräsentation des Grundrisikos (Baseline) im gültigen Studienraum mit 0.0
# Gewährleistung kontinuierlicher Polygon-Strukturen für karge Ergebnis-Landschaften
prob_flat[mask_1d] = 0.0 

# Berechnung der probabilistischen Rutschungsgefahr mittels ML-Ensemble
if np.sum(prediction_mask) > 0:
    # Kapselung in DataFrame-Struktur zur Erhaltung der Prädiktoren-Metadaten
    X_df = pd.DataFrame(X_in[prediction_mask], columns=feature_names)
    preds = rf.predict_proba(X_df)[:, 1]
    # Projektion der Prädiktionen exakt auf valide Pixel-Schnittmengen
    prob_flat[prediction_mask] = preds

# Sicherheitsfunktion: Eliminierung jeglicher Werte außerhalb der Maskierung
prob_flat[~mask_1d] = -9999.0
prob_arr = prob_flat.reshape(H, W)



# 5. Export der Modellergebnisse (GeoTIFF) und Kartografische Visualisierung
out_path = outputs_dir / f"hazard_{target_date_str}.tif"
out_png_path = outputs_dir / f"hazard_{target_date_str}.png"

template_meta.update({"dtype": "float32", "count": 1, "nodata": -9999.0})
with rasterio.open(out_path, "w", **template_meta) as dst:
    dst.write(prob_arr, 1)
    
print(f"Tiff-Export erfolgreich gesichert unter: {out_path}")

# Plotting settings
# Restriktive Grenz-Filterung (Clipping) für alle außenliegenden Array-Segmente
prob_masked = np.ma.masked_where((prob_arr == -9999.0) | (~valid_slope_mask), prob_arr)
outline_mask = valid_slope_mask

# Abbildung mit neutral-grauem Hintergrund
fig = plt.figure(figsize=(12, 9), facecolor='#d3d3d3')
ax = fig.add_subplot(111, facecolor='#d3d3d3')

# Kontinuierliche Farbskala (Linear) von Türkis (0.0) nach Rot (1.0)
cmap = LinearSegmentedColormap.from_list("continuous_turquoise_red", ["turquoise", "red"])

im = ax.imshow(prob_masked, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im, label="Hazard Probability", shrink=0.8, pad=0.02)

# Projektion der topologischen Außengrenzen (Massive Schwarzkontur)
ax.contour(outline_mask, levels=[0.5], colors='black', linewidths=1.0)

# Visuelle Integration historischer Ground Truth Koordinaten (Validierung)
print("Lade Ground Truth Punkte für die Überlagerung...")
dataset_path = base_dir / "data" / "intermediate" / "dataset_era5.csv"
if dataset_path.exists():
    df_pts = pd.read_csv(dataset_path)
    pos = df_pts[df_pts["Label"] == 1]
    neg = df_pts[df_pts["Label"] == 0]
    
    # Rückprojektion sphärischer Koordinaten auf die Bildmatrix (Image Pixels)
    to_utm = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    
    def get_pixel_coords(lats, lons):
        east, north = to_utm.transform(lons.values, lats.values)
        from rasterio.transform import rowcol
        rows, cols = rowcol(dst_transform, east, north)
        return np.array(rows), np.array(cols)
    
    py_pos_all, px_pos_all = get_pixel_coords(pos["Latitude"], pos["Longitude"])
    py_neg_all, px_neg_all = get_pixel_coords(neg["Latitude"], neg["Longitude"])
    
    def filter_by_mask(rows, cols, mask):
        valid = []
        for r, c in zip(rows, cols):
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                valid.append(mask[r, c] == 1)
            else:
                valid.append(False)
        return np.array(valid)

    mask_pos = filter_by_mask(py_pos_all, px_pos_all, valid_slope_mask)
    mask_neg = filter_by_mask(py_neg_all, px_neg_all, valid_slope_mask)

    def in_bounds(rows, cols, h, w):
        return (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    # Isolation jener Ereignisse, welche präzis am Vorhersagedatum eingetreten sind
    date_col = "Datum"
    pos_dates = pd.to_datetime(pos[date_col]).dt.normalize()
    target_norm = target_date.normalize()
    is_today = (pos_dates == target_norm).values
    
    # 1. 1. Historische Referenz-Rutschungen (Zeitlich desjunkte Punkte)
    other_pos = mask_pos & (~is_today)
    ax.scatter(px_pos_all[other_pos], py_pos_all[other_pos], color="darkred", s=10, 
                label="Historische Landslides", alpha=0.3, marker="o")
    
    # 2. 2. Test-Validierungs-Punkte (Zieldatum: Treffer-Verifikation)
    # Strikter Ausschluss sämtlicher Koordinaten außerhalb des Polygons
    today_in = is_today & mask_pos

    ax.scatter(px_pos_all[today_in], py_pos_all[today_in], color="yellow", s=150, 
                label="Landslide Heute (Inside Area)", marker="*", edgecolor="black", linewidth=1, zorder=6)
    
    # 3. Random Safe Reference points
    ax.scatter(px_neg_all[mask_neg], py_neg_all[mask_neg], color="black", s=2, 
                label="No-Landslide Points", alpha=0.2)
    
    # Berechnung der prädiktiven Trefferquote (Capture Rate)
    # Quantifizierung der Ereignisse oberhalb der definierten Alarmierungs-Schwelle
    capture_rate = 0.0
    if np.any(today_in):
        today_rows = py_pos_all[today_in]
        today_cols = px_pos_all[today_in]
        
        # Räumliche Extrahierung der vom Modell geschätzten Wahrscheinlichkeiten
        hits = 0
        total_today = len(today_rows)
        for r, c in zip(today_rows, today_cols):
            # Prüfung gegen den optimal kalibrierten Trennschwellenwert
            if prob_arr[r, c] >= model_threshold:
                hits += 1
        capture_rate = hits / total_today
    # -------------------------------------------------------------

    ax.legend(loc="upper left", fontsize='x-small', framealpha=0.9)

ax.set_title(f"Landslide Hazard Model - {target_date_str}", fontsize=14, fontweight='bold')

# Einbettung der Modellierungs-Qualitätsmetriken (Scores) in die Kartenansicht
if metrics:
    stats_text = (f"ALLGEMEINE MODELLQUALITÄT (Train Split 70/30):\n"
                  f"- Gesamte Accuracy: {metrics.get('accuracy', 0)*100:.1f}%\n"
                  f"- Landslide Precision: {metrics.get('precision', 0)*100:.1f}%\n"
                  f"- Trainierter Schwellenwert: {model_threshold:.2f}\n\n"
                  f"HEUTIGE VALIDIERUNG ({target_date_str}):\n"
                  f"- Ereignis-Erfassungsrate: {capture_rate*100:.1f}%")
    
    # Fixierung der Metrik-Legende in der unteren Kartenecke
    ax.text(0.01, 0.01, stats_text, transform=ax.transAxes, fontsize=7,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Identifikation aller validen Rasterkoordinaten zur Begrenzungs-Optimierung
rows, cols = np.where(valid_slope_mask)
if len(rows) > 0:
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    # Implementierung eines Rahmen-Puffers (5 Pixel) im cartografischen Layout
    buffer = 5
    ax.set_xlim(c_min - buffer, c_max + buffer)
    ax.set_ylim(r_max + buffer, r_min - buffer) # Inverted Y for image coordinates
else:
    # Rückfall-Mechanismus auf maximale Rastergrenzen bei Maskierungs-Fehlern
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
# ----------------------------------------------------------------------------

# Integration topografischer Gitterlinien und sphärischer Achsen-Beschriftungen
# Bestimmung projizierter Kartengrenzen im geodätischen WGS84 System
x_utm_min, y_utm_max = dst_transform * (0, 0)
x_utm_max, y_utm_min = dst_transform * (W, H)
lon_min, lat_max = transformer.transform(x_utm_min, y_utm_max)
lon_max, lat_min = transformer.transform(x_utm_max, y_utm_min)

# number tick positions (jede 0.5 degrees)
lon_ticks_geo = np.arange(np.floor(lon_min * 2) / 2, np.ceil(lon_max * 2) / 2 + 0.1, 0.5)
lat_ticks_geo = np.arange(np.floor(lat_min * 2) / 2, np.ceil(lat_max * 2) / 2 + 0.1, 0.5)

# Interpolation geodätischer Achsen in affine Array-Transformationen
transformer_inv = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
x_ticks = []
for lon in lon_ticks_geo:
    x_utm, y_utm = transformer_inv.transform(lon, (lat_min + lat_max) / 2)
    px = (x_utm - dst_transform.c) / dst_transform.a
    if 0 <= px <= W:
        x_ticks.append(px)
x_labels = [f"{lon:.1f}°" for lon in lon_ticks_geo if 0 <= (transformer_inv.transform(lon, (lat_min + lat_max) / 2)[0] - dst_transform.c) / dst_transform.a <= W]

y_ticks = []
for lat in lat_ticks_geo:
    x_utm, y_utm = transformer_inv.transform((lon_min + lon_max) / 2, lat)
    py = (y_utm - dst_transform.f) / dst_transform.e
    if 0 <= py <= H:
        y_ticks.append(py)
y_labels = [f"{lat:.1f}°" for lat in lat_ticks_geo if 0 <= (transformer_inv.transform((lon_min + lon_max) / 2, lat)[1] - dst_transform.f) / dst_transform.e <= H]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)

plt.tight_layout()
plt.savefig(out_png_path, dpi=200, facecolor=fig.get_facecolor())
print(f"Visualisierung gespeichert unter: {out_png_path}")
