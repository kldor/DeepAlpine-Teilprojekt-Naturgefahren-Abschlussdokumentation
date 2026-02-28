# ==============================================================================
# Autor: Teilprojekt Naturgefahren
# Projekt: DeepAlpine
# Modul: Daten-Extraktion & Feature Engineering (build_era5_dataset.py)
# Beschreibung: Baut den initialen Trainingsdatensatz auf, iteriert über die
#               Inventar-Daten, rastert geografische Höhen/Neigungsmodelle und
#               aggregiert historische ERA5-Wetterfenster pro Standort.
# ==============================================================================
import os
import sys

# --- GLOBAL PROJ FIX ---
proj_path = "/Users/kiliandorn/miniconda3/envs/deepalpine_env/share/proj"
if os.path.exists(proj_path):
    os.environ["PROJ_LIB"] = proj_path
    os.environ["PROJ_DATA"] = proj_path
    os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import rowcol, xy
from pathlib import Path
from pyproj import Transformer
import random
import json

# Optionale Ladebalken-Implementierung (Fortschritts-Feedback)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Hinweis: 'tqdm' ist nicht installiert. Fortschrittsbalken wird nicht angezeigt.")

# Platzhalter-Klasse zur Umgehung von Exceptions bei fehlendem tqdm-Modul
class DummyPbar:
    def update(self, n=1): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Configuration & Paths
# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")

# Definition lokaler Rohdaten-Verzeichnispfade
landslides_csv = base_dir / "data" / "raw" / "vectors" / "landslides_beta.csv"
nc_single_dir = base_dir / "data" / "raw" / "era5_single" / "nc_singel"
nc_soil_dir = base_dir / "data" / "raw" / "era5_single" / "nc_soil"
slope_raster_path = base_dir / "data" / "raw" / "slope" / "slope_10m_UTM.tif"
plan_curv_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "plan_curvature_10m.tif"
profile_curv_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "profile_curvature_10m.tif"
aspect_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "aspect_10m.tif"
twi_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "twi_10m.tif"

# Speicherpfad für die finalisierte Trainings-Datenbank
out_csv = base_dir / "data" / "intermediate" / "dataset_era5.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

dem_raster_path = base_dir / "data" / "intermediate" / "rasters" / "10m" / "dem_10m_UTM.tif"
era5_orog_path = base_dir / "data" / "intermediate" / "era5_orography.json"

# Load ERA5 orography baseline
if era5_orog_path.exists():
    with open(era5_orog_path, 'r') as f:
        era5_orog = json.load(f)
    print(f"ERA5-Referenzorographie für {len(era5_orog)} Rasterzellen erfolgreich geladen.")
else:
    era5_orog = {}
    print("Warnung: ERA5-Referenzorographie nicht gefunden. Die Lapse-Rate-Korrektur wird deaktiviert.")

# Temporal-Fenster für kumulative Klimaextrahierung (Niederschneeregime)
WINDOWS = {
    "1d": 1,
    "2d": 2,
    "7d": 7,
    "14d": 14,
    "21d": 21,
}

# 1. Initiierung des Ladeprozesses (Quellen-Parsing)
print("Lade Landslides CSV (Rutschungsinventar)...")
df_pos = pd.read_csv(landslides_csv)

# Suchfunktion und Standardisierung der Datums-Vektoren
time_col_candidates = ["Datum", "date", "Date"]
date_col = next((c for c in time_col_candidates if c in df_pos.columns), None)
if not date_col:
    raise ValueError(f"No date column found. Tried: {time_col_candidates}")

df_pos[date_col] = pd.to_datetime(df_pos[date_col])
df_all_inventory = df_pos.copy()  # Keep full inventory before filter

# MOVEMENT_C = 99 means 'No Rutschung' (non-event observation point)
if 'MOVEMENT_C' in df_pos.columns:
    before = len(df_pos)
    df_pos = df_pos[df_pos['MOVEMENT_C'] != 99].copy()
    print(f"Gefiltert nach MOVEMENT_C != 99: {before} -> {len(df_pos)} tatsächliche Rutschungsereignisse")

df_pos['Source'] = 'field'   # Feldpunkte: 500 Zufällig generierte, Nicht Steilhang Punkte
print(f"Positive Samples (Rutschungen): {len(df_pos)}")

# Lade „Keine Rutschung“-Feldbeobachtungen (MOVEMENT_C = 99) als reale Negativbeispiele.
df_field_neg = df_all_inventory[df_all_inventory['MOVEMENT_C'] == 99].copy()
df_field_neg[date_col] = pd.to_datetime(df_field_neg[date_col])
df_field_neg['Source'] = 'field'
df_field_neg['Label'] = 0
print(f"Feld-Negative Samples (Keine Rutschung): {len(df_field_neg)}")


# Initialisierung der europäischen Reanalyse-Daten (ERA5)
print("Lade ERA5 NetCDF Dateien...")
try:
    # Einlesen aller NetCDF-Archive (Terrestrische/Atmosphärische Parameter)
    datasets = []
    for p in list(nc_single_dir.glob("*.nc")) + list(nc_soil_dir.glob("*.nc")):
        print(f"  Reading: {p.name}")
        datasets.append(xr.open_dataset(p))
    
    # Synthese inkrementeller Datensätze in ein vereinheitlichtes XArray
    # Das Argument compat='override' bereinigt Versionskonflikte ('expver') innerhalb der ERA5
    print("  Führe Datensätze zusammen (Merging)...")
    ds = xr.merge(datasets, compat='override')
    
    # Systemdiagnosik: Protokollierung bei abwesenden Zeitdimensionen
    if 'time' not in ds.variables and 'time' not in ds.dims:
        print("\nDEBUG: Dataset Structure:")
        print(f"  Dimensions: {list(ds.dims)}")
        print(f"  Coordinates: {list(ds.coords)}")
        print(f"  Variables: {list(ds.data_vars)}")

    # Eliminierung der experimentellen Dimension ('expver') falls präsent 
    if 'expver' in ds.coords or 'expver' in ds.dims:
        print("  Behandle 'expver'-Dimension (Experimentelle vs. Archivierte Daten)...")
        try:
            # Versuch der kohärenten Verknüpfung archivierter und echtzeitnaher Modelle
            ds_archived = ds.sel(expver=1) if 1 in ds.expver else None
            ds_nrt = ds.sel(expver=5) if 5 in ds.expver else None
            
            if ds_archived is not None and ds_nrt is not None:
                ds = ds_archived.combine_first(ds_nrt)
            elif ds_archived is not None:
                ds = ds_archived
            elif ds_nrt is not None:
                ds = ds_nrt
        except Exception as e:
            print(f"  Note: expver handling failed ({e}), attempting to proceed anyway...")

except Exception as e:
    print(f"\nTECHNISCHER FEHLER BEIM LADEN DER NETCDF: {e}")
    print("-" * 40)
    print("Vermutlich fehlt die Bibliothek 'netcdf4' in der Conda-Umgebung.")
    print("Bitte ausführen: conda install -c conda-forge netcdf4")
    print("-" * 40 + "\n")
    raise ImportError(f"Could not load ERA5 NetCDF data: {e}")

# Bestimmung epochaler Grenzwerte zur Erzeugung validierter Null-Szenarien
# Dynamische Unterstützung klassischer und CDS-kompatibler Zeitvariablen
time_dim = 'time' if 'time' in ds.dims or 'time' in ds.coords else 'valid_time'
if time_dim not in ds.variables and time_dim not in ds.dims:
     # Ultimativer Rückfallmechanismus: Suche nach Zeit-Synonymen im Array
    time_dim = next((c for c in ds.coords if 'time' in c.lower()), 'time')

era5_min_time = pd.to_datetime(ds[time_dim].min().values)
era5_max_time = pd.to_datetime(ds[time_dim].max().values)
print(f"ERA5 Time range ({time_dim}): {era5_min_time} to {era5_max_time}")

print("\nAvailable variables in ERA5 file:")
print(list(ds.data_vars))

# Prüfung longitudinaler Kartierung (0-360° vs -180°/+180° Konventionen)
ds_lon_min = float(ds.longitude.min())
ds_lon_max = float(ds.longitude.max())
print(f"ERA5 Longitude Range: {ds_lon_min} to {ds_lon_max}")

# Variablen-Umwandlung (Short Name Hash-Map)
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

# Semantischer Iterator zur Variablen-Identifikation im Matrix-Korpus
def get_var(dataset, long_name):
    if long_name in dataset:
        return long_name
    short = VAR_MAP.get(long_name)
    if short in dataset:
        return short
    return None
    
# Erzeugung topografischer Extrema: 200 sichere Vektoren an Hängen (> 15°).
# Erzwingt vom Klassifikator das Erlernen klimatischer Dynamiken abseits platter Heuristiken.
num_synthetic_negatives = 200
print(f"Generiere {num_synthetic_negatives} synthetische Extremfall-Samples (Hangneigung >= 15°)...")

neg_samples = []

with rasterio.open(slope_raster_path) as src_slope, \
     rasterio.open(plan_curv_path) as src_plan, \
     rasterio.open(profile_curv_path) as src_prof, \
     rasterio.open(aspect_path) as src_aspect, \
     rasterio.open(twi_path) as src_twi, \
     rasterio.open(dem_raster_path) as src_dem:
    
    height = src_slope.height
    width = src_slope.width
    crs_utm = src_slope.crs
    transform = src_slope.transform
    nodata = src_slope.nodata

    # Initialisierung der Koordinaten-Transformationen (UTM Projektion auf WGS84)
    transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)

    print(f"  Randomly sampling points from raster ({width}x{height}) until 200 steep points are found...")
    
    # Rejection-Sampling: Akkumulierung stochastischer Pixel oberhalb der Neigungsschwelle
    found = 0
    if HAS_TQDM:
        pbar = tqdm(total=num_synthetic_negatives, desc="Hard Negatives")
    else:
        pbar = DummyPbar()
    
    while found < num_synthetic_negatives:
        # Aufruf randständiger Zeilen/Spalten-Indizes
        r = random.randint(0, height - 1)
        c = random.randint(0, width - 1)
        
        # I/O Operation für den spezifizierten Matrix-Vektor
        window = rasterio.windows.Window(c, r, 1, 1)
        val = src_slope.read(1, window=window)[0, 0]
        
        # Hard Negative Condition: Slope must be >= 15.0 degrees
        if val != nodata and np.isfinite(val) and val >= 15.0 and val <= 90.0:
            x_utm, y_utm = src_slope.xy(r, c)
            lon, lat = transformer.transform(x_utm, y_utm)
            
            # Select random valid date in ERA5 range
            # Implementierung eines Vorlaufpuffers für stabile Temporalanalysen (Rolling Features)
            start_safe = era5_min_time + pd.Timedelta(days=22)
            days_range = (era5_max_time - start_safe).days
            if days_range <= 0:
                 random_date = era5_min_time 
            else:
                random_days = random.randint(0, days_range)
                random_date = start_safe + pd.Timedelta(days=random_days)
            
            neg_samples.append({
                date_col: random_date,
                "Latitude": lat,
                "Longitude": lon,
                "Label": 0, # Non-landslide
                "Source": "synthetic_hard_negative",
                "Slope": float(val),
                "Plan_Curvature": float(src_plan.read(1, window=window)[0, 0]),
                "Profile_Curvature": float(src_prof.read(1, window=window)[0, 0]),
                "Aspect": float(src_aspect.read(1, window=window)[0, 0]),
                "TWI": float(src_twi.read(1, window=window)[0, 0]),
                "Elevation": float(src_dem.read(1, window=window)[0, 0])
            })
            found += 1
            pbar.update(1)
            
    pbar.close()

df_neg = pd.DataFrame(neg_samples)
print(f"  {len(df_neg)} synthetic hard negatives created.")

# Validierung affiner UTM Rück Transformationen
with rasterio.open(slope_raster_path) as src_slope:
    crs_utm = src_slope.crs
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)

# 2.5 Auslesung geo-morphologischer Rasterwerte für Ground Truth Koordinaten
print("Extrahiere Terrain-Features für Positive Samples...")
with rasterio.open(slope_raster_path) as src_slope, \
     rasterio.open(plan_curv_path) as src_plan, \
     rasterio.open(profile_curv_path) as src_prof, \
     rasterio.open(aspect_path) as src_aspect, \
     rasterio.open(twi_path) as src_twi, \
     rasterio.open(dem_raster_path) as src_dem:
    
    nodata_s = src_slope.nodata
    plan_vals = []
    prof_vals = []
    slope_vals = []
    aspect_vals = []
    twi_vals = []
    elev_vals = []
    
    for idx, row in df_pos.iterrows():
        east, north = to_utm.transform(row["Longitude"], row["Latitude"])
        r, c = rowcol(src_plan.transform, east, north)
        
        # Standard-Fallback bei ungültigen Vektoren
        p_val, pr_val, s_val, a_val, t_val, h_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if 0 <= r < src_plan.height and 0 <= c < src_plan.width:
            win = rasterio.windows.Window(c, r, 1, 1)
            p_val  = src_plan.read(1, window=win)[0, 0]
            pr_val = src_prof.read(1, window=win)[0, 0]
            s_val  = src_slope.read(1, window=win)[0, 0]
            a_val  = src_aspect.read(1, window=win)[0, 0]
            t_val  = src_twi.read(1, window=win)[0, 0]
            h_val  = src_dem.read(1, window=win)[0, 0]
            
        plan_vals.append(float(p_val) if np.isfinite(p_val) else 0.0)
        prof_vals.append(float(pr_val) if np.isfinite(pr_val) else 0.0)
        aspect_vals.append(float(a_val) if np.isfinite(a_val) else 0.0)
        twi_vals.append(float(t_val) if np.isfinite(t_val) else 0.0)
        elev_vals.append(float(h_val) if np.isfinite(h_val) else 0.0)
        # Wert-Korrektur (Clamping): Eliminierung physikalisch unmögliche Hangneigungen (> 90°)
        s_clean = float(s_val) if s_val != nodata_s and np.isfinite(s_val) and s_val <= 90.0 else 0.0
        slope_vals.append(s_clean)

    df_pos["Plan_Curvature"] = plan_vals
    df_pos["Profile_Curvature"] = prof_vals
    df_pos["Aspect"] = aspect_vals
    df_pos["TWI"] = twi_vals
    df_pos["Elevation"] = elev_vals
    df_pos["Slope"] = slope_vals
    print(f"  Positive sample slopes: mean={sum(slope_vals)/len(slope_vals):.1f}°, "
          f"max={max(slope_vals):.1f}°, zeros={slope_vals.count(0.0)}")


# Iteration geomorphologischer Felder für negative Terrain-Befunde (0-Daten)
print("Extrahiere Terrain-Features für Feld-Negative Samples...")
with rasterio.open(slope_raster_path) as src_slope, \
     rasterio.open(plan_curv_path) as src_plan, \
     rasterio.open(profile_curv_path) as src_prof, \
     rasterio.open(aspect_path) as src_aspect, \
     rasterio.open(twi_path) as src_twi, \
     rasterio.open(dem_raster_path) as src_dem:
    crs_utm2 = src_slope.crs
    to_utm2 = Transformer.from_crs("EPSG:4326", crs_utm2, always_xy=True)
    nodata2 = src_slope.nodata
    plan_vals2, prof_vals2, slope_vals2, aspect_vals2, twi_vals2, elev_vals2 = [], [], [], [], [], []
    for idx, row in df_field_neg.iterrows():
        east, north = to_utm2.transform(row["Longitude"], row["Latitude"])
        r2, c2 = rowcol(src_plan.transform, east, north)
        p_val, pr_val, s_val, a_val, t_val, h_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if 0 <= r2 < src_plan.height and 0 <= c2 < src_plan.width:
            win2 = rasterio.windows.Window(c2, r2, 1, 1)
            p_val = src_plan.read(1, window=win2)[0, 0]
            pr_val = src_prof.read(1, window=win2)[0, 0]
            s_val = src_slope.read(1, window=win2)[0, 0]
            a_val = src_aspect.read(1, window=win2)[0, 0]
            t_val = src_twi.read(1, window=win2)[0, 0]
            h_val = src_dem.read(1, window=win2)[0, 0]
        plan_vals2.append(float(p_val) if np.isfinite(p_val) else 0.0)
        prof_vals2.append(float(pr_val) if np.isfinite(pr_val) else 0.0)
        aspect_vals2.append(float(a_val) if np.isfinite(a_val) else 0.0)
        twi_vals2.append(float(t_val) if np.isfinite(t_val) else 0.0)
        elev_vals2.append(float(h_val) if np.isfinite(h_val) else 0.0)
        # Clamp: slope > 90° = corrupt raster pixel -> treat as 0
        s_clean2 = float(s_val) if s_val != nodata2 and np.isfinite(s_val) and s_val <= 90.0 else 0.0
        slope_vals2.append(s_clean2)

    df_field_neg["Plan_Curvature"] = plan_vals2
    df_field_neg["Profile_Curvature"] = prof_vals2
    df_field_neg["Aspect"] = aspect_vals2
    df_field_neg["TWI"] = twi_vals2
    df_field_neg["Elevation"] = elev_vals2
    df_field_neg["Slope"] = slope_vals2
    
    # --- HARD NEGATIVE FILTERING REMOVED ---
    # Integrationsrichtlinie: Einspeisung sämtlicher empirischer Negativdaten in das Training
    # Abschaltung restriktiver Hangfilter zur Erhaltung kritischer Quantitäten
    print(f"  Using all field negative samples: {len(df_field_neg)} points")
    print(f"  Field neg slopes: mean={df_field_neg['Slope'].mean():.1f}°, "
          f"max={df_field_neg['Slope'].max():.1f}°, zeros={len(df_field_neg[df_field_neg['Slope'] == 0])}")

# Aggregationsabschluss der Hauptmatrix (Ground-Truth, Safe-Points, Synthetics)
df_pos["Label"] = 1
cols_to_keep = [date_col, "Latitude", "Longitude", "Label", "Source", "Plan_Curvature", "Profile_Curvature", "Aspect", "TWI", "Slope", "Elevation"]

# Erhalt der UUID Kennung zum korrekten Mapping optischer Satellitenszenen
if "UUID" in df_pos.columns:
    cols_to_keep.append("UUID")

# Build field_neg columns list (no UUID)
cols_field_neg = [date_col, "Latitude", "Longitude", "Label", "Source", "Plan_Curvature", "Profile_Curvature", "Aspect", "TWI", "Slope", "Elevation"]

df_combined = pd.concat([
    df_pos[cols_to_keep],
    df_field_neg[cols_field_neg],
    df_neg  # Synthetic hard negatives (has matching columns)
], ignore_index=True)

print(f"Datensatz-Größe: {len(df_combined)} (Pos: {len(df_pos)}, Feld-Neg: {len(df_field_neg)}, Synth-Neg: {len(df_neg)})")

# 2.6 Merge Satellite Features (Optional) Nicht in engültigen Modell, soll der Reproduzierbarkeit des Versuchs der Integrierung von SatData dienen
satellite_features_path = base_dir / "data" / "intermediate" / "satellite_features.csv"
if satellite_features_path.exists():
    print("\nFüge Satelliten-Features hinzu (Merging)...")
    df_sat = pd.read_csv(satellite_features_path)
    
    # Verknüpfung (Left-Join) des Sentinel-2 Arrays über den Primärschlüssel UUID
    df_combined = df_combined.merge(df_sat, on="UUID", how="left")
    
    # Fill missing satellite values with 0 (neutral value)
    satellite_cols = ["NDVI_pre", "NDWI_pre", "BSI_pre"]
    for col in satellite_cols:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(0.0)
    
    # Quantifizierung erfolgreicher Satelliten-Daten Injektionen
    has_satellite = df_combined[satellite_cols].notna().all(axis=1).sum()
    print(f"  Satellite coverage: {has_satellite}/{len(df_combined)} samples ({100*has_satellite/len(df_combined):.1f}%)")
    print(f"  Added features: {satellite_cols}")
else:
    print("\nKeine Satelliten-Features gefunden. Überspringe Schritt.")
    print(f"  (Bitte zuerst extract_satellite_features.py ausführen)")


# 3. Feature Extraction (ERA5)

error_count = 0
def extract_features(row):
    """
    Main extraction function called for every point in the dataset.
    Extracts weather conditions at the specific time and place.
    """
    global error_count
    lat = row["Latitude"]
    lon = row["Longitude"]
    date = row[date_col]
    
    # Handle ERA5 0-360 longitude mapping
    if ds_lon_max > 180 and lon < 0:
        lon_query = lon + 360
    else:
        lon_query = lon

    feats = {}
    
    # Lapse Rate Correction Prep
    h_point = row.get("Elevation", 0.0)
    # Suche nach kongruenten ERA5 Gitterzellen
    lat_r = round(lat * 4) / 4
    lon_r = round(lon_query * 4) / 4
    cell_key = f"{lat_r:.2f}_{lon_r:.2f}"
    h_cell = era5_orog.get(cell_key)
    
    lapse_rate = -0.0065  # °C per meter
    temp_delta = 0.0
    rain_factor = 1.0  # Default: no scaling
    
    if h_cell is not None and h_cell != -9999.0 and h_point > 0:
        elevation_diff = h_point - h_cell
        temp_delta = elevation_diff * lapse_rate
        # Orografische Skalierung: +8% Niederschlagskorrektur per 100m Höhendifferenz
        # Limiten-Clamping des Faktors [0.5, 2.0] als Puffer arithmetischer Divergenzen
        rain_factor = np.clip(1 + (elevation_diff / 100.0) * 0.08, 0.5, 2.0)
    
    try:
        # Sicherheits-Check: Datums-Valitität in der extrahierten XArray-Zeitachse
        if date < era5_min_time or date > era5_max_time:
            if error_count < 10:
                print(f"Date out of range: {date}")
            return pd.Series(dtype="float64")

        # 1. Initialzustände (State Features): Exakte meteorologische Parameter am Event-Tag
        ds_day = ds.sel({time_dim: date}, method="nearest").interp(
            latitude=lat, longitude=lon_query, method="nearest"
        )
        
        v_sd = get_var(ds_day, "snow_depth")
        if v_sd:
            snow_now = float(ds_day[v_sd]) * 1000.0  # Convert m to mm
            feats["Snow_depth"] = snow_now
            
            # Snowmelt rate: Snow_depth(t) minus Snow_depth(t-3d)
            # Bedeutung negativer Werte: Aktive thermische Schneeschmelze -> Signifikantes Rutsch-Risiko
            date_3d = date - pd.Timedelta(days=3)
            if date_3d >= era5_min_time:
                ds_3d = ds.sel({time_dim: date_3d}, method="nearest").interp(
                    latitude=lat, longitude=lon_query, method="nearest"
                )
                snow_3d = float(ds_3d[v_sd]) * 1000.0
                feats["Snowmelt_rate_3d"] = snow_now - snow_3d  # negative = melting
            else:
                feats["Snowmelt_rate_3d"] = 0.0
        
        # Extrahierung bodenphysikalischer Wasser-Retention über alle Ebenen (Layers)
        for l in range(1, 5):
            v_soil = get_var(ds_day, f"volumetric_soil_water_layer_{l}")
            if v_soil: feats[f"Soil_L{l}"] = float(ds_day[v_soil])
            
        # Capture Soil Moisture change over the last 7 days (Trend / Flash saturation)
        date_7d = date - pd.Timedelta(days=7)
        if date_7d >= era5_min_time:
            ds_7d = ds.sel({time_dim: date_7d}, method="nearest").interp(
                latitude=lat, longitude=lon_query, method="nearest"
            )
            for l in [1, 4]:  # Track top layer (fast) and bottom layer (slow) changes
                v_soil = get_var(ds_7d, f"volumetric_soil_water_layer_{l}")
                if v_soil and f"Soil_L{l}" in feats: 
                    feats[f"Soil_Change_7d_L{l}"] = feats[f"Soil_L{l}"] - float(ds_7d[v_soil])
        else:
            for l in [1, 4]:
                if f"Soil_L{l}" in feats:
                    feats[f"Soil_Change_7d_L{l}"] = 0.0
        
        # 2. Rolling-Windows: Kumulative Aggregation der Historie (Niederschläge, Thermik)
        for label, days in WINDOWS.items():
            start_date = date - pd.Timedelta(days=days)
            
            # Selektion der zutreffenden Temporalabschnitte
            ds_win = ds.sel({time_dim: slice(start_date, date)})
            
            # Räumliche Interpolation (Nearest Neighbor / Bilinear) zum Schadenspunkt
            ds_pt = ds_win.interp(latitude=lat, longitude=lon_query, method="nearest")
            
            # Niederschlags-Logik: Aggregation physikalischer Flüssig/Fest-Massen (Summe)
            # Einheits-Umformung: Transformation der ERA5 Tensor-Metrik (Meter) in Millimeter (mm)
            # Applikation des abgeleiteten Orografie-Strafmaßes (Elevation Korrektur)
            v_tp = get_var(ds_pt, "total_precipitation")
            if v_tp: 
                raw_tp = float(ds_pt[v_tp].sum(dim=time_dim)) * 1000.0
                feats[f"Rainfall_{label}_tp"] = raw_tp * rain_factor
            
            v_sf = get_var(ds_pt, "snowfall")
            if v_sf: 
                raw_sf = float(ds_pt[v_sf].sum(dim=time_dim)) * 1000.0
                feats[f"Snowfall_{label}"] = raw_sf * rain_factor
            
            # Temperatur-Logik: Höhenabgleich (Lapse-Rate) und Kelvin/Celsius Umrechnung
            v_t2m = get_var(ds_pt, "2m_temperature")
            if v_t2m: 
                raw_temp_k = float(ds_pt[v_t2m].mean(dim=time_dim))
                raw_temp_c = raw_temp_k - 273.15
                feats[f"T2m_mean_{label}"] = raw_temp_c + temp_delta
                
    except Exception as e:
        if error_count < 10:
            print(f"Extraction Error at {lat}, {lon}, {date}: {e}")
        error_count += 1
        
    return pd.Series(feats)

print("Verarbeite Punkte (Fortschritt wird alle 10 Zeilen angezeigt)...")
total = len(df_combined)
results = []
for i, (idx, row) in enumerate(df_combined.iterrows()):
    results.append(extract_features(row))
    if (i + 1) % 10 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total} done")
features_df = pd.DataFrame(results, index=df_combined.index)

# Matrix-Zusammenführung von Vektoren (Slope, Aspekt) mit Klimavariablen
final_df = pd.concat([df_combined, features_df], axis=1)

# 4. Post-processing (Auch alte Versuche dabei, ignorierbar)
# Slope x Soil 
if "Slope" in final_df.columns and "Soil_L1" in final_df.columns:
    print("Berechne Interaktions-Feature: Slope_x_Soil_L1...")
    final_df["Slope_x_Soil_L1"] = final_df["Slope"] * final_df["Soil_L1"]

# Slope x Soil_L4 
if "Slope" in final_df.columns and "Soil_L4" in final_df.columns:
    print("Berechne Interaktions-Feature: Slope_x_Soil_L4...")
    final_df["Slope_x_Soil_L4"] = final_df["Slope"] * final_df["Soil_L4"]

# Slope x Snowmelt
# Verstärkte negative Schmelzraten induzieren sofortigen Wassereintrag ins Lockergestein
if "Slope" in final_df.columns and "Snowmelt_rate_3d" in final_df.columns:
    print("Berechne Interaktions-Feature: Slope_x_Snowmelt...")
    # Vorzeichenwechsel (-Melt = Positiver Eintrag) erleichtert Baumalgorithmen die Splits
    # Clippen bei Ebene 0: Akkumulierender Neuschnee verringert Schmelzraten
    # Mathematischer Angleich als Basiswert=0. Parallelität zur Logik in predict_rf gewahrt.
    melt = (-final_df["Snowmelt_rate_3d"]).clip(lower=0)
    final_df["Slope_x_Snowmelt"] = final_df["Slope"] * melt

# TWI x Rainfall
if "TWI" in final_df.columns and "Rainfall_2d_tp" in final_df.columns:
    print("Berechne Interaktions-Feature: TWI_x_Rainfall_2d...")
    final_df["TWI_x_Rainfall_2d"] = final_df["TWI"] * final_df["Rainfall_2d_tp"]

if "Aspect" in final_df.columns:
    print("Berechne Aspect_North und Aspect_East...")
    final_df["Aspect_North"] = np.cos(np.radians(final_df["Aspect"]))
    final_df["Aspect_East"] = np.sin(np.radians(final_df["Aspect"]))

# 5. Save Results
final_df.to_csv(out_csv, index=False)
print("Abgeschlossen!")
print(f"Gespeichert unter: {out_csv}")
print(final_df.head())
