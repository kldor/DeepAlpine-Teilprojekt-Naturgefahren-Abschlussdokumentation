================================================================================
Ausführung der Pipeline für model_Era5
================================================================================

Systemvoraussetzung: Conda-Umgebung (deepalpine_env) ist aktiv.

Die nachfolgenden 6 Skripte müssen zwingend in dieser chronologischen 
Reihenfolge zur Neu-Initialisierung des Modells ausgeführt werden:

1. GEOMORPHOLOGISCHE VORVERARBEITUNG (Einmalig)
- python3 build_template_grid.py
  * Generiert ein leeres 500m-Raster (Bounding Box der Untersuchungsregion) 
    als geometrisches Referenzobjekt.
  
- python3 process_terrain_features.py
  * Verwendet das vorhandene 10m-Hangneigungs-Raster (Slope) als strukturelle 
    Referenz-Schablone (Match-Reference).
  * Berechnet daraus und aus dem Basis-DEM weitere physikalische Derivate: 
    Exposition (Aspect), Krümmungen (Curvatures) und den topographischen 
    Feuchtigkeitsindex (TWI).

- python3 create_coarse_dem.py
  * Berechnet eine grob aufgelöste Höhen-Referenzschicht (ERA5-Orographie).
  * Zwingend erforderlich für die spätere physikalische Interpolation der 
    Temperaturdaten (Lapse-Rate-Korrektur).

2. DATENEXTRAKTION UND MERGING
- python3 build_era5_dataset.py
  * Iteriert über die verifizierten Koordinaten des Rutschungsinventars.
  * Extrahiert historische Wetterhistorien (2 bis 21 Tage) aus den NetCDF-Archiven.
  * Generiert 200 synthetische Null-Variablen auf steilen Hängen (basierend 
    auf dem Slope-Raster) zur Vermeidung von Fehl-Heuristiken.
  * Exportiert den finalisierten Datensatz nach: data/intermediate/dataset_era5.csv

3. MODELL-TRAINING
- python3 train_rf_era5.py
  * Initialisiert das Training des Random Forests anhand der kompilierten 
    CSV-Daten.
  * Gewährleistet eine strikte Trennung von Train- und Testdaten.
  * Bestimmt den optimalen Klassifikations-Schwellenwert (F0.6-Score).
  * Serialisiert das trainierte Modell-Artefakt nach: models/rf_era5.joblib

4. INFERENZ UND VISUALISIERUNG
- python3 predict_rf_era5.py YYYY-MM-DD (Bsp.: python3 predict_rf_era5.py 2024-09-10)
  * Lädt das vortrainierte Klassifikationsmodell.
  * Verschneidet die topographischen Variablen mit dem ERA5-Wetter am definierten 
    Zieldatum.
  * Erzeugt ein probabilistisches GeoTIFF sowie eine gerenderte png-Karte mit 
    überlagerten Validierungspunkten unter /outputs/.
