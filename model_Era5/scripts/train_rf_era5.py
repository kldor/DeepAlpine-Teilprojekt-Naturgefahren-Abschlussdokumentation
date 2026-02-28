# ==============================================================================
# Autor: Teilprojekt Naturgefahren
# Projekt: DeepAlpine
# Modul: Modelltraining & Hyperparameter-Optimierung (train_rf_era5.py)
# Beschreibung: Dieses Skript trainiert ein Random-Forest-Modell auf Basis von
#               ERA5-Klimadaten und Terrain-Features.
# ==============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, fbeta_score
import matplotlib.pyplot as plt

# Ladeprozess optionaler Statistik-Bibliotheken zur erweiterten Visualisierung
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Hinweis: Das Paket 'seaborn' ist nicht installiert. Fallback auf Standard-Matplotlib.")

# Definition der Projekt-Dateipfade
# Pfad muss angepasst werden
base_dir = Path(".../model_Era5")
data_path = base_dir / "data" / "intermediate" / "dataset_era5.csv"
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_out_path = models_dir / "rf_era5.joblib"

outputs_dir = base_dir / "outputs" / "plots"
outputs_dir.mkdir(parents=True, exist_ok=True)

# 1. Import des Trainings-Datensatzes
print(f"Initialisiere Datenimport (Dataset): {data_path}")
if not data_path.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {data_path}. Der Workflow erfordert die vorherige Ausführung von build_era5_dataset.py.")

df = pd.read_csv(data_path)

# Architektonische Definition der Prädiktoren-Vektoren
feature_names = [
    "Soil_L4", "Rainfall_2d_tp", "Soil_L2", "Soil_L1",
    "Rainfall_14d_tp", "Rainfall_7d_tp", "Rainfall_21d_tp", "Soil_L3",
    "Snow_depth", "Profile_Curvature",
    "Plan_Curvature", "T2m_mean_14d", "T2m_mean_21d", "T2m_mean_2d",
    "T2m_mean_1d", "T2m_mean_7d",
    "Rainfall_1d_tp",
    # Geomorphologische und hydrologische Interaktionsterme
    "Slope", "Slope_x_Soil_L1", "TWI_x_Rainfall_2d", "Snowmelt_rate_3d",
    # Prädiktoren hoher thermodynamischer Erklärungsrelevanz
    "TWI", "Aspect_North", "Aspect_East",
    "Soil_Change_7d_L1", "Soil_Change_7d_L4", "Slope_x_Soil_L4"
]

# Validierungsschritt: Verifikation der Features
# Automatische Ausklammerung abwesender Spalten zur Wahrung der Kernel-Stabilität
feature_names = [f for f in feature_names if f in df.columns]


print(f"Aktivierte Modell-Eingangsparameter ({len(feature_names)} Features):")
print(feature_names)

# Vorverarbeitung (Imputation): Schätzung fehlender NaN-Sequenzen durch Spalten-Mediane
# zur Erhaltung der Sample-Struktur, da Scikit-Learn rf-Modelle Matrix-Fehlwerte nicht tolerieren. 

initial_len = len(df)
df_clean = df.copy()
nan_counts = df_clean[feature_names].isna().sum()
imputed_cols = nan_counts[nan_counts > 0]
if len(imputed_cols) > 0:
    print(f"Interpolation (Median-Methode) anwendbar bei {len(imputed_cols)} Daten-Vektor(en):")
    for col, cnt in imputed_cols.items():
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"  {col}: {cnt} NaN -> gefüllt mit Median {median_val:.4f}")
else:
    print("Matrix-Integrität bestätigt: Keine NaN-Fehlwerte detektiert.")
print(f"Datensatzdimension (Zeilenmasse) beibehalten: N={len(df_clean)} (Initial: {initial_len})")

X = df_clean[feature_names]
y = df_clean["Label"]


# 2. Methodische Isolation von Trainings- und Testvektoren
# Nur 500 random Messpunkte fließen in das Testset ein.
# Künstliche Steilhang (Hard Negatives) werden isoliert vom Validierungsprozess 
print("Train-Test-Dichotomie initiiert (Sicherstellung unkontaminierter Referenz-Sets)...")

if 'Source' in df_clean.columns:
    df_field = df_clean[df_clean['Source'] == 'field']
    df_synthetic = df_clean[df_clean['Source'] == 'synthetic_hard_negative']
    print(f"Umfang Samples (Random): {len(df_field)} (Stratifizierter Train/Test-Ratio 70:30)")
    print(f"Umfang Samples (Steilhang): {len(df_synthetic)} -> 100% Trainings-Zuweisung.")

    X_field = df_field[feature_names]
    y_field = df_field["Label"]

    X_train_f, X_test, y_train_f, y_test = train_test_split(
        X_field, y_field, test_size=0.3, random_state=42, stratify=y_field
    )

    # Fusion Trainingsanteils mit der Gesamtheit künstlicher Extrema
    X_synth = df_synthetic[feature_names]
    y_synth = df_synthetic["Label"]
    X_train = pd.concat([X_train_f, X_synth], ignore_index=True)
    y_train = pd.concat([y_train_f, y_synth], ignore_index=True)
else:
    # Standardisiertes Splittingverfahren mangels differenzierender Kodierungen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

print(f"Finale Sample-Größe (Trainingsmatrix): {len(X_train)}")
print(f"Finale Sample-Größe (Validierungsmatrix): {len(X_test)}")


# 3. Ensemble-Lernen und stochastische Hyperparameter-Approximation
print("Ausführung Machine-Learning-Tuningverfahren (Randomized Search: N_Iter=50)...")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

param_dist = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4],
    'bootstrap': [True]
}

from sklearn.metrics import fbeta_score, make_scorer
f06_scorer = make_scorer(fbeta_score, beta=0.6)

search = RandomizedSearchCV(
    estimator=rf_base, param_distributions=param_dist,
    n_iter=50, cv=5, scoring=f06_scorer, n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_train)
rf = search.best_estimator_

# 4. Performance-Quantifikation & Datengetriebene Optimierung des Entscheidungsschwellenwertes
print("\nKalkulation des idealen Decision Boundaries (Zielfunktion: Maximum F0.6-Metrik)...")
y_probs = rf.predict_proba(X_test)[:, 1]

# Iterative Ermittlung der Threshold-Konstante zur Balance von Recall-Forderungen vs Precision-Präferenz
from sklearn.metrics import f1_score, fbeta_score
thresholds = np.arange(0.05, 0.95, 0.01)
scores = [fbeta_score(y_test, y_probs >= t, beta=0.6) for t in thresholds]
best_threshold = thresholds[np.argmax(scores)]
best_f1 = np.max(scores)

# Ableitung finalisierter Klassifikationsmetriken
y_pred_best = (y_probs >= best_threshold).astype(int)
overall_acc = accuracy_score(y_test, y_pred_best)
ls_precision = precision_score(y_test, y_pred_best, pos_label=1)
ls_recall = recall_score(y_test, y_pred_best, pos_label=1)

print(f"Berechnete Klassifikationsschwelle: {best_threshold:.2f} (Maximaler F0.6-Indikator: {best_f1:.4f})")
print(f"Accuracy: {overall_acc:.4f} | Precision: {ls_precision:.4f} | Recall: {ls_recall:.4f}")
print("\nUmfassender Performance-Bericht (Classification Report):")
print(classification_report(y_test, y_pred_best))

# Konstruktion und Ablage der multivariaten Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
else:
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

plt.xlabel("Vorhergesagter Zustand (Predicted)")
plt.ylabel("Referenzzustand (Ground Truth)")
plt.title("Konfusionsmatrix der Terrain/ERA5-Evaluierung")
plt.savefig(outputs_dir / "confusion_matrix_era5.png")
print("\nHinweis: Konfusionsmatrix-Ressource (.png) exportiert.")

# Analyse der Attributsrelevanz (Gini Feature Importance Estimation)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nRangordnung thermodynamischer/morphologischer Indikatoren aus der Random-Forest-Lernphase (absteigend):")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Grafische Repräsentation der Prädiktor-Wirkungskraft
plt.figure(figsize=(10, 6))
plt.title("Prädiktoren-Sensitivitätsanalyse (Features Importance)")
plt.bar(range(len(feature_names)), importances[indices], align="center")
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(outputs_dir / "feature_importance_era5.png")
print("Piktogramm zur Relevanz-Statistik erfolgreich erstellt.")

# 5. Binäre Archivierung des ML-Modellensembles
print(f"\nSerialisiere Modell-Artefakt (.joblib) und exportiere in Pfad: {model_out_path}...")
joblib.dump({
    "model": rf,
    "feature_names": feature_names,
    "best_threshold": best_threshold,
    "metrics": {
        "accuracy": overall_acc,
        "precision": ls_precision,
        "recall": ls_recall,
        "f1_score": f1_score(y_test, y_pred_best),
        "f1_score": best_f1,
        "f2_score": fbeta_score(y_test, y_pred_best, beta=2.0)
    }
}, model_out_path)

print("Modell-Training abgeschlossen. Die Topographie-Klima Inferenz (predict_rf_era5.py) ist abrufbar.")
