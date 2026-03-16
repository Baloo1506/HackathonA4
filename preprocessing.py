"""
HR Dataset Preprocessing Pipeline
===================================
Objectif : anonymiser le dataset RH et le préparer pour la prédiction d'attrition.

Étapes :
  1. Suppression des colonnes confidentielles identifiantes
     (nom, ID employé, code postal, nom/ID manager)
  2. Transformation des dates en métriques numériques utiles
     - DOB → Age (en années)
     - DateofHire → TenureYears (ancienneté en années)
     - LastPerformanceReview_Date → DaysSinceLastReview
  3. Suppression des colonnes redondantes
     (versions numériques-ID de colonnes catégorielles déjà présentes)
  4. Nettoyage de la colonne DateofTermination :
     supprimée car trop corrélée à la cible Termd et non disponible en temps réel
  5. Encodage des variables catégorielles (One-Hot)
  6. Normalisation des variables numériques continues

Usage :
    python preprocessing.py
    → produit : HRDataset_preprocessed.csv  (données anonymisées prêtes pour ML)
    → produit : HRDataset_anonymized.csv     (données anonymisées lisibles, avant encodage)
"""

import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
INPUT_FILE = "HRDataset_v14_enriched.csv"
OUTPUT_ANONYMIZED = "HRDataset_anonymized.csv"
OUTPUT_ML_READY = "HRDataset_preprocessed.csv"

# Colonnes à supprimer car directement identifiantes (PII)
PII_COLUMNS = [
    "Employee_Name",   # Nom complet de l'employé
    "EmpID",           # Identifiant unique employé
    "Zip",             # Code postal (localisation précise)
    "ManagerName",     # Nom du manager
    "ManagerID",       # Identifiant du manager
]

# Colonnes redondantes : versions ID numériques de colonnes catégorielles déjà présentes
REDUNDANT_ID_COLUMNS = [
    "MarriedID",       # doublon de MaritalDesc
    "MaritalStatusID", # doublon de MaritalDesc
    "GenderID",        # doublon de Sex
    "EmpStatusID",     # doublon de EmploymentStatus
    "DeptID",          # doublon de Department
    "PerfScoreID",     # doublon de PerformanceScore
    "PositionID",      # doublon de Position
]

# Colonnes de dates à transformer
DATE_COLUMNS_TO_TRANSFORM = {
    "DOB": "Age",
    "DateofHire": "TenureYears",
    "LastPerformanceReview_Date": "DaysSinceLastReview",
}

# Colonnes à supprimer après transformation ou inutiles pour la prédiction
COLUMNS_TO_DROP_AFTER_TRANSFORM = [
    "DateofTermination",  # non disponible en prédiction temps réel (futur)
    "TermReason",         # non disponible en prédiction temps réel (futur)
    "EmploymentStatus",   # directement dérivée de Termd (cible), cause data leakage
]

# Variables numériques continues à normaliser
NUMERIC_COLS_TO_SCALE = [
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "Age",
    "TenureYears",
    "DaysSinceLastReview",
]

# Variables catégorielles à encoder (One-Hot)
CATEGORICAL_COLS = [
    "Position",
    "State",
    "Sex",
    "MaritalDesc",
    "CitizenDesc",
    "HispanicLatino",
    "RaceDesc",
    "RecruitmentSource",
    "PerformanceScore",
    "Department",
]


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def parse_date(series: pd.Series) -> pd.Series:
    """Convertit une colonne de dates en format datetime (plusieurs formats acceptés)."""
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    # Correction des années sur 2 chiffres mal interprétées (ex: 05/05/75 → 1975 et non 2075)
    future_cutoff = pd.Timestamp(date.today()) + pd.DateOffset(years=1)
    future_mask = parsed > future_cutoff
    if future_mask.any():
        parsed = parsed.where(~future_mask,
                              parsed[future_mask] - pd.DateOffset(years=100))
    return parsed


def compute_age(dob_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Calcule l'âge en années entières à partir de la date de naissance."""
    if reference_date is None:
        reference_date = date.today()
    dob = parse_date(dob_series)
    ref = pd.Timestamp(reference_date)
    age = (ref - dob).dt.days // 365
    return age.astype("Int64")


def compute_tenure_years(hire_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Calcule l'ancienneté en années à partir de la date d'embauche."""
    if reference_date is None:
        reference_date = date.today()
    hire = parse_date(hire_series)
    ref = pd.Timestamp(reference_date)
    tenure = (ref - hire).dt.days / 365.25
    return tenure.round(2)


def compute_days_since_review(review_series: pd.Series, reference_date: date = None) -> pd.Series:
    """Calcule le nombre de jours depuis le dernier entretien d'évaluation."""
    if reference_date is None:
        reference_date = date.today()
    review = parse_date(review_series)
    ref = pd.Timestamp(reference_date)
    days = (ref - review).dt.days
    return days.astype("Int64")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_preprocessing(input_file: str = INPUT_FILE,
                      output_anonymized: str = OUTPUT_ANONYMIZED,
                      output_ml_ready: str = OUTPUT_ML_READY,
                      reference_date: date = None) -> pd.DataFrame:
    """
    Exécute le pipeline de prétraitement complet.

    Paramètres
    ----------
    input_file : chemin vers le CSV source
    output_anonymized : chemin vers le CSV anonymisé lisible
    output_ml_ready : chemin vers le CSV prêt pour ML (encodé + normalisé)
    reference_date : date de référence pour les calculs (utile pour les tests)

    Retourne
    --------
    df_ml : DataFrame prêt pour l'entraînement d'un modèle ML
    """
    print(f"[1/7] Chargement de '{input_file}'...")
    df = pd.read_csv(input_file)
    print(f"      {len(df)} lignes × {len(df.columns)} colonnes chargées.")

    # ------------------------------------------------------------------
    # Étape 1 : Suppression des colonnes PII
    # ------------------------------------------------------------------
    print("[2/7] Suppression des colonnes confidentielles (PII)...")
    cols_to_remove = [c for c in PII_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_remove)
    print(f"      Colonnes supprimées : {cols_to_remove}")

    # ------------------------------------------------------------------
    # Étape 2 : Transformation des dates en métriques numériques
    # ------------------------------------------------------------------
    print("[3/7] Transformation des dates en métriques numériques...")
    df["Age"] = compute_age(df["DOB"], reference_date)
    df["TenureYears"] = compute_tenure_years(df["DateofHire"], reference_date)
    df["DaysSinceLastReview"] = compute_days_since_review(
        df["LastPerformanceReview_Date"], reference_date
    )
    df = df.drop(columns=["DOB", "DateofHire", "LastPerformanceReview_Date"])
    print("      DOB → Age, DateofHire → TenureYears, LastPerformanceReview_Date → DaysSinceLastReview")

    # ------------------------------------------------------------------
    # Étape 3 : Suppression des colonnes redondantes (ID numériques)
    # ------------------------------------------------------------------
    print("[4/7] Suppression des colonnes ID redondantes...")
    cols_redundant = [c for c in REDUNDANT_ID_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_redundant)
    print(f"      Colonnes supprimées : {cols_redundant}")

    # ------------------------------------------------------------------
    # Étape 4 : Suppression des colonnes non disponibles en production
    # ------------------------------------------------------------------
    print("[5/7] Suppression des colonnes causant du data leakage...")
    cols_leakage = [c for c in COLUMNS_TO_DROP_AFTER_TRANSFORM if c in df.columns]
    df = df.drop(columns=cols_leakage)
    print(f"      Colonnes supprimées : {cols_leakage}")

    # Colonne cible : Termd (1 = parti, 0 = encore en poste)
    target = df["Termd"].copy()

    # ------------------------------------------------------------------
    # Enregistrement du CSV anonymisé lisible (avant encodage)
    # ------------------------------------------------------------------
    print(f"      → Sauvegarde de '{output_anonymized}' ({len(df)} lignes × {len(df.columns)} colonnes)")
    df.to_csv(output_anonymized, index=False)

    # ------------------------------------------------------------------
    # Étape 5 : Traitement des valeurs manquantes
    # ------------------------------------------------------------------
    print("[6/7] Traitement des valeurs manquantes...")

    # Feedback_RH et Internal_Transfer_Request : colonnes texte libre supprimées
    # car non structurées pour le ML classique et potentiellement identifiantes
    for free_text_col in ("Feedback_RH", "Internal_Transfer_Request"):
        if free_text_col in df.columns:
            df = df.drop(columns=[free_text_col])
            print(f"      '{free_text_col}' supprimé (texte libre, non structuré).")

    # Colonnes numériques : imputation par la médiane
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      '{col}' : NaN remplacés par médiane ({median_val:.2f})")

    # Colonnes catégorielles : imputation par le mode
    cat_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"      '{col}' : NaN remplacés par mode ('{mode_val}')")

    # ------------------------------------------------------------------
    # Étape 6 : Encodage One-Hot des variables catégorielles
    # ------------------------------------------------------------------
    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True, dtype=int)

    # ------------------------------------------------------------------
    # Étape 7 : Normalisation des variables numériques continues
    # ------------------------------------------------------------------
    cols_to_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # ------------------------------------------------------------------
    # Sauvegarde du CSV ML-ready
    # ------------------------------------------------------------------
    print(f"[7/7] Sauvegarde de '{output_ml_ready}' ({len(df)} lignes × {len(df.columns)} colonnes)...")
    df.to_csv(output_ml_ready, index=False)

    print("\n✅ Prétraitement terminé.")
    print(f"   • {output_anonymized}  → données anonymisées, lisibles par les RH")
    print(f"   • {output_ml_ready}    → données encodées et normalisées, prêtes pour ML")
    print(f"   • Cible : 'Termd'  —  0 = actif ({(target==0).sum()}), 1 = parti ({(target==1).sum()})")
    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
