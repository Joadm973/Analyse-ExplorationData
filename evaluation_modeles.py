"""
Projet B3 - Analyse de données
Backtesting de prévision de ventes par produit.

Modèle A : Prophet (si installé) OU Holt-Winters saisonnalité additive
Modèle B : Holt-Winters trend additive uniquement (sans saisonnalité)
Split : train <= 2017-12-31  |  test = 2018
"""

import warnings
import logging
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ============================================================
# Vérification Prophet
# ============================================================
try:
    from prophet import Prophet
    # Silence cmdstanpy (ajoute des handlers à chaque fit) via disable global
    pass
    PROPHET_OK = True
    NOM_MODELE_A = "Prophet"
    print("Prophet installé : Modèle A = Prophet")
except ImportError:
    PROPHET_OK = False
    NOM_MODELE_A = "HoltWinters_saisonnier"
    print("Prophet non disponible : Modèle A = Holt-Winters saisonnier (fallback)")

# ============================================================
# PARAMETRES
# ============================================================
MIN_HISTORIQUE = 36
TRAIN_FIN  = "2017-12-31"
TEST_DEBUT = "2018-01-01"
TEST_FIN   = "2018-12-31"

DB_USER = "root"
DB_PASS = "azerty"
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "data"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ============================================================
# 1. Charger et agréger par produit / mois
# ============================================================
print("Chargement des ventes...")
query = """
SELECT
    Cle_Produit      AS code_produit,
    Date_Facturation AS date_vente,
    Montant_HT       AS valeur
FROM ventes
WHERE Date_Facturation IS NOT NULL
  AND Montant_HT IS NOT NULL
"""
df = pd.read_sql(query, engine, parse_dates=["date_vente"])
print(f"{len(df)} lignes chargées")

df["periode"] = df["date_vente"].dt.to_period("M").dt.to_timestamp()
agg = df.groupby(["code_produit", "periode"])["valeur"].sum().reset_index()

# ============================================================
# 2. Filtrage d'éligibilité (4 critères successifs)
# ============================================================
MIN_MOIS_NON_NULS  = 6   # mois non nuls sur les 18 derniers mois avant coupure
MIN_MOIS_ACTIF     = 1   # ventes sur les 6 derniers mois avant coupure
MIN_MOIS_TEST      = 6   # mois réels en 2018 pour un R² significatif

refus = {
    "historique_insuffisant": 0,
    "trop_de_zeros": 0,
    "produit_inactif": 0,
    "test_trop_court": 0,
}

tous_les_produits = agg["code_produit"].unique().tolist()
produits_eligibles = []

for code in tous_les_produits:
    serie = agg[agg["code_produit"] == code].set_index("periode")["valeur"]
    serie = serie.asfreq("MS", fill_value=0).sort_index()

    train = serie[serie.index <= TRAIN_FIN]
    test  = serie[(serie.index >= TEST_DEBUT) & (serie.index <= TEST_FIN)]

    # Critère 1 : >= 36 mois d'historique total (train + éventuellement test)
    if len(serie) < MIN_HISTORIQUE:
        refus["historique_insuffisant"] += 1
        continue

    # Critère 2 : >= 6 mois non nuls sur les 18 derniers mois du train
    derniers_18_train = train.iloc[-18:]
    if (derniers_18_train > 0).sum() < MIN_MOIS_NON_NULS:
        refus["trop_de_zeros"] += 1
        continue

    # Critère 3 : >= 1 vente sur les 6 derniers mois du train (produit actif)
    derniers_6_train = train.iloc[-6:]
    if (derniers_6_train > 0).sum() < MIN_MOIS_ACTIF:
        refus["produit_inactif"] += 1
        continue

    # Critère 4 : >= 6 mois réels en 2018 (test assez long pour R²)
    if (test > 0).sum() < MIN_MOIS_TEST:
        refus["test_trop_court"] += 1
        continue

    produits_eligibles.append(code)

print(
    f"{len(produits_eligibles)} produits éligibles "
    f"sur {len(tous_les_produits)} ayant des ventes"
)

# ============================================================
# Helpers d'entraînement
# ============================================================

def fit_prophet(train: pd.Series, horizon: int) -> pd.Series | None:
    """Entraîne Prophet et retourne les prévisions sur `horizon` mois."""
    try:
        df_train = pd.DataFrame({
            "ds": train.index,
            "y": train.values,
        })
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive")
        # Désactive les logs cmdstanpy pendant le fit (réinstalle ses handlers à chaque appel)
        logging.disable(logging.INFO)
        try:
            m.fit(df_train)
        finally:
            logging.disable(logging.NOTSET)
        future = m.make_future_dataframe(periods=horizon, freq="MS",
                                         include_history=False)
        forecast = m.predict(future)
        return forecast["yhat"].clip(lower=0).values
    except Exception:
        return None


def fit_hw_saisonnier(train: pd.Series, horizon: int) -> pd.Series | None:
    """Holt-Winters trend + saisonnalité additive."""
    if len(train) < 2 * 12:
        return None
    try:
        m = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=12
        ).fit(optimized=True)
        return m.forecast(horizon).values
    except Exception:
        return None


def fit_hw_trend(train: pd.Series, horizon: int) -> pd.Series | None:
    """Holt-Winters trend additive uniquement."""
    try:
        m = ExponentialSmoothing(
            train, trend="add", seasonal=None
        ).fit(optimized=True)
        return m.forecast(horizon).values
    except Exception:
        return None

# ============================================================
# 3. Backtesting
# ============================================================
resultats_eval    = []
resultats_courbes = []
echecs = 0

for i, code in enumerate(produits_eligibles, 1):
    serie = agg[agg["code_produit"] == code].set_index("periode")["valeur"]
    serie = serie.asfreq("MS", fill_value=0)

    train   = serie[serie.index <= TRAIN_FIN]
    test    = serie[(serie.index >= TEST_DEBUT) & (serie.index <= TEST_FIN)]
    horizon = len(test)
    reelles = test.values

    # Modèle A
    if PROPHET_OK:
        prev_A = fit_prophet(train, horizon)
    else:
        prev_A = fit_hw_saisonnier(train, horizon)

    r2_A = r2_score(reelles, prev_A) if prev_A is not None else float("nan")

    # Modèle B
    prev_B = fit_hw_trend(train, horizon)
    r2_B   = r2_score(reelles, prev_B) if prev_B is not None else float("nan")

    if pd.isna(r2_A) and pd.isna(r2_B):
        echecs += 1
        continue

    if pd.isna(r2_A):
        gagnant = "B"
    elif pd.isna(r2_B):
        gagnant = "A"
    else:
        gagnant = "A" if r2_A >= r2_B else "B"

    resultats_eval.append({
        "code_produit"  : code,
        "modele_A"      : NOM_MODELE_A,
        "modele_B"      : "HoltWinters_trend",
        "r2_modele_A"   : round(float(r2_A), 6) if not pd.isna(r2_A) else None,
        "r2_modele_B"   : round(float(r2_B), 6) if not pd.isna(r2_B) else None,
        "modele_gagnant": gagnant,
        "nb_mois_train" : len(train),
        "nb_mois_test"  : horizon,
    })

    vals_A = prev_A.tolist() if prev_A is not None else [None] * horizon
    vals_B = prev_B.tolist() if prev_B is not None else [None] * horizon

    for date, reel, pa, pb in zip(test.index, reelles, vals_A, vals_B):
        resultats_courbes.append({
            "code_produit"    : code,
            "date"            : date.date(),
            "valeur_reelle"   : round(float(reel), 2),
            "valeur_predite_A": round(float(pa), 2) if pa is not None else None,
            "valeur_predite_B": round(float(pb), 2) if pb is not None else None,
        })

    if i % 50 == 0:
        print(f"  {i}/{len(produits_eligibles)} traités")

print(
    f"Backtesting terminé — {len(resultats_eval)} produits évalués, "
    f"{echecs} échecs"
)

# ============================================================
# 4. Écriture SQL
# ============================================================
df_eval    = pd.DataFrame(resultats_eval)
df_courbes = pd.DataFrame(resultats_courbes)

df_eval.to_sql("evaluation_modeles", engine, if_exists="replace", index=False)
print("Table 'evaluation_modeles' écrite")

df_courbes.to_sql("courbes_evaluation", engine, if_exists="replace", index=False)
print("Table 'courbes_evaluation' écrite")

# ============================================================
# 5. Résumé
# ============================================================
nb_eval  = len(df_eval)
nb_A     = (df_eval["modele_gagnant"] == "A").sum()
nb_B     = (df_eval["modele_gagnant"] == "B").sum()
r2_moy_A = df_eval["r2_modele_A"].mean()
r2_moy_B = df_eval["r2_modele_B"].mean()
r2_med_A = df_eval["r2_modele_A"].median()
r2_med_B = df_eval["r2_modele_B"].median()

print()
print("===== FILTRAGE D'ÉLIGIBILITÉ =====")
print(f"Produits avec des ventes                              : {len(tous_les_produits)}")
print(f"Refusés — historique insuffisant (< {MIN_HISTORIQUE} mois)       : {refus['historique_insuffisant']}")
print(f"Refusés — trop de zéros (< {MIN_MOIS_NON_NULS} mois non nuls / 18 avant 2018) : {refus['trop_de_zeros']}")
print(f"Refusés — produit inactif (0 vente sur 6 mois avant 2018)   : {refus['produit_inactif']}")
print(f"Refusés — test trop court (< {MIN_MOIS_TEST} mois non nuls en 2018)    : {refus['test_trop_court']}")
print(f"Éligibles final                                       : {len(produits_eligibles)}")
print()
print("===== RÉSUMÉ BACKTESTING =====")
print(f"Prophet installé          : {'Oui' if PROPHET_OK else 'Non (fallback HW saisonnier)'}")
print(f"Produits évalués          : {nb_eval}")
print(f"R² moyen   — Modèle A ({NOM_MODELE_A:<26}): {r2_moy_A:.4f}  (médiane {r2_med_A:.4f})")
print(f"R² moyen   — Modèle B (HoltWinters_trend          ): {r2_moy_B:.4f}  (médiane {r2_med_B:.4f})")
print(f"Modèle A gagnant          : {nb_A}  ({100 * nb_A / nb_eval:.1f} %)")
print(f"Modèle B gagnant          : {nb_B}  ({100 * nb_B / nb_eval:.1f} %)")
