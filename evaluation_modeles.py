"""
Projet B3 - Analyse de données
Backtesting de prévision de ventes par produit.

Modèle A : Holt-Winters trend additive + saisonnalité additive (seasonal_periods=12)
Modèle B : Holt-Winters trend additive uniquement (sans saisonnalité)
Split : train <= 2017-12, test = 2018
"""

import warnings
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ============================================================
# PARAMETRES
# ============================================================
MIN_HISTORIQUE = 36
TRAIN_FIN      = "2017-12-31"
TEST_DEBUT     = "2018-01-01"
TEST_FIN       = "2018-12-31"

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
    Cle_Produit  AS code_produit,
    Date_Facturation AS date_vente,
    Montant_HT   AS valeur
FROM ventes
WHERE Date_Facturation IS NOT NULL
  AND Montant_HT IS NOT NULL
"""
df = pd.read_sql(query, engine, parse_dates=["date_vente"])
print(f"{len(df)} lignes chargées")

df["periode"] = df["date_vente"].dt.to_period("M").dt.to_timestamp()
agg = df.groupby(["code_produit", "periode"])["valeur"].sum().reset_index()

# ============================================================
# 2. Produits éligibles : >= 36 mois de données au total
# ============================================================
nb_periodes = agg.groupby("code_produit")["periode"].nunique()
produits_eligibles = nb_periodes[nb_periodes >= MIN_HISTORIQUE].index.tolist()
print(
    f"{len(produits_eligibles)} produits éligibles (>= {MIN_HISTORIQUE} mois) "
    f"sur {agg['code_produit'].nunique()} ayant des ventes"
)

# ============================================================
# 3. Backtesting produit par produit
# ============================================================
resultats_eval   = []
resultats_courbes = []
ignores = 0
echecs  = 0

for i, code in enumerate(produits_eligibles, 1):
    serie = agg[agg["code_produit"] == code].set_index("periode")["valeur"]
    serie = serie.asfreq("MS", fill_value=0)

    train = serie[serie.index <= TRAIN_FIN]
    test  = serie[(serie.index >= TEST_DEBUT) & (serie.index <= TEST_FIN)]

    if len(test) == 0:
        ignores += 1
        continue

    horizon  = len(test)
    reelles  = test.values

    # -- Modèle A : trend + saisonnalité additive --
    prev_A = None
    r2_A   = float("nan")
    if len(train) >= 2 * 12:
        try:
            modele_A = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=12
            ).fit(optimized=True)
            prev_A = modele_A.forecast(horizon).values
            r2_A   = r2_score(reelles, prev_A)
        except Exception:
            pass

    # -- Modèle B : trend additive uniquement --
    prev_B = None
    r2_B   = float("nan")
    try:
        modele_B = ExponentialSmoothing(
            train, trend="add", seasonal=None
        ).fit(optimized=True)
        prev_B = modele_B.forecast(horizon).values
        r2_B   = r2_score(reelles, prev_B)
    except Exception:
        echecs += 1

    # Si aucun modèle n'a pu être entraîné, on passe
    if pd.isna(r2_A) and pd.isna(r2_B):
        echecs += 1
        continue

    # Meilleur modèle (NaN compte comme pire)
    if pd.isna(r2_A):
        gagnant = "B"
    elif pd.isna(r2_B):
        gagnant = "A"
    else:
        gagnant = "A" if r2_A >= r2_B else "B"

    resultats_eval.append({
        "code_produit"  : code,
        "modele_gagnant": gagnant,
        "r2_modele_A"   : round(float(r2_A), 6) if not pd.isna(r2_A) else None,
        "r2_modele_B"   : round(float(r2_B), 6) if not pd.isna(r2_B) else None,
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

    if i % 100 == 0:
        print(f"  {i}/{len(produits_eligibles)} traités")

print(
    f"Backtesting terminé — {len(resultats_eval)} produits évalués, "
    f"{ignores} ignorés (pas de données 2018), {echecs} échecs modèle"
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
nb_eval = len(df_eval)
r2_gagnant = df_eval.apply(
    lambda r: r["r2_modele_A"] if r["modele_gagnant"] == "A" else r["r2_modele_B"],
    axis=1,
)
r2_moyen = r2_gagnant.mean()
nb_A = (df_eval["modele_gagnant"] == "A").sum()
nb_B = (df_eval["modele_gagnant"] == "B").sum()

print()
print("===== RÉSUMÉ =====")
print(f"Produits évalués          : {nb_eval}")
print(f"R² moyen (meilleur modèle): {r2_moyen:.4f}")
print(f"Modèle A gagnant          : {nb_A}  ({100 * nb_A / nb_eval:.1f} %)")
print(f"Modèle B gagnant          : {nb_B}  ({100 * nb_B / nb_eval:.1f} %)")
