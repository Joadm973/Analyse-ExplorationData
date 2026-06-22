"""
Projet B3 - Analyse de données
Partie 2 : prévision Python

Génère une prévision de ventes pour chaque produit éligible,
horizon et fréquence paramétrables. Écrit le résultat dans une
table SQL (previsions_ventes) contenant : paramètres, saisonnalité,
dates, valeurs.
"""

import warnings
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ============================================================
# PARAMETRES (à ajuster selon le besoin)
# ============================================================
HORIZON = 6                 # nb de périodes à prévoir
FREQUENCE = "M"              # "D" jour, "W" semaine, "M" mois
MIN_HISTORIQUE = 18          # nb minimum de périodes pour qu'un produit soit éligible
METRIC = "Quantite"          # "Quantite" ou "Montant_HT"

SEASONAL_PERIODS = {"D": 365, "W": 52, "M": 12}[FREQUENCE]
PANDAS_FREQ = {"D": "D", "W": "W", "M": "MS"}[FREQUENCE]

# Connexion à la base
DB_USER = "root"
DB_PASS = "azerty"
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "data"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ============================================================
# 1. Charger les ventes
# ============================================================
print("Chargement des ventes...")
query = f"""
SELECT
    Cle_Produit AS code_produit,
    Date_Facturation AS date_vente,
    {METRIC} AS valeur
FROM ventes
WHERE Date_Facturation IS NOT NULL
  AND {METRIC} IS NOT NULL
"""
df = pd.read_sql(query, engine, parse_dates=["date_vente"])
print(f"{len(df)} lignes chargées")

# ============================================================
# 2. Agréger à la fréquence choisie
# ============================================================
df["periode"] = df["date_vente"].dt.to_period(FREQUENCE).dt.to_timestamp()
agg = df.groupby(["code_produit", "periode"])["valeur"].sum().reset_index()

# ============================================================
# 3. Déterminer les produits éligibles (filtres successifs)
# ============================================================
MIN_MOIS_NON_NULS   = 6   # sur les 18 derniers mois
MIN_MOIS_ACTIF      = 1   # sur les 6 derniers mois

refus = {
    "historique_insuffisant": 0,
    "trop_de_zeros": 0,
    "produit_inactif": 0,
}

tous_les_produits = agg["code_produit"].unique().tolist()
produits_eligibles = []

for code in tous_les_produits:
    serie = agg[agg["code_produit"] == code].set_index("periode")["valeur"]
    serie = serie.asfreq(PANDAS_FREQ, fill_value=0).sort_index()

    # Critère 1 : historique minimum
    if len(serie) < MIN_HISTORIQUE:
        refus["historique_insuffisant"] += 1
        continue

    # Critère 2 : au moins 6 mois non nuls sur les 18 derniers mois
    derniers_18 = serie.iloc[-18:]
    if (derniers_18 > 0).sum() < MIN_MOIS_NON_NULS:
        refus["trop_de_zeros"] += 1
        continue

    # Critère 3 : au moins 1 vente sur les 6 derniers mois
    derniers_6 = serie.iloc[-6:]
    if (derniers_6 > 0).sum() < MIN_MOIS_ACTIF:
        refus["produit_inactif"] += 1
        continue

    produits_eligibles.append(code)

print(f"{len(produits_eligibles)} produits éligibles sur {len(tous_les_produits)} "
      f"ayant des ventes (fréquence = {FREQUENCE})")

# ============================================================
# 4. Prévision produit par produit
# ============================================================
resultats = []

for i, code_produit in enumerate(produits_eligibles, 1):
    serie = agg[agg["code_produit"] == code_produit].set_index("periode")["valeur"]
    serie = serie.asfreq(PANDAS_FREQ, fill_value=0)

    try:
        try:
            modele = ExponentialSmoothing(
                serie, trend="add", seasonal="add",
                seasonal_periods=SEASONAL_PERIODS
            ).fit()
            saisonnalite = "additive"
        except Exception:
            modele = ExponentialSmoothing(serie, trend="add", seasonal=None).fit()
            saisonnalite = "aucune"

        prevision = modele.forecast(HORIZON)
        dernieres_dates = pd.date_range(
            start=serie.index[-1], periods=HORIZON + 1, freq=PANDAS_FREQ
        )[1:]

        for date_prev, valeur in zip(dernieres_dates, prevision):
            resultats.append({
                "code_produit": code_produit,
                "frequence": FREQUENCE,
                "horizon": HORIZON,
                "metrique": METRIC,
                "saisonnalite": saisonnalite,
                "date_prevision": date_prev.date(),
                "valeur_prevue": max(0, round(float(valeur), 2)),
            })

    except Exception as e:
        print(f"Échec pour {code_produit}: {e}")

    if i % 50 == 0:
        print(f"{i}/{len(produits_eligibles)} produits traités")

df_resultats = pd.DataFrame(resultats)

# ============================================================
# 5. Écriture dans la base SQL
# ============================================================
df_resultats.to_sql("previsions_ventes", engine, if_exists="replace", index=False)

# ============================================================
# 6. Résumé des refus et résultat final
# ============================================================
print()
print("===== FILTRAGE D'ÉLIGIBILITÉ =====")
print(f"Produits avec des ventes                          : {len(tous_les_produits)}")
print(f"Refusés — historique insuffisant (< {MIN_HISTORIQUE} mois)   : {refus['historique_insuffisant']}")
print(f"Refusés — trop de zéros (< {MIN_MOIS_NON_NULS} mois non nuls / 18) : {refus['trop_de_zeros']}")
print(f"Refusés — produit inactif (0 vente sur 6 derniers mois) : {refus['produit_inactif']}")
print(f"Éligibles final                                   : {len(produits_eligibles)}")
print()
print(f"Lignes écrites dans previsions_ventes             : {len(df_resultats)}")
print("Table 'previsions_ventes' écrite dans MySQL")
