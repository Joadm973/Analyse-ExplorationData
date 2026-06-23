# =============================================================================
# previsions_ventes_prophet.py
# Projet B3 - Ynov | Prévisions ventes Top 5 produits avec Prophet
# =============================================================================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from prophet import Prophet
import logging
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- Paramètres configurables ---
HOST           = "127.0.0.1"
PORT           = 3306
USER           = "root"
PASSWORD       = "azerty"
DATABASE       = "data"
HORIZON        = 6
MIN_HISTORIQUE = 24
TOP_N          = 5
METRIC         = "Quantite"
TABLE_SORTIE   = "previsions_ventes_prophet"

# =============================================================================
# 1. CONNEXION ET CHARGEMENT
# =============================================================================
print("Connexion à MySQL...")
engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}",
    echo=False
)

query = f"""
    SELECT
        Cle_Produit,
        DATE_FORMAT(Date_Facturation, '%%Y-%%m-01') AS periode,
        SUM({METRIC}) AS valeur
    FROM ventes
    WHERE Date_Facturation IS NOT NULL
    GROUP BY Cle_Produit, periode
"""

print("Chargement des ventes depuis MySQL...")
df = pd.read_sql(query, engine)
df['periode'] = pd.to_datetime(df['periode'])
print(f"  {len(df):,} lignes chargées ({df['Cle_Produit'].nunique():,} produits distincts)")

# =============================================================================
# 2. IDENTIFICATION DU TOP 5
# =============================================================================
top5 = (
    df.groupby('Cle_Produit')['valeur']
    .sum()
    .nlargest(TOP_N)
    .index.tolist()
)

print(f"\nTop {TOP_N} produits les plus vendus (par {METRIC}) :")
for i, code in enumerate(top5, 1):
    total = df[df['Cle_Produit'] == code]['valeur'].sum()
    print(f"  {i}. {code}  —  total {METRIC} : {total:,.0f}")

# =============================================================================
# 3. CONSTRUCTION DES SÉRIES AVEC ZÉROS POUR MOIS MANQUANTS
# =============================================================================
date_min = df['periode'].min()
date_max = df['periode'].max()
all_months = pd.date_range(start=date_min, end=date_max, freq='MS')

print(f"\nPlage temporelle : {date_min.strftime('%Y-%m')} à {date_max.strftime('%Y-%m')} "
      f"({len(all_months)} mois)")

# =============================================================================
# 4. PRÉVISIONS PROPHET PRODUIT PAR PRODUIT
# =============================================================================
results = []

for i, code in enumerate(top5, 1):
    print(f"\n[{i}/{TOP_N}] Produit {code}")

    serie = (
        df[df['Cle_Produit'] == code]
        .set_index('periode')['valeur']
        .reindex(all_months, fill_value=0)
    )

    nb_points = len(serie)
    print(f"  Points disponibles : {nb_points} mois")

    if nb_points < MIN_HISTORIQUE:
        print(f"  !! Ignoré (seuil minimum : {MIN_HISTORIQUE} mois)")
        continue

    prophet_df = pd.DataFrame({
        'ds': serie.index,
        'y':  serie.values.astype(float)
    })

    # Silence Prophet/cmdstanpy
    logging.disable(logging.INFO)
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model.fit(prophet_df)
    finally:
        logging.disable(logging.NOTSET)

    future = model.make_future_dataframe(periods=HORIZON, freq='MS')
    forecast = model.predict(future)

    forecast_future = (
        forecast[forecast['ds'] > date_max][['ds', 'yhat']]
        .copy()
        .rename(columns={'ds': 'date', 'yhat': 'valeur_prevue'})
    )

    forecast_future['valeur_prevue'] = forecast_future['valeur_prevue'].clip(lower=0).round(2)
    forecast_future['Cle_Produit']  = code

    results.append(forecast_future)
    print(f"  {len(forecast_future)} prévisions générées")
    print(forecast_future[['date', 'valeur_prevue']].to_string(index=False))

# =============================================================================
# 5. CONSOLIDATION ET ÉCRITURE EN BASE
# =============================================================================
if not results:
    print("\nAucun produit éligible. Aucune donnée écrite.")
else:
    df_out = pd.concat(results, ignore_index=True)
    df_out = df_out[['Cle_Produit', 'date', 'valeur_prevue']]

    print(f"\n--- Résumé ---")
    print(f"Produits traités   : {df_out['Cle_Produit'].nunique()}")
    print(f"Lignes générées    : {len(df_out)}")
    print(f"Prévisions néga.   : {(df_out['valeur_prevue'] < 0).sum()}")

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_SORTIE}"))
        conn.commit()

    df_out.to_sql(TABLE_SORTIE, engine, if_exists='replace', index=False)
    print(f"\nTable '{TABLE_SORTIE}' écrite avec succès dans la base '{DATABASE}'.")

    print("\nAperçu des données :")
    print(df_out.to_string(index=False))
