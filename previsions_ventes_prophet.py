# =============================================================================
# previsions_ventes_prophet.py
# Projet B3 - Ynov | Prévisions ventes Top 5 produits avec Prophet
# =============================================================================

import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
import logging
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

HOST           = "127.0.0.1"
PORT           = 3306
USER           = "root"
PASSWORD       = "azerty"
DATABASE       = "data"
TABLE_SORTIE   = "previsions_ventes_prophet"
DATE_MAX_HIST  = "2018-12-31"
DATE_MIN_HIST  = "2015-01-01"
HORIZON_START  = "2019-01-01"
HORIZON_END    = "2019-12-01"
MIN_HISTORIQUE = 24
TOP_N          = 5

# =============================================================================
# 1. CONNEXION ET CHARGEMENT
# =============================================================================
print("Connexion à MySQL...")
engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}",
    echo=False
)

query = """
    SELECT
        Cle_Produit,
        Date_Facturation,
        Montant_HT
    FROM ventes
    WHERE Date_Facturation IS NOT NULL
      AND Date_Facturation <= '2018-12-31'
"""

print("Chargement des ventes (filtre <= 2018-12-31)...")
df = pd.read_sql(query, engine)
df['Date_Facturation'] = pd.to_datetime(df['Date_Facturation'])
df['Montant_HT'] = pd.to_numeric(df['Montant_HT'], errors='coerce').fillna(0)
print(f"  {len(df):,} lignes chargées ({df['Cle_Produit'].nunique():,} produits distincts)")

# =============================================================================
# 2. IDENTIFICATION DU TOP 5 (sur période 2015-2018)
# =============================================================================
df_hist = df[
    (df['Date_Facturation'] >= DATE_MIN_HIST) &
    (df['Date_Facturation'] <= DATE_MAX_HIST)
].copy()

top5 = (
    df_hist.groupby('Cle_Produit')['Montant_HT']
    .sum()
    .nlargest(TOP_N)
    .index.tolist()
)

print(f"\nTop {TOP_N} produits (Montant_HT 2015-2018) :")
for i, code in enumerate(top5, 1):
    total = df_hist[df_hist['Cle_Produit'] == code]['Montant_HT'].sum()
    print(f"  {i}. {code}  —  total : {total:,.2f}")

# =============================================================================
# 3. PRÉVISIONS PROPHET PAR PRODUIT
# =============================================================================
all_months = pd.date_range(start=DATE_MIN_HIST, end=DATE_MAX_HIST, freq='MS')
forecast_months = pd.date_range(start=HORIZON_START, end=HORIZON_END, freq='MS')

results: list[pd.DataFrame] = []

for i, code in enumerate(top5, 1):
    print(f"\n[{i}/{TOP_N}] Produit {code}")

    monthly = (
        df_hist[df_hist['Cle_Produit'] == code]
        .groupby(pd.Grouper(key='Date_Facturation', freq='MS'))['Montant_HT']
        .sum()
    )

    serie = monthly.reindex(all_months, fill_value=0)
    nb_points = (serie > 0).sum()
    print(f"  Mois avec ventes : {nb_points} / {len(serie)}")

    if len(serie) < MIN_HISTORIQUE:
        print(f"  !! Ignoré (seuil minimum : {MIN_HISTORIQUE} mois)")
        continue

    prophet_df = pd.DataFrame({'ds': serie.index, 'y': serie.values.astype(float)})

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

    future = pd.DataFrame({'ds': pd.date_range(start=DATE_MIN_HIST, end=HORIZON_END, freq='MS')})
    forecast = model.predict(future)

    forecast_out = (
        forecast[forecast['ds'].isin(forecast_months)][['ds', 'yhat']]
        .copy()
        .rename(columns={'ds': 'date', 'yhat': 'valeur_prevue'})
    )
    forecast_out['valeur_prevue'] = forecast_out['valeur_prevue'].clip(lower=0).round(2)
    forecast_out['Code_Produit'] = code

    results.append(forecast_out)
    print(f"  {len(forecast_out)} prévisions générées (2019-01 à 2019-12)")

# =============================================================================
# 4. CONSOLIDATION ET ÉCRITURE EN BASE
# =============================================================================
if not results:
    print("\nAucun produit éligible. Aucune donnée écrite.")
else:
    df_out = pd.concat(results, ignore_index=True)[['Code_Produit', 'date', 'valeur_prevue']]

    df_out.to_sql(TABLE_SORTIE, engine, if_exists='replace', index=False)

    print(f"\n--- Résumé ---")
    print(f"Produits retenus   : {df_out['Code_Produit'].unique().tolist()}")
    print(f"Lignes générées    : {len(df_out)}  (attendu : 60)")
    print(f"\nAperçu (premières lignes) :")
    print(df_out.head(15).to_string(index=False))
    print(f"\nTable '{TABLE_SORTIE}' écrite dans la base '{DATABASE}'.")
