# Carnet de bord — Projet B3 Ynov

## Partie 2 : Prévisions Python

### 4.6 Mise à jour des consignes (session du professeur)

Suite aux consignes du professeur Florent Pradel lors d'une session récente,
la méthode de prévision a été entièrement revue.

| Point             | Ancienne approche                  | Nouvelle approche              |
|-------------------|------------------------------------|--------------------------------|
| Bibliothèque      | Holt-Winters (statsmodels)         | Prophet                        |
| Seuil éligibilité | 18 mois                            | 24 mois                        |
| Mois sans ventes  | Ignorés                            | Remplis avec 0                 |
| Périmètre         | Tous produits éligibles (1 986)    | Top 5 produits les plus vendus |
| Table de sortie   | previsions_ventes                  | previsions_ventes_prophet      |

Un nouveau script `previsions_ventes_prophet.py` a été créé en remplacement
de `previsions_ventes.py`.

### 4.7 Comparaison Holt-Winters vs Prophet

| Critère                  | Holt-Winters | Prophet         |
|--------------------------|--------------|-----------------|
| Gestion des zéros        | Mauvaise     | Native          |
| Ruptures de tendance     | Non détectées| Auto-détectées  |
| Séries irrégulières      | Fragile      | Robuste         |
| Vitesse                  | Rapide       | Plus lente      |
| Installation             | Triviale     | Quelques dépend.|

Le choix initial de Holt-Winters était justifié par des contraintes
d'environnement. Sur la qualité pure des prévisions pour des données B2B
mensuelles avec mois creux, Prophet est la méthode recommandée.

Leçon retenue : Prophet est plus adapté aux séries B2B irrégulières.
Le choix algorithmique doit être justifié dans le rapport technique
en soulignant la gestion native des zéros et la détection des ruptures
de tendance comme critères principaux de sélection.
