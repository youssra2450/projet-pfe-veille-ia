"""
Script pour exécuter UNIQUEMENT le backtesting
Sans refaire les 3 heures de prédictions
Version avec correction des index en double
"""

import logging
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

# Importer les classes nécessaires
from prediction_models import TrendPredictor, PROPHET_AVAILABLE, PMDARIMA_AVAILABLE, compute_metrics

print("="*60)
print(" BACKTESTING UNIQUEMENT (sans prédictions)")
print("="*60)

# 1. Charger les données
try:
    ts = pd.read_parquet("data/processed/topic_timeseries.parquet")
    print(f"✅ Données chargées : {len(ts)} périodes × {len(ts.columns)} topics")
except FileNotFoundError:
    print("❌ Fichier topic_timeseries.parquet non trouvé !")
    exit(1)

# 2. Sélectionner un échantillon
SAMPLE_SIZE = 10
if SAMPLE_SIZE > 0:
    topics_to_test = ts.columns[:SAMPLE_SIZE].tolist()
    ts_sample = ts[topics_to_test]
    print(f"🔬 Backtesting sur {SAMPLE_SIZE} topics : {topics_to_test}")
else:
    ts_sample = ts
    print(f"🔬 Backtesting sur TOUS les {len(ts.columns)} topics")

# 3. Fonction de backtesting corrigée (sans utiliser celle de la classe)
def safe_backtest(timeseries, method='prophet'):
    """Backtesting robuste avec correction des index en double."""
    results = []
    
    for col in timeseries.columns:
        try:
            series = timeseries[col].dropna()
            
            # 🔧 CORRECTION : Réindexer avec des dates uniques
            if not series.index.is_unique:
                series = series.groupby(series.index).mean()
            
            if len(series) < 10:
                continue
            
            # Split 70/30
            split = int(len(series) * 0.7)
            train = series.iloc[:split]
            test = series.iloc[split:]
            
            # 🔧 S'assurer que l'index est unique
            if not train.index.is_unique:
                train = train.groupby(train.index).mean()
            if not test.index.is_unique:
                test = test.groupby(test.index).mean()
            
            # Appeler la méthode appropriée
            if method == 'prophet' and PROPHET_AVAILABLE:
                from prophet import Prophet
                import logging as _log
                _log.getLogger("prophet").setLevel(_log.WARNING)
                _log.getLogger("cmdstanpy").setLevel(_log.WARNING)
                
                df_train = pd.DataFrame({"ds": train.index, "y": train.values})
                
                # 🔧 Supprimer les doublons dans les dates
                if df_train['ds'].duplicated().any():
                    df_train = df_train.groupby('ds', as_index=False).mean()
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                model.fit(df_train)
                
                future = model.make_future_dataframe(periods=len(test), freq='MS', include_history=False)
                fc = model.predict(future)
                
                common = test.index.intersection(pd.to_datetime(fc['ds']))
                if len(common) > 1:
                    y_true = test.loc[common].values
                    y_pred = fc[fc['ds'].isin(common)]['yhat'].values
                    metrics = compute_metrics(y_true, y_pred)
                    if not np.isnan(metrics['mae']):
                        results.append({
                            'topic_id': col,
                            'method': method,
                            **metrics
                        })
                        
            elif method == 'arima' and PMDARIMA_AVAILABLE:
                import pmdarima as pm
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = pm.auto_arima(
                        train.values,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        random_state=42
                    )
                
                test_pred = model.predict(n_periods=len(test))
                common = test.index
                if len(common) > 1:
                    metrics = compute_metrics(test.values, test_pred)
                    if not np.isnan(metrics['mae']):
                        results.append({
                            'topic_id': col,
                            'method': method,
                            **metrics
                        })
                        
        except Exception as e:
            # Ignorer silencieusement les erreurs
            continue
    
    return pd.DataFrame(results)

# 4. Exécuter le backtesting
print("\n" + "-"*40)
print("BACKTESTING")
print("-"*40)

results = {}

if PROPHET_AVAILABLE:
    print("\n📊 Prophet:")
    try:
        bt = safe_backtest(ts_sample, method='prophet')
        if not bt.empty and 'mae' in bt.columns:
            results['prophet'] = bt
            print(f"  ✅ MAE moyenne : {bt['mae'].mean():.6f} (±{bt['mae'].std():.6f})")
            if 'rmse' in bt.columns:
                print(f"  ✅ RMSE moyenne : {bt['rmse'].mean():.6f}")
            print(f"  ✅ Nombre de résultats : {len(bt)}")
        else:
            print("  ⚠️ Aucun résultat")
    except Exception as e:
        print(f"  ❌ Erreur : {str(e)[:200]}")

if PMDARIMA_AVAILABLE:
    print("\n📊 ARIMA:")
    try:
        bt = safe_backtest(ts_sample, method='arima')
        if not bt.empty and 'mae' in bt.columns:
            results['arima'] = bt
            print(f"  ✅ MAE moyenne : {bt['mae'].mean():.6f} (±{bt['mae'].std():.6f})")
            if 'rmse' in bt.columns:
                print(f"  ✅ RMSE moyenne : {bt['rmse'].mean():.6f}")
            print(f"  ✅ Nombre de résultats : {len(bt)}")
        else:
            print("  ⚠️ Aucun résultat")
    except Exception as e:
        print(f"  ❌ Erreur : {str(e)[:200]}")

# 5. Résumé final
print("\n" + "="*60)
print(" RÉSUMÉ")
print("="*60)

for method, df in results.items():
    print(f"\n{method.upper()}:")
    if 'mae' in df.columns:
        print(f"  MAE moyenne : {df['mae'].mean():.6f} (±{df['mae'].std():.6f})")
    if 'rmse' in df.columns and not df['rmse'].isna().all():
        print(f"  RMSE moyenne : {df['rmse'].mean():.6f}")
    if 'r2' in df.columns and not df['r2'].isna().all():
        print(f"  R² moyen : {df['r2'].mean():.4f}")

print("\n✅ Backtesting terminé !")