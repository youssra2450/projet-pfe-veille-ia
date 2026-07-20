"""
Script pour exécuter UNIQUEMENT le backtesting
Sans refaire les prédictions (utilise les données déjà préparées)
Version corrigée pour les index en double
"""

import logging
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

print("="*60)
print(" BACKTESTING UNIQUEMENT (sans refaire les prédictions)")
print("="*60)

# 1. Charger les données
try:
    ts = pd.read_parquet("data/processed/topic_timeseries.parquet")
    print(f"✅ Données chargées : {len(ts)} périodes × {len(ts.columns)} topics")
except FileNotFoundError:
    print("❌ Fichier topic_timeseries.parquet non trouvé !")
    exit(1)

# 2. Fonction de backtesting corrigée
def compute_metrics_manual(y_true, y_pred):
    """Calcule les métriques manuellement."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filtrer les valeurs nulles
    mask = y_true != 0
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    
    y_true_m, y_pred_m = y_true[mask], y_pred[mask]
    
    if len(y_true_m) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    
    mae = np.mean(np.abs(y_true_m - y_pred_m))
    rmse = np.sqrt(np.mean((y_true_m - y_pred_m) ** 2))
    
    # R²
    ss_res = np.sum((y_true_m - y_pred_m) ** 2)
    ss_tot = np.sum((y_true_m - np.mean(y_true_m)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {"mae": mae, "rmse": rmse, "r2": r2}

def safe_backtest(timeseries, method='prophet'):
    """Backtesting robuste avec correction des index en double."""
    results = []
    total = len(timeseries.columns)
    
    for idx, col in enumerate(timeseries.columns):
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
            
            if method == 'prophet':
                try:
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
                    
                    # Aligner les dates
                    fc_dates = pd.to_datetime(fc['ds'])
                    common = test.index.intersection(fc_dates)
                    
                    if len(common) > 1:
                        y_true = test.loc[common].values
                        y_pred = fc[fc_dates.isin(common)]['yhat'].values
                        metrics = compute_metrics_manual(y_true, y_pred)
                        if not np.isnan(metrics['mae']):
                            results.append({
                                'topic_id': col,
                                'method': method,
                                **metrics
                            })
                except Exception as e:
                    continue
                    
            elif method == 'arima':
                try:
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
                    metrics = compute_metrics_manual(test.values, test_pred)
                    if not np.isnan(metrics['mae']):
                        results.append({
                            'topic_id': col,
                            'method': method,
                            **metrics
                        })
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
        
        # Progression
        if (idx + 1) % 100 == 0:
            print(f"  Progression : {idx+1}/{total} topics")
    
    return pd.DataFrame(results)

# 3. Exécuter le backtesting
print("\n" + "-"*40)
print("BACKTESTING")
print("-"*40)

results = {}

# Option : limiter à un échantillon pour tester rapidement
SAMPLE_SIZE = 20  # Mettre à 0 pour TOUS les topics

if SAMPLE_SIZE > 0:
    ts_sample = ts.iloc[:, :SAMPLE_SIZE]
    print(f"🔬 Backtesting sur {SAMPLE_SIZE} topics (échantillon)")
else:
    ts_sample = ts
    print(f"🔬 Backtesting sur TOUS les {len(ts.columns)} topics")

# Prophet
try:
    print("\n📊 PROPHET:")
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
    print(f"  ❌ Erreur : {str(e)[:150]}")

# ARIMA
try:
    print("\n📊 ARIMA:")
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
    print(f"  ❌ Erreur : {str(e)[:150]}")

# 4. Résumé final
print("\n" + "="*60)
print(" RÉSUMÉ")
print("="*60)

if results:
    print("\n📊 Métriques du backtesting:")
    for method, df in results.items():
        print(f"\n{method.upper()}:")
        if 'mae' in df.columns and not df['mae'].isna().all():
            print(f"  MAE moyenne : {df['mae'].mean():.6f} (±{df['mae'].std():.6f})")
        if 'rmse' in df.columns and not df['rmse'].isna().all():
            print(f"  RMSE moyenne : {df['rmse'].mean():.6f}")
        if 'r2' in df.columns and not df['r2'].isna().all():
            print(f"  R² moyen : {df['r2'].mean():.4f}")
else:
    print("  Aucun résultat de backtesting disponible.")

print("\n✅ Backtesting terminé !")