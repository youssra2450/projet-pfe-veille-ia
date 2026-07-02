"""
Agent 3 - Module de Prédiction Temporelle
Laboratoire de Recherche en IA - Projet Veille Technologique

Modèles : Prophet, ARIMA/SARIMA, LSTM (optionnel), Ensemble Learning
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
SEED = 42
np.random.seed(SEED)

# Vérification de la disponibilité de TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow disponible - LSTM activé")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow non installé - LSTM désactivé")

# Vérification de Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet non installé")

# Vérification de pmdarima
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima non installé - ARIMA désactivé")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Ensemble complet de métriques de performance."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, 
                "smape": np.nan, "mase": np.nan, "r2": np.nan}
    
    y_true_m, y_pred_m = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(y_true_m, y_pred_m)
    rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
    mape = np.mean(np.abs((y_true_m - y_pred_m) / y_true_m)) * 100
    denominator = (np.abs(y_true_m) + np.abs(y_pred_m)) / 2
    smape = np.mean(np.abs(y_true_m - y_pred_m) / denominator) * 100
    
    if len(y_true_m) > 1:
        naive_error = np.mean(np.abs(np.diff(y_true_m)))
        mase = mae / naive_error if naive_error > 0 else np.nan
        ss_res = np.sum((y_true_m - y_pred_m) ** 2)
        ss_tot = np.sum((y_true_m - np.mean(y_true_m)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        mase, r2 = np.nan, np.nan
    
    return {"mae": round(mae, 5), "rmse": round(rmse, 5), 
            "mape": round(mape, 2), "smape": round(smape, 2),
            "mase": round(mase, 5) if not np.isnan(mase) else None,
            "r2": round(r2, 4) if not np.isnan(r2) else None}


class TrendPredictor:
    """
    Système de prédiction multi-modèles pour séries temporelles.
    Modèles disponibles : Prophet, ARIMA, LSTM (si TensorFlow installé)
    """
    
    def __init__(self, horizon_months: int = 12, test_ratio: float = 0.15,
                 use_deep_learning: bool = True, use_ensemble: bool = True):
        self.horizon = horizon_months
        self.test_ratio = test_ratio
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.use_ensemble = use_ensemble
        self._cache = {}

        # Log des modèles disponibles
        models_status = []
        if PROPHET_AVAILABLE:
            models_status.append("Prophet")
        if PMDARIMA_AVAILABLE:
            models_status.append("ARIMA")
        if self.use_deep_learning:
            models_status.append("LSTM")
        logger.info(f"Modèles disponibles : {', '.join(models_status) if models_status else 'Aucun'}")

    def _prepare_lstm_data(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prépare les données pour LSTM avec fenêtre de contexte."""
        values = series.fillna(0).values.reshape(-1, 1)
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)
        lookback = min(12, max(3, len(values_scaled) // 4))
        X, y = [], []
        for i in range(lookback, len(values_scaled)):
            X.append(values_scaled[i-lookback:i, 0])
            y.append(values_scaled[i, 0])
        return np.array(X), np.array(y), scaler

    def _build_lstm(self, input_shape: int):
        """Construit le modèle LSTM."""
        if not self.use_deep_learning or not TENSORFLOW_AVAILABLE:
            return None
        try:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(input_shape, 1)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            logger.warning(f"Erreur construction LSTM: {e}")
            return None

    def predict_lstm(self, series: pd.Series) -> Optional[Dict]:
        """Prédiction avec LSTM."""
        if not self.use_deep_learning or not TENSORFLOW_AVAILABLE:
            return None

        values = series.fillna(0).values
        n_test = max(2, int(len(values) * self.test_ratio))
        if len(values) < 8:
            return None

        X, y, scaler = self._prepare_lstm_data(series)
        if len(X) < n_test + 2:
            return None

        train_size = len(X) - n_test
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = self._build_lstm(X_train.shape[1])
        if model is None:
            return None

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                  callbacks=[early_stop], verbose=0)

        y_pred = model.predict(X_test, verbose=0).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_test_orig, y_pred_orig)

        last_seq = X_test[-1].reshape(1, -1, 1)
        future = []
        for _ in range(self.horizon):
            pred = model.predict(last_seq, verbose=0).flatten()[0]
            future.append(pred)
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = pred

        future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
        residuals = y_test_orig - y_pred_orig
        std_res = np.std(residuals) if len(residuals) > 1 else 0.1

        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=self.horizon + 1, freq="MS")[1:]

        return {"topic_id": series.name, "method": "LSTM",
                "forecast": pd.Series(future, index=future_dates),
                "lower": pd.Series(future - 1.96 * std_res, index=future_dates),
                "upper": pd.Series(future + 1.96 * std_res, index=future_dates),
                **metrics}

    def predict_arima(self, series: pd.Series) -> Optional[Dict]:
        """Prédiction avec ARIMA."""
        if not PMDARIMA_AVAILABLE:
            return None

        values = series.fillna(0).values
        n_test = max(2, int(len(values) * self.test_ratio))
        if len(values) < 6:
            return None

        train, test = values[:-n_test], values[-n_test:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(train, seasonal=False, stepwise=True,
                                  suppress_warnings=True, error_action="ignore",
                                  random_state=SEED)

        test_pred, _ = model.predict(n_periods=n_test, return_conf_int=True)
        metrics = compute_metrics(test, test_pred)

        forecast, conf_int = model.predict(n_periods=self.horizon, return_conf_int=True)
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=self.horizon + 1, freq="MS")[1:]

        return {"topic_id": series.name, "method": "ARIMA",
                "forecast": pd.Series(forecast, index=future_dates),
                "lower": pd.Series(conf_int[:, 0], index=future_dates),
                "upper": pd.Series(conf_int[:, 1], index=future_dates),
                **metrics}

    def predict_prophet(self, series: pd.Series) -> Optional[Dict]:
        """Prédiction avec Prophet."""
        if not PROPHET_AVAILABLE:
            return None

        values = series.fillna(0).values
        n_test = max(2, int(len(values) * self.test_ratio))
        if len(values) < 6:
            return None

        train_s, test_s = series.iloc[:-n_test], series.iloc[-n_test:]
        df_train = pd.DataFrame({"ds": train_s.index, "y": train_s.values})

        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                            daily_seasonality=False, changepoint_prior_scale=0.05)
            try:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            except:
                pass
            model.fit(df_train)

        future_test = model.make_future_dataframe(periods=n_test, freq='MS', include_history=False)
        fc_test = model.predict(future_test)
        metrics = compute_metrics(test_s.values, fc_test["yhat"].values)

        future = model.make_future_dataframe(periods=self.horizon, freq='MS', include_history=False)
        forecast = model.predict(future)
        future_dates = pd.to_datetime(forecast["ds"].values)

        return {"topic_id": series.name, "method": "Prophet",
                "forecast": pd.Series(forecast["yhat"].values, index=future_dates),
                "lower": pd.Series(forecast["yhat_lower"].values, index=future_dates),
                "upper": pd.Series(forecast["yhat_upper"].values, index=future_dates),
                **metrics}

    def ensemble_forecast(self, predictions: List[Dict]) -> Optional[Dict]:
        """Combine les prédictions par moyenne pondérée."""
        if not predictions or len(predictions) < 2:
            return predictions[0] if predictions else None

        weights = []
        for p in predictions:
            w = 1.0 / p.get('mae', 1.0) if p.get('mae', 0) > 0 else 0.1
            weights.append(w)
        weights = np.array(weights) / np.sum(weights)

        combined_forecast = np.zeros_like(predictions[0]['forecast'].values)
        combined_lower = np.zeros_like(predictions[0]['forecast'].values)
        combined_upper = np.zeros_like(predictions[0]['forecast'].values)

        for i, p in enumerate(predictions):
            combined_forecast += weights[i] * p['forecast'].values
            combined_lower += weights[i] * p.get('lower', p['forecast']).values
            combined_upper += weights[i] * p.get('upper', p['forecast']).values

        w_mae = np.sum([w * p.get('mae', 0) for w, p in zip(weights, predictions) if p.get('mae')])
        w_rmse = np.sum([w * p.get('rmse', 0) for w, p in zip(weights, predictions) if p.get('rmse')])

        return {"topic_id": predictions[0]['topic_id'], "method": "Ensemble",
                "forecast": pd.Series(combined_forecast, index=predictions[0]['forecast'].index),
                "lower": pd.Series(combined_lower, index=predictions[0]['forecast'].index),
                "upper": pd.Series(combined_upper, index=predictions[0]['forecast'].index),
                "mae": round(w_mae, 5), "rmse": round(w_rmse, 5),
                "weights": weights.tolist(), "models": [p.get('method') for p in predictions]}

    def predict_all(self, timeseries: pd.DataFrame, methods: List[str] = None) -> List[Dict]:
        """Prédiction avec tous les modèles disponibles."""
        if methods is None:
            methods = []
            if PROPHET_AVAILABLE:
                methods.append('prophet')
            if PMDARIMA_AVAILABLE:
                methods.append('arima')
            if self.use_deep_learning:
                methods.append('lstm')
            
            if not methods:
                logger.error("Aucun modèle disponible pour la prédiction")
                return []

        results = []
        from tqdm import tqdm

        method_map = {}
        if PROPHET_AVAILABLE:
            method_map['prophet'] = self.predict_prophet
        if PMDARIMA_AVAILABLE:
            method_map['arima'] = self.predict_arima
        if self.use_deep_learning:
            method_map['lstm'] = self.predict_lstm

        for col in tqdm(timeseries.columns, desc="Prédictions"):
            predictions = []
            for method in methods:
                if method in method_map:
                    try:
                        result = method_map[method](timeseries[col])
                        if result:
                            predictions.append(result)
                    except Exception as e:
                        logger.debug(f"{method} {col}: {e}")

            if self.use_ensemble and len(predictions) > 1:
                ensemble = self.ensemble_forecast(predictions)
                if ensemble:
                    results.append(ensemble)
            elif predictions:
                best = min(predictions, key=lambda x: x.get('mae', float('inf')))
                results.append(best)

        logger.info(f"Prédictions : {len(results)} topics")
        return results

    def backtest(self, timeseries: pd.DataFrame, method: str = 'prophet') -> pd.DataFrame:
        """Backtesting avec validation croisée temporelle."""
        results = []
        tscv = TimeSeriesSplit(n_splits=3)

        method_map = {}
        if PROPHET_AVAILABLE:
            method_map['prophet'] = self.predict_prophet
        if PMDARIMA_AVAILABLE:
            method_map['arima'] = self.predict_arima

        if method not in method_map:
            return pd.DataFrame()

        for col in timeseries.columns:
            series = timeseries[col].fillna(0)
            for fold, (train_idx, test_idx) in enumerate(tscv.split(series)):
                train = pd.Series(series.iloc[train_idx].values, index=series.index[train_idx])
                test = pd.Series(series.iloc[test_idx].values, index=series.index[test_idx])

                pred = method_map[method](train)
                if pred:
                    common = test.index.intersection(pred['forecast'].index)
                    if len(common) > 0:
                        metrics = compute_metrics(test.loc[common].values, pred['forecast'].loc[common].values)
                        results.append({'topic_id': col, 'fold': fold, 'method': method, **metrics})

        return pd.DataFrame(results)

    def save_forecasts(self, forecasts: List[Dict], output_dir: str = "data/processed/forecasts") -> Path:
        """Sauvegarde des prévisions."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for fc in forecasts:
            for date, val in fc["forecast"].items():
                rows.append({
                    "topic_id": fc["topic_id"],
                    "method": fc["method"],
                    "date": date,
                    "forecast": val,
                    "lower": fc.get("lower", pd.Series([np.nan]*len(fc["forecast"]), index=fc["forecast"].index)).get(date, np.nan),
                    "upper": fc.get("upper", pd.Series([np.nan]*len(fc["forecast"]), index=fc["forecast"].index)).get(date, np.nan),
                    "mae": fc.get("mae", np.nan),
                    "rmse": fc.get("rmse", np.nan),
                    "mape": fc.get("mape", np.nan),
                    "smape": fc.get("smape", np.nan),
                    "r2": fc.get("r2", np.nan),
                })

        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "forecasts.parquet", index=False)

        metrics_df = df.groupby(['topic_id', 'method']).agg({
            'forecast': 'mean', 'mae': 'first', 'rmse': 'first',
            'mape': 'first', 'smape': 'first', 'r2': 'first'
        }).reset_index()
        metrics_df.to_parquet(out_dir / "forecasts_metrics.parquet", index=False)

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_topics": len(forecasts),
            "horizon": self.horizon,
            "methods": list(set([f.get('method') for f in forecasts])),
            "avg_mae": float(df['mae'].mean()) if not df['mae'].isna().all() else None,
            "avg_rmse": float(df['rmse'].mean()) if not df['rmse'].isna().all() else None,
        }
        with open(out_dir / "performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Forecasts sauvegardés : {out_dir / 'forecasts.parquet'}")
        return out_dir / "forecasts.parquet"

    @staticmethod
    def summary(forecasts: List[Dict]) -> pd.DataFrame:
        """Résumé des performances."""
        df = pd.DataFrame([{
            "topic_id": f["topic_id"],
            "method": f["method"],
            "mae": f.get("mae", np.nan),
            "rmse": f.get("rmse", np.nan),
            "mape": f.get("mape", np.nan),
            "smape": f.get("smape", np.nan),
            "r2": f.get("r2", np.nan),
        } for f in forecasts])
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        ts = pd.read_parquet("data/processed/topic_timeseries.parquet")
        print(f"✅ Chargé : {len(ts)} périodes × {len(ts.columns)} topics")
    except FileNotFoundError:
        print("⚠️  Génération de données synthétiques...")
        dates = pd.date_range(start="2022-01-01", periods=24, freq="MS")
        n_topics = 20
        np.random.seed(SEED)
        data = np.random.dirichlet(np.ones(n_topics), size=24).T
        ts = pd.DataFrame(data, columns=[f"topic_{i}" for i in range(n_topics)], index=dates)

    predictor = TrendPredictor(horizon_months=6, use_deep_learning=True, use_ensemble=True)
    
    # Méthodes disponibles automatiquement détectées
    methods = []
    if PROPHET_AVAILABLE:
        methods.append('prophet')
    if PMDARIMA_AVAILABLE:
        methods.append('arima')
    if predictor.use_deep_learning:
        methods.append('lstm')

    if not methods:
        print("⚠️  Aucun modèle disponible. Installez prophet, pmdarima ou tensorflow.")
    else:
        print(f"\n📊 Méthodes disponibles : {', '.join(methods)}")
        forecasts = predictor.predict_all(ts, methods=methods)

        if forecasts:
            metrics = predictor.summary(forecasts)
            print("\n📊 Métriques par méthode:")
            print(metrics.groupby('method').agg({
                'mae': ['mean', 'std'], 'rmse': ['mean', 'std'], 'r2': ['mean', 'std']
            }).round(4))

            if PROPHET_AVAILABLE or PMDARIMA_AVAILABLE:
                print("\n🔬 Backtesting:")
                for method in ['prophet', 'arima']:
                    if method in methods:
                        bt = predictor.backtest(ts, method=method)
                        if not bt.empty:
                            print(f"  {method.upper()}: MAE={bt['mae'].mean():.4f} (±{bt['mae'].std():.4f})")

            output = predictor.save_forecasts(forecasts)
            print(f"\n✅ Sauvegardé : {output}")
        else:
            print("⚠️  Aucune prédiction générée.")