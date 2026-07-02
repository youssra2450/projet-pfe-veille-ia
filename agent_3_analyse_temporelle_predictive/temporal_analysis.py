"""
Agent 3 - Module d'Analyse Temporelle
Laboratoire de Recherche en IA - Projet Veille Technologique

Fonctionnalités : Série temporelle, Lissage, Détection de ruptures (PELT/Binseg),
Décomposition STL, Tests de stationnarité, Statistiques descriptives
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyse approfondie des séries temporelles de topics.
    Méthodes : Lissage, Détection de ruptures (PELT, Binseg),
    Décomposition STL, Tests de stationnarité, Statistiques robustes.
    """

    def __init__(self, freq: str = "W", smooth_window: int = 3):
        self.freq = freq
        self.smooth_window = smooth_window

    def build_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit la série temporelle à partir des articles."""
        df = df.copy()
        df["published_date"] = pd.to_datetime(df["published_date"])
        df["period"] = df["published_date"].dt.to_period(self.freq).dt.to_timestamp()
        df = df[df["topic_id"] != -1]

        counts = df.groupby(["period", "topic_id"]).size().reset_index(name="count")
        totals = df.groupby("period").size().rename("total")
        counts = counts.join(totals, on="period")
        counts["proportion"] = counts["count"] / counts["total"]

        pivot = counts.pivot_table(index="period", columns="topic_id",
                                   values="proportion", fill_value=0)
        pivot.index.name = "date"
        logger.info(f"Série temporelle : {pivot.shape[0]} périodes × {pivot.shape[1]} topics")
        return pivot

    def smooth(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Lissage par moyenne mobile centrée."""
        return ts.rolling(window=self.smooth_window, min_periods=1, center=True).mean()

    def smooth_savgol(self, ts: pd.DataFrame, window: int = 7, order: int = 2) -> pd.DataFrame:
        """Lissage par filtre de Savitzky-Golay."""
        result = ts.copy()
        for col in ts.columns:
            if len(ts[col].dropna()) >= window:
                result[col] = savgol_filter(ts[col].fillna(0).values, window, order)
        return result

    def detect_breakpoints_pelt(self, series: pd.Series, penalty: float = 3.0) -> List[int]:
        """Détection de ruptures par PELT (Pruned Exact Linear Time)."""
        try:
            import ruptures as rpt
            values = series.fillna(0).values
            if len(values) < 6:
                return []
            algo = rpt.Pelt(model="rbf", min_size=2).fit(values)
            bps = algo.predict(pen=penalty)
            return [b for b in bps if b < len(values)]
        except Exception as e:
            logger.warning(f"PELT error: {e}")
            return []

    def detect_breakpoints_binseg(self, series: pd.Series, n_breaks: int = 3) -> List[int]:
        """Détection de ruptures par Binary Segmentation."""
        try:
            import ruptures as rpt
            values = series.fillna(0).values
            if len(values) < 6:
                return []
            algo = rpt.Binseg(model="rbf", min_size=2).fit(values)
            bps = algo.predict(n_bkps=n_breaks)
            return [b for b in bps if b < len(values)]
        except Exception as e:
            logger.warning(f"Binseg error: {e}")
            return []

    def detect_all_breakpoints(self, ts: pd.DataFrame, method: str = "pelt") -> Dict:
        """Détecte les ruptures pour tous les topics."""
        result = {}
        fn = self.detect_breakpoints_pelt if method == "pelt" else self.detect_breakpoints_binseg
        for col in ts.columns:
            bps = fn(ts[col])
            if bps:
                result[col] = bps
        logger.info(f"Ruptures détectées : {len(result)} topics")
        return result

    def decompose_stl(self, series: pd.Series, period: int = 12) -> Optional[Dict]:
        """Décomposition STL (Seasonal-Trend decomposition using LOESS)."""
        try:
            from statsmodels.tsa.seasonal import STL
            if len(series.dropna()) < period * 2:
                return None
            stl = STL(series.fillna(0), period=period, robust=True)
            res = stl.fit()
            return {"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}
        except Exception as e:
            logger.warning(f"STL error: {e}")
            return None

    def stationarity_test(self, series: pd.Series) -> Dict:
        """Tests de stationnarité ADF et KPSS."""
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            values = series.dropna().values
            if len(values) < 10:
                return {"adf_pvalue": np.nan, "kpss_pvalue": np.nan, "is_stationary": None}

            adf = adfuller(values)
            kpss_result = kpss(values, regression='c')

            is_stationary = adf[1] < 0.05 and kpss_result[1] > 0.05
            return {"adf_pvalue": round(adf[1], 4), "kpss_pvalue": round(kpss_result[1], 4),
                    "is_stationary": is_stationary}
        except Exception as e:
            logger.warning(f"Stationarity error: {e}")
            return {"adf_pvalue": np.nan, "kpss_pvalue": np.nan, "is_stationary": None}

    def granger_causality(self, ts: pd.DataFrame, target: str, max_lag: int = 3) -> Dict:
        """Test de causalité de Granger entre topics."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            results = {}
            for col in ts.columns:
                if col == target:
                    continue
                data = ts[[target, col]].dropna().values
                if len(data) < max_lag * 3:
                    continue
                try:
                    test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                    p_values = [test[i][0]['ssr_chi2test'][1] for i in range(1, max_lag + 1)]
                    results[col] = {"min_pvalue": min(p_values), "max_lag": np.argmin(p_values) + 1}
                except:
                    continue
            return results
        except Exception as e:
            logger.warning(f"Granger error: {e}")
            return {}

    def compute_stats_advanced(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Statistiques avancées avec régression robuste (Theil-Sen)."""
        rows = []
        for col in ts.columns:
            series = ts[col].fillna(0)
            x = np.arange(len(series))

            # Régression standard
            if len(series) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(x, series.values)
            else:
                slope, r_value, p_value = 0.0, 0.0, 1.0

            # Régression robuste (Theil-Sen)
            robust_slope, robust_intercept = theilslopes(series.values, x)[:2]

            # Tests de stationnarité
            stationarity = self.stationarity_test(series)

            # Volatilité
            volatility = series.std() / (series.mean() + 1e-10)

            rows.append({
                "topic_id": col,
                "mean": round(series.mean(), 4),
                "std": round(series.std(), 4),
                "min": round(series.min(), 4),
                "max": round(series.max(), 4),
                "slope": round(slope, 6),
                "r2": round(r_value ** 2, 4),
                "p_value": round(p_value, 4),
                "robust_slope": round(robust_slope, 6),
                "volatility": round(volatility, 4),
                "is_stationary": stationarity.get("is_stationary"),
                "adf_pvalue": stationarity.get("adf_pvalue"),
                "kpss_pvalue": stationarity.get("kpss_pvalue"),
            })

        return pd.DataFrame(rows).sort_values("mean", ascending=False)

    def compute_correlation_matrix(self, ts: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        """Matrice de corrélation entre topics."""
        return ts.corr(method=method)

    def compute_rolling_correlation(self, ts: pd.DataFrame, window: int = 4) -> Dict:
        """Corrélation glissante entre tous les topics."""
        results = {}
        for i, col1 in enumerate(ts.columns):
            for col2 in ts.columns[i+1:]:
                corr = ts[col1].rolling(window=window).corr(ts[col2])
                results[f"{col1}_{col2}"] = corr
        return results

    def save(self, ts: pd.DataFrame, path: str = "data/processed/topic_timeseries.parquet") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ts.to_parquet(path)
        logger.info(f"Série temporelle sauvegardée : {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        df = pd.read_parquet("data/processed/articles_with_topics.parquet")
        print(f"✅ Articles chargés : {len(df)}")
    except FileNotFoundError:
        print("⚠️  Génération de données synthétiques...")
        n_articles, n_topics = 5000, 20
        df = pd.DataFrame({
            "published_date": pd.date_range("2023-01-01", periods=n_articles),
            "topic_id": np.random.randint(0, n_topics, n_articles)
        })

    analyzer = TemporalAnalyzer(freq="W", smooth_window=3)
    ts = analyzer.build_timeseries(df)
    ts_smooth = analyzer.smooth(ts)
    ts_savgol = analyzer.smooth_savgol(ts, window=7, order=2)

    # Statistiques avancées
    stats = analyzer.compute_stats_advanced(ts_smooth)
    print("\n📊 Statistiques par topic :")
    print(stats[["topic_id", "mean", "slope", "r2", "is_stationary", "volatility"]].head(10).to_string())

    # Détection de ruptures
    breakpoints = analyzer.detect_all_breakpoints(ts_smooth, method="pelt")
    if breakpoints:
        print(f"\n🔴 Ruptures détectées : {len(breakpoints)} topics")

    # Décomposition STL (exemple)
    if len(ts.columns) > 0:
        sample_topic = ts.columns[0]
        stl_result = analyzer.decompose_stl(ts[sample_topic], period=12)
        if stl_result:
            print(f"\n📈 Décomposition STL - Topic {sample_topic}:")
            print(f"  Trend variance: {stl_result['trend'].var():.6f}")
            print(f"  Seasonal variance: {stl_result['seasonal'].var():.6f}")

    # Matrice de corrélation
    if len(ts.columns) > 1:
        corr_matrix = analyzer.compute_correlation_matrix(ts_smooth)
        high_corr = corr_matrix[corr_matrix > 0.8].stack().dropna().index.tolist()
        print(f"\n🔗 Corrélations élevées (>0.8) : {len(high_corr)} paires")

    analyzer.save(ts_smooth)
    stats.to_parquet("data/processed/topic_stats_advanced.parquet", index=False)

    print("\n✅ Analyse temporelle terminée.")