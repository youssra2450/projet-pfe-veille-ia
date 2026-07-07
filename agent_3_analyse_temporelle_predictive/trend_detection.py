import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def compute_growth_rate(series: pd.Series, n: int = 3) -> float:
    """Taux de croissance entre la moyenne des n dernières et n précédentes périodes."""
    clean = series.dropna()
    if len(clean) < 2 * n:
        return np.nan
    recent = clean.iloc[-n:].mean()
    past = clean.iloc[-(2 * n):-n].mean()
    return (recent - past) / past if past != 0 else np.nan


def compute_momentum(series: pd.Series, n: int = 3) -> float:
    """Momentum : différence absolue de croissance normalisée."""
    gr = compute_growth_rate(series, n)
    if np.isnan(gr):
        return np.nan
    return gr / (series.std() + 1e-10)


def compute_volatility(series: pd.Series, n: int = 3) -> float:
    """Volatilité"""
    recent = series.dropna().iloc[-n:] if len(series.dropna()) >= n else series.dropna()
    if len(recent) < 2 or recent.mean() == 0:
        return np.nan
    return recent.std() / recent.mean()


def classify_trend(series: pd.Series, n: int = 3) -> str:
    """Classification robuste des tendances."""
    gr = compute_growth_rate(series, n)
    if np.isnan(gr):
        return "insufficient_data"

    # Seuils ajustés pour une classification plus fine
    if gr > 0.5:
        return "emerging"
    elif gr > 0.15:
        return "growing"
    elif gr < -0.3:
        return "declining"
    elif gr < -0.1:
        return "decreasing"
    else:
        return "stable"


def compute_emergence_score(series: pd.Series, n: int = 3, alpha: float = 0.5) -> float:
    """
    Score d'émergence pondéré.
    
    """
    growth = compute_growth_rate(series, n)
    if np.isnan(growth) or growth <= 0:
        return 0.0

    current_volume = series.iloc[-n:].mean()
    max_volume = series.max()
    if max_volume == 0:
        return 0.0

    maturity = current_volume / max_volume
    volume_ratio = current_volume / max_volume

    score = growth * (1 - maturity) ** alpha * volume_ratio
    return max(0.0, float(score))


def compute_emergence_score_weighted(series: pd.Series, n: int = 3) -> float:
    """
    Score d'émergence avec pondération par volatilité et momentum.
    """
    base_score = compute_emergence_score(series, n)
    if base_score == 0:
        return 0.0

    volatility = compute_volatility(series, n)
    if np.isnan(volatility):
        volatility = 0.1

    momentum = compute_momentum(series, n)
    if np.isnan(momentum):
        momentum = 0.0

    # Pondération : plus de volatilité = plus de risque mais aussi plus de potentiel
    risk_factor = 1 + 0.5 * min(volatility, 1.0)
    momentum_factor = 1 + 0.3 * max(0, min(momentum, 2.0))

    return base_score * risk_factor * momentum_factor


def detect_emerging_topics_advanced(
    timeseries: pd.DataFrame,
    topic_labels: Dict = None,
    n: int = 3,
    top_n: int = 10,
    threshold_std: float = 1.5
) -> pd.DataFrame:
    scores_base = {}
    scores_weighted = {}
    growths = {}
    momentums = {}
    volatilities = {}

    for col in timeseries.columns:
        series = timeseries[col]
        scores_base[col] = compute_emergence_score(series, n)
        scores_weighted[col] = compute_emergence_score_weighted(series, n)
        growths[col] = compute_growth_rate(series, n)
        momentums[col] = compute_momentum(series, n)
        volatilities[col] = compute_volatility(series, n)

    score_series = pd.Series(scores_weighted)
    mean_s = score_series.mean()
    std_s = score_series.std()
    threshold = mean_s + threshold_std * std_s

    logger.info(f"Seuil d'émergence : {threshold:.5f} (moyenne={mean_s:.5f}, std={std_s:.5f})")

    rows = []
    for topic_id in timeseries.columns:
        rows.append({
            "topic_id": topic_id,
            "topic_label": (topic_labels or {}).get(topic_id, f"topic_{topic_id}"),
            "emergence_score": round(scores_weighted.get(topic_id, 0), 5),
            "base_score": round(scores_base.get(topic_id, 0), 5),
            "growth_rate": round(growths.get(topic_id, np.nan), 4),
            "momentum": round(momentums.get(topic_id, np.nan), 4),
            "volatility": round(volatilities.get(topic_id, np.nan), 4),
            "current_volume": round(timeseries[topic_id].iloc[-n:].mean(), 5),
            "is_emerging": scores_weighted.get(topic_id, 0) >= threshold,
        })

    df = pd.DataFrame(rows).sort_values("emergence_score", ascending=False).reset_index(drop=True)
    logger.info(f"Topics émergents : {df['is_emerging'].sum()} sur {len(df)}")

    return df.head(top_n)


def compute_trend_matrix(timeseries: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Matrice des tendances entre topics."""
    trends = {}
    for col in timeseries.columns:
        trends[col] = classify_trend(timeseries[col], n)

    df = pd.DataFrame.from_dict(trends, orient='index', columns=['trend'])
    return df


def compute_growth_matrix(timeseries: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Matrice des taux de croissance."""
    growths = {}
    for col in timeseries.columns:
        growths[col] = compute_growth_rate(timeseries[col], n)

    df = pd.DataFrame.from_dict(growths, orient='index', columns=['growth_rate'])
    return df


def identify_leading_lagging(topics_df: pd.DataFrame, growth_col: str = "growth_rate") -> Dict:
    """Identification des topics leaders et suiveurs basée sur la croissance."""
    if len(topics_df) == 0:
        return {"leading": [], "lagging": []}

    mean_growth = topics_df[growth_col].mean()
    std_growth = topics_df[growth_col].std()

    leading = topics_df[topics_df[growth_col] > mean_growth + std_growth]["topic_id"].tolist()
    lagging = topics_df[topics_df[growth_col] < mean_growth - std_growth]["topic_id"].tolist()

    return {"leading": leading, "lagging": lagging, "mean_growth": mean_growth, "std_growth": std_growth}


def generate_trend_report(emerging_df: pd.DataFrame, growth_df: pd.DataFrame, top_n: int = 5) -> str:
    """Génère un rapport textuel des tendances."""
    lines = ["RAPPORT DE TENDANCES - IA RESEARCH", "=" * 40, ""]

    # Topics émergents
    lines.append(" TOPICS ÉMERGENTS")
    for _, row in emerging_df.head(top_n).iterrows():
        lines.append(f"  • {row['topic_label']}: +{row['growth_rate']*100:.1f}% "
                     f"(score={row['emergence_score']:.3f})")

    # Topics en déclin
    declining = growth_df[growth_df['growth_rate'].notna()].sort_values('growth_rate').head(top_n)
    if not declining.empty:
        lines.append("\n TOPICS EN DÉCLIN")
        for idx, row in declining.iterrows():
            lines.append(f"  • {row.name}: {row['growth_rate']*100:.1f}%")

    # Statistiques générales
    lines.append(f"\n STATISTIQUES")
    lines.append(f"  • Topics analysés : {len(growth_df)}")
    lines.append(f"  • Croissance moyenne : {growth_df['growth_rate'].mean()*100:.1f}%")
    lines.append(f"  • Topics émergents : {emerging_df['is_emerging'].sum()}")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        ts = pd.read_parquet("data/processed/topic_timeseries.parquet")
        print(f" Série temporelle chargée : {ts.shape}")
    except FileNotFoundError:
        print("  Génération de données synthétiques...")
        dates = pd.date_range(start="2022-01-01", periods=24, freq="MS")
        n_topics = 20
        np.random.seed(42)
        data = np.random.dirichlet(np.ones(n_topics), size=24).T
        # Ajout de tendances artificielles
        for i in range(n_topics):
            data[i] = data[i] * (1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 24)))
        ts = pd.DataFrame(data, columns=[f"topic_{i}" for i in range(n_topics)], index=dates)

    # 1. Matrice des taux de croissance
    growth_df = compute_growth_matrix(ts, n=3)
    print("\n Taux de croissance :")
    print(growth_df.sort_values('growth_rate', ascending=False).head(10).to_string())

    # 2. Matrice des tendances
    trend_df = compute_trend_matrix(ts, n=3)
    print("\n Classification des tendances :")
    print(trend_df['trend'].value_counts().to_string())

    # 3. Détection des topics émergents
    emerging = detect_emerging_topics_advanced(ts, top_n=15, threshold_std=1.5)
    print("\n TOPICS ÉMERGENTS :")
    print(emerging[["topic_id", "emergence_score", "growth_rate", "momentum", "volatility"]].head(10).to_string())

    # 4. Identification leaders/suiveurs
    leadership = identify_leading_lagging(growth_df.reset_index().rename(columns={'index': 'topic_id'}))
    print(f"\n Topics leaders : {len(leadership['leading'])}")
    print(f" Topics suiveurs : {len(leadership['lagging'])}")

    # 5. Rapport
    report = generate_trend_report(emerging, growth_df)
    print("\n" + report)

    # Sauvegarde
    emerging.to_parquet("data/processed/emerging_topics_advanced.parquet", index=False)
    growth_df.to_parquet("data/processed/growth_rates.parquet")
    trend_df.to_parquet("data/processed/trend_classification.parquet")

    print("\n Détection des tendances terminée.")