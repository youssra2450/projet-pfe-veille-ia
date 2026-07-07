import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicTopics:

    def __init__(self, bertopic_model, freq="ME"):
        self.model = bertopic_model
        self.freq = freq

    def compute_topic_timeseries(self, df):
        df = df.copy()
        df["published_date"] = pd.to_datetime(
            df["published_date"])
        df["period"] = (
            df["published_date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        df = df[df["topic_id"] != -1]

        counts = (
            df.groupby(["period", "topic_id"])
            .size()
            .reset_index(name="count")
        )
        totals = df.groupby("period").size().rename("total")
        counts = counts.join(totals, on="period")
        counts["proportion"] = counts["count"] / counts["total"]

        pivot = counts.pivot_table(
            index="period",
            columns="topic_id",
            values="proportion",
            fill_value=0,
        )
        pivot.index.name = "date"
        logger.info("Série temporelle : %d périodes × %d topics",
                    *pivot.shape)
        return pivot

    def smooth(self, ts, window=3):
        return ts.rolling(
            window=window, min_periods=1, center=True).mean()

    def _growth_rate(self, series, n=3):
        clean = series.dropna()
        if len(clean) < 2 * n:
            return float("nan")
        recent = clean.iloc[-n:].mean()
        past = clean.iloc[-(2 * n):-n].mean()
        if past == 0:
            return float("nan")
        return (recent - past) / past

    def classify_topics(self, timeseries):
        smoothed = self.smooth(timeseries)
        rows = []
        for topic_id in timeseries.columns:
            series = smoothed[topic_id]
            growth = self._growth_rate(series)
            mean_prop = series.mean()
            std_prop = series.std()
            cv = std_prop / mean_prop if mean_prop > 0 else 0

            if np.isnan(growth):
                classification = "insufficient_data"
            elif growth > 0.3 and mean_prop < 0.05:
                classification = "emerging"
            elif growth > 0.15:
                classification = "growing"
            elif growth < -0.2:
                classification = "declining"
            elif cv > 0.5:
                classification = "cyclical"
            else:
                classification = "stable"

            try:
                words = ", ".join(
                    [w for w, _ in
                     self.model.get_topic(topic_id)[:3]])
            except Exception:
                words = f"topic_{topic_id}"

            rows.append({
                "topic_id": topic_id,
                "topic_words": words,
                "classification": classification,
                "growth_rate_3m": (
                    round(growth, 4)
                    if not np.isnan(growth) else None),
                "mean_proportion": round(mean_prop, 4),
            })

        df = pd.DataFrame(rows).sort_values(
            "mean_proportion", ascending=False)
        logger.info("Classification : %s",
                    df["classification"].value_counts().to_dict())
        return df

    def save_timeseries(
            self, ts,
            path="data/processed/topic_timeseries.parquet"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ts.to_parquet(path)
        logger.info("Série temporelle sauvegardée : %s", path)


if __name__ == "__main__":
    from bertopic_model import BERTopicModel
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(
        "data/processed/articles_with_topics.parquet")
    bert = BERTopicModel.load("data/models/bertopic")

    dyn = DynamicTopics(bert.topic_model)
    ts = dyn.compute_topic_timeseries(df)
    classified = dyn.classify_topics(ts)

    print("\n Classification des topics :")
    print(classified.to_string(index=False))

    dyn.save_timeseries(ts)
    classified.to_parquet(
        "data/processed/topics_classified.parquet",
        index=False)
    print(" Dynamic topic modeling terminé.")