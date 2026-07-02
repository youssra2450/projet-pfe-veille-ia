import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)
SEED = 42


class LDAModel:

    def __init__(self, n_topics=10, max_iter=20,
                 max_features=5000, min_df=2, max_df=0.95):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
        )
        self.model = None
        self.doc_topic_matrix = None
        self.dtm = None

    def fit(self, texts):
        logger.info("Vectorisation de %d textes...", len(texts))
        self.dtm = self.vectorizer.fit_transform(texts)
        logger.info("Entraînement LDA (n_topics=%d)...",
                    self.n_topics)
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method="online",
            random_state=SEED,
            n_jobs=-1,
        )
        self.doc_topic_matrix = self.model.fit_transform(self.dtm)
        logger.info("LDA entraîné. Perplexité : %.2f",
                    self.model.perplexity(self.dtm))
        return self

    @classmethod
    def grid_search(cls, texts, topic_range=range(5, 20, 5)):
        logger.info("Grid search sur %s topics...", list(topic_range))
        dtm = CountVectorizer(
            max_features=5000, min_df=2, max_df=0.95
        ).fit_transform(texts)
        results = []
        for n in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n, max_iter=10,
                random_state=SEED, n_jobs=-1)
            lda.fit(dtm)
            perp = lda.perplexity(dtm)
            results.append({"n_topics": n, "perplexity": perp})
            logger.info("  n_topics=%d → perplexity=%.2f", n, perp)
        return pd.DataFrame(results)

    def get_top_words(self, n_top=10):
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        vocab = self.vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(self.model.components_):
            top_idx = topic.argsort()[-n_top:][::-1]
            topics[idx] = [vocab[i] for i in top_idx]
        return topics

    def get_topic_summary_df(self, n_top=10):
        top_words = self.get_top_words(n_top)
        rows = []
        for tid, words in top_words.items():
            rows.append({
                "topic_id": tid,
                "top_words": ", ".join(words),
                "weight_sum": float(
                    self.model.components_[tid].sum()),
            })
        return pd.DataFrame(rows).sort_values(
            "weight_sum", ascending=False)

    def save(self, path="data/models/lda_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Modèle LDA sauvegardé : %s", path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet("data/processed/articles_clean.parquet")
    texts = df["clean_abstract"].dropna().tolist()
    print(f"Articles chargés : {len(texts)}")

    results = LDAModel.grid_search(
        texts, topic_range=range(5, 15, 5))
    print(results)

    best_n = int(
        results.loc[results["perplexity"].idxmin(), "n_topics"])
    print(f"Meilleur nombre de topics : {best_n}")

    model = LDAModel(n_topics=best_n).fit(texts)
    print(model.get_topic_summary_df())
    model.save()
    print("✅ LDA terminé.")