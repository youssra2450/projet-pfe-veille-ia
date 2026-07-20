"""
coherence_evaluation.py
------------------------
Évalue la cohérence thématique du modèle LDA (UMass, NPMI, C_v)
en s'appuyant sur gensim.models.CoherenceModel (méthode standard,
Röder et al. 2015 pour C_v).

Prérequis :
    pip install gensim --break-system-packages   (ou sans ce flag hors venv restreint)

Fichiers attendus (chemins relatifs à la racine du projet projet_PFE) :
    - data/models/lda_model.pkl
    - data/processed/articles_clean.parquet

Usage :
    python agent_2_topic_modeling/coherence_evaluation.py
"""

import pickle
import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# La classe doit être redéfinie ici (structure identique à lda_model.py)
# pour que pickle puisse reconstruire l'objet sauvegardé.
class LDAModel:
    def __init__(self, n_topics=10, max_iter=20,
                 max_features=5000, min_df=2, max_df=0.95):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.vectorizer = CountVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df,
            ngram_range=(1, 2),
        )
        self.model = None
        self.doc_topic_matrix = None
        self.dtm = None


def load_lda_model(path="data/models/lda_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_top_words_per_topic(lda_obj, n_top=10):
    """Retourne, pour chaque topic, la liste des n_top mots les plus caractéristiques."""
    vocab = lda_obj.vectorizer.get_feature_names_out()
    topics_words = []
    for topic_weights in lda_obj.model.components_:
        top_idx = topic_weights.argsort()[-n_top:][::-1]
        topics_words.append([vocab[i] for i in top_idx])
    return topics_words


def load_tokenized_texts(path="data/processed/articles_clean.parquet",
                          text_column="clean_abstract"):
    """Charge et tokenize (par simple découpage sur les espaces, le texte
    étant déjà lemmatisé/nettoyé par l'Agent 1)."""
    df = pd.read_parquet(path)
    texts = df[text_column].dropna().astype(str).tolist()
    logger.info("Documents chargés pour l'évaluation de cohérence : %d", len(texts))
    return [t.split() for t in texts]


def compute_coherence(topics_words, tokenized_texts, coherence_type, window_size=None):
    """
    coherence_type : 'u_mass', 'c_npmi' ou 'c_v'
    window_size    : taille de la fenêtre glissante (requis pour NPMI/C_v,
                      ignoré pour u_mass qui n'a pas besoin des textes bruts)
    """
    dictionary = Dictionary(tokenized_texts)
    kwargs = dict(topics=topics_words, texts=tokenized_texts, dictionary=dictionary,
                   coherence=coherence_type)
    if window_size is not None:
        kwargs["window_size"] = window_size
    cm = CoherenceModel(**kwargs)
    return cm.get_coherence_per_topic(), cm.get_coherence()


def main():
    lda_obj = load_lda_model()
    topics_words = get_top_words_per_topic(lda_obj, n_top=10)
    tokenized_texts = load_tokenized_texts()

    # UMass : n'a besoin que de la co-présence dans un même document.
    # NPMI et C_v : nécessitent une fenêtre glissante sur le texte réel.
    npmi_per_topic, npmi_overall = compute_coherence(
        topics_words, tokenized_texts, coherence_type="c_npmi", window_size=10)
    cv_per_topic, cv_overall = compute_coherence(
        topics_words, tokenized_texts, coherence_type="c_v", window_size=110)

    print(f"\n{'Topic':<6}{'NPMI':<10}{'C_v':<10}Top mots")
    for i, words in enumerate(topics_words):
        print(f"{i:<6}{npmi_per_topic[i]:<10.4f}{cv_per_topic[i]:<10.4f}{words[:5]}")

    print(f"\nNPMI moyen ({len(topics_words)} topics) : {round(npmi_overall, 4)}")
    print(f"C_v moyen ({len(topics_words)} topics)   : {round(cv_overall, 4)}")

    # Sauvegarde pour intégration au rapport / dashboard
    out = pd.DataFrame({
        "topic_id": range(len(topics_words)),
        "top_words": [", ".join(w) for w in topics_words],
        "npmi": [round(v, 4) for v in npmi_per_topic],
        "c_v": [round(v, 4) for v in cv_per_topic],
    })
    out_path = Path("data/processed/lda_coherence.csv")
    out.to_csv(out_path, index=False)
    logger.info("Résultats sauvegardés : %s", out_path)


if __name__ == "__main__":
    main()