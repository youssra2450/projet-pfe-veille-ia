import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import re

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Optionnel : pour utiliser un LLM local
try:
    from bertopic.representation import Llama2
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("Llama2 non disponible. Installez 'transformers' pour l'utiliser.")

logger = logging.getLogger(__name__)
SEED = 42


class BERTopicModel:

    def __init__(self, 
                 use_llm_for_labels: bool = False,
                 llm_model: str = "llama2-7b-chat",
                 use_keybert: bool = True):
        logger.info("Chargement embeddings : all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=SEED,
        )
        
    
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=3,      
            min_samples=2,          
            cluster_selection_epsilon=0.05,  
            metric="euclidean",
            prediction_data=True,
        )
        
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        
        # Configuration du modèle de représentation
        self.representation_models = []
        
        # 1. KeyBERTInspired pour une meilleure extraction des mots-clés
        if use_keybert:
            logger.info("Ajout de KeyBERTInspired pour l'extraction des mots-clés")
            self.representation_models.append(
                KeyBERTInspired(top_n_words=15)
            )
        
        # 2. MaximalMarginalRelevance pour la diversité
        logger.info("Ajout de MaximalMarginalRelevance pour la diversité")
        self.representation_models.append(
            MaximalMarginalRelevance(diversity=0.3)
        )
        
        # 3. LLM pour la génération de noms (optionnel)
        self.llm_representation = None
        self.use_llm_for_labels = use_llm_for_labels
        
        if use_llm_for_labels and LLAMA_AVAILABLE:
            logger.info(f"Chargement du LLM pour la génération de noms : {llm_model}")
            try:
                # Prompt personnalisé pour générer des noms de topics en français
                prompt = """ Mots-clés : {keywords}"""
                self.llm_representation = Llama2(
                    model=llm_model,
                    prompt=prompt,
                    delay_in_seconds=1,
                )
                self.representation_models.append(self.llm_representation)
                logger.info("LLM chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du LLM : {e}")
                self.use_llm_for_labels = False
        
        # Création du modèle BERTopic avec les représentations configurées
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_models,
            top_n_words=10,
            verbose=True,
        )
        self.topics = None
        self.probs = None
        self.embeddings = None
        self.topic_labels = {}

    def fit(self, texts: List[str], docs_metadata: Optional[pd.DataFrame] = None):
        """
        Entraîne le modèle BERTopic.
        
        Args:
            texts: Liste des textes à traiter
            docs_metadata: Métadonnées des documents (optionnel)
        """
        logger.info("Encodage de %d textes...", len(texts))
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        self.topics, self.probs = self.topic_model.fit_transform(
            texts, embeddings=self.embeddings)

        n_topics = len(self.topic_model.get_topic_info()) - 1
        n_outliers = self.topics.count(-1)
        logger.info("BERTopic : %d topics, %d outliers", n_topics, n_outliers)
        
        # Génération des noms de topics
        self._generate_topic_labels()
        
        return self.topics, self.probs

    def _generate_topic_labels(self):
        """
        Génère des noms lisibles pour chaque topic.
        Utilise le LLM si disponible, sinon KeyBERTInspired.
        """
        topic_info = self.topic_model.get_topic_info()
        self.topic_labels = {}
        
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:  # Ignorer les outliers
                continue
                
            # Récupérer les mots-clés du topic
            words = self.topic_model.get_topic(topic_id)
            if not words:
                continue
                
            keywords = ", ".join([w for w, _ in words[:10]])
            
            # 1. Essayer d'utiliser le LLM si disponible
            if self.use_llm_for_labels and self.llm_representation:
                try:
                    label = self._get_llm_label(topic_id, keywords)
                    if label and len(label) > 5:
                        self.topic_labels[topic_id] = label
                        continue
                except Exception as e:
                    logger.warning(f"Erreur LLM pour topic {topic_id}: {e}")
            
            # 2. Fallback : générer un nom à partir des mots-clés avec KeyBERT
            label = self._generate_label_from_keywords(keywords)
            self.topic_labels[topic_id] = label
            
        logger.info(f"{len(self.topic_labels)} noms de topics générés")

    def _get_llm_label(self, topic_id: int, keywords: str) -> str:
        """
        Utilise le LLM pour générer un nom de topic.
        """
        try:
            representation = self.topic_model.get_topic(topic_id, full=True)
            if representation and len(representation) > 0:
                label = representation[0][0] if isinstance(representation[0], tuple) else representation[0]
                label = label.strip().strip('"\'')
                return label
        except Exception as e:
            logger.warning(f"Erreur lors de la génération LLM : {e}")
        return None

    def _generate_label_from_keywords(self, keywords: str) -> str:
        """
        Génère un nom de topic à partir des mots-clés.
        """
        words = [w.strip() for w in keywords.split(',')]
        words = [w for w in words if w and len(w) > 2]
        
        if not words:
            return "Topic non nommé"
        
        important_prefixes = ['learning', 'network', 'model', 'vision', 'language', 
                              'agent', 'quantum', 'optimization', 'neural', 'deep']
        
        selected = []
        for w in words[:5]:
            if any(prefix in w.lower() for prefix in important_prefixes):
                selected.append(w.capitalize())
            elif len(selected) < 2:
                selected.append(w.capitalize())
        
        if not selected:
            selected = [words[0].capitalize()] if words else ["Topic"]
        
        if len(selected) == 1:
            return f"{selected[0]} related"
        elif len(selected) == 2:
            return f"{selected[0]} & {selected[1]}"
        else:
            return f"{selected[0]}, {selected[1]} & {selected[2]}"

    def get_topic_info(self) -> pd.DataFrame:
        """
        Retourne les informations des topics avec les noms générés.
        """
        info = self.topic_model.get_topic_info()
        info = info[info["Topic"] != -1].copy()
        
        info["topic_label"] = info["Topic"].map(
            lambda t: self.topic_labels.get(t, f"Topic {t}")
        )
        
        info["top_words"] = info["Topic"].apply(
            lambda t: ", ".join(
                [w for w, _ in self.topic_model.get_topic(t)[:10]])
        )
        
        return info

    def get_topic_label(self, topic_id: int) -> str:
        """
        Retourne le nom d'un topic spécifique.
        """
        return self.topic_labels.get(topic_id, f"Topic {topic_id}")

    def set_custom_label(self, topic_id: int, label: str):
        """
        Permet de définir manuellement un nom pour un topic.
        """
        self.topic_labels[topic_id] = label
        logger.info(f"Label personnalisé pour topic {topic_id}: {label}")

    def save(self, dir_path="data/models/bertopic"):
        """Sauvegarde le modèle et les labels."""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        self.topic_model.save(
            str(Path(dir_path) / "bertopic_model"),
            serialization="pickle",
            save_ctfidf=True,
            save_embedding_model=True,
        )
        if self.embeddings is not None:
            np.save(
                str(Path(dir_path) / "embeddings.npy"),
                self.embeddings
            )
        
        label_path = Path(dir_path) / "topic_labels.json"
        import json
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(self.topic_labels, f, ensure_ascii=False, indent=2)
            
        logger.info("BERTopic sauvegardé : %s", dir_path)

    @classmethod
    def load(cls, dir_path="data/models/bertopic"):
        """Charge un modèle sauvegardé."""
        instance = cls.__new__(cls)
        instance.topic_model = BERTopic.load(
            str(Path(dir_path) / "bertopic_model"))
        
        emb_path = Path(dir_path) / "embeddings.npy"
        instance.embeddings = (
            np.load(str(emb_path))
            if emb_path.exists() else None
        )
        
        label_path = Path(dir_path) / "topic_labels.json"
        if label_path.exists():
            import json
            with open(label_path, "r", encoding="utf-8") as f:
                instance.topic_labels = json.load(f)
        else:
            instance.topic_labels = {}
            
        instance.topics = None
        instance.probs = None
        instance.use_llm_for_labels = False
        instance.llm_representation = None
        
        return instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet("data/processed/articles_clean.parquet")
    texts = df["clean_abstract"].dropna().tolist()
    print(f"Articles chargés : {len(texts)}")

    model = BERTopicModel(
        use_llm_for_labels=False,  
        use_keybert=True,          
    )
    
    topics, probs = model.fit(texts)

    df["topic_id"] = topics
    df.to_parquet(
        "data/processed/articles_with_topics.parquet",
        index=False)

    print("\n Topics trouvés avec noms générés :")
    topic_info = model.get_topic_info()
    print(topic_info[["Topic", "topic_label", "Count", "top_words"]].to_string())
    model.save()
    print(" BERTopic terminé.")