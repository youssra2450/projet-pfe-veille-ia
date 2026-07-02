import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import spacy
import yaml

logger = logging.getLogger(__name__)


class TextCleaner:

    _LATEX_BLOCK  = re.compile(r"\$\$.*?\$\$", re.DOTALL)
    _LATEX_INLINE = re.compile(r"\$[^$]+?\$")
    _LATEX_CMD    = re.compile(r"\\[a-zA-Z]+\{[^}]*\}")
    _BIBREF       = re.compile(r"\[[\d,\s\-]+\]")
    _SPACES       = re.compile(r"\s+")

    def __init__(self, config: dict):
        clean_cfg          = config.get("cleaning", {})
        self.remove_latex  = clean_cfg.get("remove_latex", True)
        self.remove_refs   = clean_cfg.get("remove_references", True)
        self.min_len       = clean_cfg.get("min_abstract_length", 50)
        spacy_model        = clean_cfg.get("spacy_model", "en_core_web_sm")

        logger.info("Chargement spaCy : %s", spacy_model)
        self.nlp = spacy.load(spacy_model)

        self.processed_dir = Path(config["storage"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup directory
        self.backup_dir = self.processed_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def basic_clean(self, text: str) -> str:
        if not text:
            return ""
        if self.remove_latex:
            text = self._LATEX_BLOCK.sub(" ", text)
            text = self._LATEX_INLINE.sub(" ", text)
            text = self._LATEX_CMD.sub(" ", text)
        if self.remove_refs:
            text = self._BIBREF.sub(" ", text)
        return self._SPACES.sub(" ", text).strip()

    def clean_text_spacy(self, text: str) -> str:
        if not text:
            return ""
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.is_alpha
        ]
        return " ".join(tokens)

    def process_text(self, text: str) -> str:
        return self.clean_text_spacy(self.basic_clean(text))

    def load_existing_clean_data(self, filepath: Path) -> pd.DataFrame:
        """Charge les données nettoyées existantes"""
        if filepath.exists():
            try:
                df_existing = pd.read_parquet(filepath)
                logger.info(f"📂 Données nettoyées existantes : {len(df_existing)} articles")
                return df_existing
            except Exception as e:
                logger.warning(f"Erreur chargement données nettoyées: {e}")
                return pd.DataFrame()
        else:
            logger.info("📂 Aucune donnée nettoyée existante")
            return pd.DataFrame()

    def clean_dataframe(self, df: pd.DataFrame, existing_df: pd.DataFrame = None) -> pd.DataFrame:
        """Nettoie le dataframe et fusionne avec l'historique existant"""
        df = df.copy()
        
        # Filtrer les abstracts trop courts
        before = len(df)
        df = df[df["abstract"].str.len() >= self.min_len].reset_index(drop=True)
        logger.info("Filtrés (trop courts) : %d", before - len(df))
        
        # Nettoyer les nouveaux articles
        logger.info("Nettoyage de %d nouveaux articles...", len(df))

        from tqdm import tqdm
        tqdm.pandas(desc="Nettoyage abstracts")
        df["clean_abstract"] = df["abstract"].progress_apply(self.process_text)

        tqdm.pandas(desc="Nettoyage titres")
        df["clean_title"] = df["title"].progress_apply(self.process_text)

        df["abstract_word_count"] = df["clean_abstract"].str.split().str.len()
        
        # Fusionner avec l'historique existant si disponible
        if existing_df is not None and len(existing_df) > 0:
            # Garder les colonnes existantes dans l'historique
            for col in existing_df.columns:
                if col not in df.columns and col in existing_df.columns:
                    df[col] = None
            
            # Fusionner et supprimer les doublons
            df_combined = pd.concat([existing_df, df], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
            logger.info(f"📊 Fusion : {len(existing_df)} anciens + {len(df)} nouveaux = {len(df_combined)} total")
        else:
            df_combined = df
        
        logger.info("Nettoyage terminé : %d articles", len(df_combined))
        return df_combined

    def save_processed(self, df: pd.DataFrame,
                       output_path: str = None) -> Path:
        """Sauvegarde les données nettoyées"""
        output_path = Path(output_path or self.processed_dir / "articles_clean.parquet")
        
        # Sauvegarder une copie de backup si le fichier existe déjà
        if output_path.exists():
            backup_path = self.backup_dir / f"articles_clean_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df_existing = pd.read_parquet(output_path)
            df_existing.to_parquet(backup_path, index=False)
            logger.info(f"💾 Backup sauvegardé : {backup_path}")
        
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Sauvegardé : %s (%d articles)", output_path, len(df))
        return output_path


if __name__ == "__main__":
    from pathlib import Path
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    raw_path = Path(config["storage"]["processed_dir"]) / "articles_raw.parquet"
    if not raw_path.exists():
        print(f"❌ Fichier introuvable : {raw_path}")
        print("   Lancez d'abord : python arxiv_fetcher.py")
        exit(1)

    # Charger les données brutes
    df_raw = pd.read_parquet(raw_path)
    print(f"📚 Chargé : {len(df_raw)} articles bruts")
    
    # Charger les données nettoyées existantes
    cleaner = TextCleaner(config)
    clean_path = cleaner.processed_dir / "articles_clean.parquet"
    df_existing_clean = cleaner.load_existing_clean_data(clean_path)
    
    # Nettoyer et fusionner
    df_clean = cleaner.clean_dataframe(df_raw, df_existing_clean)
    
    # Sauvegarder
    out = cleaner.save_processed(df_clean)
    print(f"\n✅ Nettoyage terminé : {len(df_clean)} articles → {out}")