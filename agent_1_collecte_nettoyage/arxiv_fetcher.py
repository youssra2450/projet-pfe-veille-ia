import os
import json
import time
import hashlib
import logging
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import yaml
import backoff
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(message)s")
    log_file = log_cfg.get("log_file", "logs/agent1.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


class ArxivCollector:

    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self, config: dict):
        self.arxiv_cfg  = config["arxiv"]
        self.storage_cfg = config["storage"]
        self.base_url   = self.arxiv_cfg["base_url"]
        self.rate_limit = self.arxiv_cfg["rate_limit_seconds"]
        self.user_agent = self.arxiv_cfg["user_agent"]
        self.max_per_query = self.arxiv_cfg["max_results_per_query"]
        self.total_max  = self.arxiv_cfg["total_max_results"]

        self.raw_dir       = Path(self.storage_cfg["raw_dir"])
        self.processed_dir = Path(self.storage_cfg["processed_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._seen_hashes       = set()
        self._last_request_time = 0.0
        
        # Backup directory pour l'historique
        self.backup_dir = self.processed_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ArxivCollector prêt — catégories : %s",
                    self.arxiv_cfg["categories"])

    def _respect_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        wait = self.rate_limit - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    @backoff.on_exception(
        backoff.expo,
        (urllib.error.URLError, urllib.error.HTTPError),
        max_tries=3,
    )
    def _fetch_url(self, url: str) -> bytes:
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    @staticmethod
    def _compute_hash(title: str, abstract: str) -> str:
        content = (title + abstract[:300]).lower().encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _parse_entries(self, xml_data: bytes) -> List[Dict]:
        root    = ET.fromstring(xml_data)
        ns      = self.NAMESPACE
        entries = root.findall("atom:entry", ns)
        articles = []

        for entry in entries:
            try:
                article_id = entry.find("atom:id", ns).text.split("/")[-1]
                title      = re.sub(r"\s+", " ",
                             entry.find("atom:title", ns).text.strip())
                abstract   = re.sub(r"\s+", " ",
                             entry.find("atom:summary", ns).text.strip())
                published  = entry.find("atom:published", ns).text
                updated    = entry.find("atom:updated", ns).text

                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ]
                categories = [
                    c.attrib.get("term", "")
                    for c in entry.findall("atom:category", ns)
                ]

                h = self._compute_hash(title, abstract)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)

                articles.append({
                    "id":             article_id,
                    "title":          title,
                    "abstract":       abstract,
                    "published_date": published,
                    "updated_date":   updated,
                    "authors":        authors,
                    "categories":     categories,
                    "hash":           h,
                    "collected_at":   datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.warning("Erreur parsing : %s", e)
                continue

        return articles

    def fetch_category(self, category, max_results=None):
        max_results = max_results or self.total_max
        articles = []
        start = 0
        chunk = min(self.max_per_query, max_results)

        # Filtre de date dans la query arXiv
        start_date = self.arxiv_cfg.get("start_date", "2022-01-01")
        end_date = self.arxiv_cfg.get("end_date", "2024-12-31")

        # Format arXiv : YYYYMMDD
        start_fmt = start_date.replace("-", "") + "0000"
        end_fmt = end_date.replace("-", "") + "2359"

        query = (
            f"cat:{category}+AND+"
            f"submittedDate:[{start_fmt}+TO+{end_fmt}]"
        )

        logger.info(
            "Collecte %s de %s à %s (max %d)...",
            category, start_date, end_date, max_results)

        with tqdm(total=max_results,
                desc=f"arXiv {category}", unit="art") as pbar:
            while start < max_results:
                self._respect_rate_limit()
                url = (
                    f"{self.base_url}?search_query={query}"
                    f"&start={start}&max_results={chunk}"
                    f"&sortBy=submittedDate"
                    f"&sortOrder=descending"
                )
                try:
                    xml_data = self._fetch_url(url)
                    batch = self._parse_entries(xml_data)
                    if not batch:
                        break
                    articles.extend(batch)
                    pbar.update(len(batch))
                    start += chunk
                    chunk = min(self.max_per_query,
                                max_results - start)
                except Exception as e:
                    logger.error("Erreur : %s", e)
                    break

        logger.info("%s : %d articles collectés",
                    category, len(articles))
        return articles

    def fetch_all(self) -> List[Dict]:
        categories   = self.arxiv_cfg["categories"]
        all_articles = []
        for cat in categories:
            batch = self.fetch_category(
                cat, max_results=self.total_max // len(categories)
            )
            all_articles.extend(batch)
        logger.info("Total : %d articles uniques", len(all_articles))
        return all_articles

    def load_existing_articles(self, filepath: Path) -> pd.DataFrame:
        """Charge les articles existants s'ils existent"""
        if filepath.exists():
            try:
                df_existing = pd.read_parquet(filepath)
                logger.info(f"📂 Historique chargé : {len(df_existing)} articles existants")
                return df_existing
            except Exception as e:
                logger.warning(f"Erreur chargement historique: {e}")
                return pd.DataFrame()
        else:
            logger.info("📂 Aucun historique existant, création d'un nouveau fichier")
            return pd.DataFrame()

    def save_to_parquet(self, articles: List[Dict],
                        output_path: str = None) -> Path:
        """Sauvegarde en préservant l'historique existant"""
        output_path = Path(output_path or self.processed_dir / "articles_raw.parquet")
        
        # Nouveaux articles en DataFrame
        df_new = pd.DataFrame(articles)
        for col in ["authors", "categories"]:
            if col in df_new.columns:
                df_new[col] = df_new[col].apply(json.dumps)
        
        # Charger les articles existants
        df_existing = self.load_existing_articles(output_path)
        
        # Fusionner les données
        if len(df_existing) > 0:
            # Supprimer les doublons basés sur l'ID
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
            
            # Sauvegarder une copie de backup avant l'écrasement
            backup_path = self.backup_dir / f"articles_raw_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df_existing.to_parquet(backup_path, index=False)
            logger.info(f"💾 Backup sauvegardé : {backup_path}")
            
            logger.info(f"📊 Fusion : {len(df_existing)} anciens + {len(df_new)} nouveaux = {len(df_combined)} total")
        else:
            df_combined = df_new
        
        # Sauvegarder le fichier complet
        df_combined.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"💾 Sauvegardé : {output_path} ({len(df_combined)} articles)")
        
        return output_path


if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    collector = ArxivCollector(config)
    articles  = collector.fetch_all()
    path      = collector.save_to_parquet(articles)
    print(f"\n✅ Collecte terminée : {len(articles)} nouveaux articles → {path}")