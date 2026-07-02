"""
run_full_pipeline.py
Pipeline complet : Agents → Base de données → Power BI
Niveau Master : Orchestration professionnelle
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class MasterPipeline:
    """Pipeline complet du projet Master"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.start_time = None
    
    def run_agents(self):
        """Exécute tous les agents"""
        agents = [
            ("Agent 1 - Collecte", "agent_1_collecte_nettoyage/arxiv_fetcher.py"),
            ("Agent 1 - Nettoyage", "agent_1_collecte_nettoyage/data_cleaner.py"),
            ("Agent 2 - LDA", "agent_2_topic_modeling/lda_model.py"),
            ("Agent 2 - BERTopic", "agent_2_topic_modeling/bertopic_model.py"),
            ("Agent 2 - Dynamic Topics", "agent_2_topic_modeling/dynamic_topic_modeling.py"),
            ("Agent 3 - Analyse", "agent_3_analyse_temporelle_predictive/temporal_analysis.py"),
            ("Agent 3 - Tendances", "agent_3_analyse_temporelle_predictive/trend_detection.py"),
            ("Agent 3 - Prédictions", "agent_3_analyse_temporelle_predictive/prediction_models.py"),
            ("Agent 4 - Synthèse", "agent_4_synthese_llm/report_generator.py"),
        ]
        
        for name, script in agents:
            logger.info(f"🚀 {name}...")
            result = subprocess.run([sys.executable, script], cwd=self.root)
            if result.returncode != 0:
                logger.error(f"❌ {name} a échoué")
                return False
            logger.info(f"✅ {name} terminé")
        return True
    
    def update_database(self):
        """Met à jour la base de données Power BI"""
        logger.info("🔄 Mise à jour de la base de données...")
        from database_manager import ResearchDatabase
        db = ResearchDatabase()
        metrics = db.load_data_from_agents()
        db.close()
        logger.info(f"📊 Métriques: {metrics}")
        return True
    
    def generate_powerbi_report(self):
        """Génère un fichier .pbix via Python (optionnel)"""
        logger.info("📊 Génération du rapport Power BI...")
        # Cette fonction pourrait générer un template Power BI
        pass
    
    def run(self):
        """Exécute le pipeline complet"""
        logger.info("="*60)
        logger.info("🎓 PIPELINE MASTER - VEILLE IA")
        logger.info("="*60)
        
        self.start_time = datetime.now()
        
        # 1. Exécuter les agents
        if not self.run_agents():
            logger.error("Pipeline interrompu")
            return False
        
        # 2. Mettre à jour la base de données
        if not self.update_database():
            logger.error("Base de données non mise à jour")
            return False
        
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info("="*60)
        logger.info(f"✅ PIPELINE COMPLET - {duration:.1f}s")
        logger.info("📁 Base de données : data/research.db")
        logger.info("📊 Ouvrez Power BI et connectez-vous à ce fichier")
        logger.info("="*60)
        
        return True

if __name__ == "__main__":
    pipeline = MasterPipeline()
    pipeline.run()