"""
Collecte incrémentale - Ne collecte que les nouveaux articles depuis la dernière collecte
À exécuter chaque semaine
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import yaml

def incremental_collect():
    # Calculer la date d'il y a 7 jours
    last_week = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"📅 Collecte incrémentale depuis le {last_week}")
    
    # Chemin du fichier de configuration
    config_path = Path("agent_1_collecte_nettoyage/config.yaml")
    
    # Lire la configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Ajouter la date de début
    config['arxiv']['start_date'] = last_week
    
    # Sauvegarder temporairement
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # Lancer la collecte
    result = subprocess.run(
        ["python", "agent_1_collecte_nettoyage/arxiv_fetcher.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode == 0:
        print("✅ Collecte incrémentale terminée avec succès")
    else:
        print(f"❌ Erreur: {result.stderr}")

if __name__ == "__main__":
    incremental_collect()