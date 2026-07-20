import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Aller dans le dossier du projet
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Liste des scripts (chemins relatifs simples)
scripts = [
    ("Agent 1 - Collecte", "arxiv_fetcher.py", "agent_1_collecte_nettoyage"),
    ("Agent 1 - Nettoyage", "data_cleaner.py", "agent_1_collecte_nettoyage"),
    ("Agent 2 - LDA", "lda_model.py", "agent_2_topic_modeling"),
    ("Agent 2 - BERTopic", "bertopic_model.py", "agent_2_topic_modeling"),
    ("Agent 2 - Dynamic Topics", "dynamic_topic_modeling.py", "agent_2_topic_modeling"),
    ("Agent 3 - Analyse temporelle", "temporal_analysis.py", "agent_3_analyse_temporelle_predictive"),
    ("Agent 3 - Detection tendances", "trend_detection.py", "agent_3_analyse_temporelle_predictive"),
    ("Agent 3 - Predictions", "prediction_models.py", "agent_3_analyse_temporelle_predictive"),
    ("Agent 4 - Synthese", "report_generator.py", "agent_4_synthese_llm"),
    ("Export ", "export.py", "."),
]

log_file = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("="*50)
print(" PIPELINE MULTI-AGENTS - VEILLE IA")
print("="*50)
print(f" Logs sauvegardes dans : {log_file}\n")

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"Pipeline execute le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("="*50 + "\n\n")
    
    success_count = 0
    fail_count = 0
    
    for nom, script, working_dir in scripts:
        print(f" {nom}...")
        log.write(f"\n {nom} - {script}\n")
        log.write("-"*30 + "\n")
        
        script_path = os.path.join(working_dir, script)
        
        if not os.path.exists(script_path):
            msg = f" Fichier introuvable : {script_path}"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
            continue
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            log.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                log.write(f"STDERR:\n{result.stderr}\n")
            
            if result.returncode != 0:
                msg = f" Erreur dans {nom} (code: {result.returncode})"
                print(msg)
                log.write(msg + "\n")
                fail_count += 1
            else:
                msg = f" {nom} termine avec succes"
                print(msg)
                log.write(msg + "\n")
                success_count += 1
                
        except subprocess.TimeoutExpired:
            msg = f" Timeout dans {nom} (> 1 heure)"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
        except Exception as e:
            msg = f" Exception dans {nom} : {str(e)}"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
    
    print("\n" + "="*50)
    print(" RESUME DE L'EXECUTION")
    print("="*50)
    print(f" Succes : {success_count}/{len(scripts)}")
    print(f" Echecs : {fail_count}/{len(scripts)}")
    print(f" Logs : {log_file}")
    
    log.write("\n" + "="*50 + "\n")
    log.write(f"RESUME : {success_count} succes, {fail_count} echecs\n")
    log.write("="*50 + "\n")

print("\n" + "="*50)
if fail_count == 0:
    print(" TOUS LES AGENTS ONT TERMINE AVEC SUCCES")
    print(" Tu peux maintenant lancer le dashboard !")
else:
    print(" Certains agents ont echoue. Verifie les logs.")
print("="*50)

# Lancer le dashboard
try:
    open_dashboard = input("\n Lancer le dashboard Streamlit ? (o/n): ").lower()
    if open_dashboard == 'o':
        if os.path.exists("dashboard.py"):
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        else:
            print(" dashboard.py non trouve")
except:
    print(" Dashboard non lance.")