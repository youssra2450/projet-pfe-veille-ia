import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Aller dans le bon dossier
os.chdir(r"C:\Users\youss\OneDrive\Desktop\projet_PFE")

# Liste des scripts à exécuter avec leur dossier de travail
# Liste des scripts qui ont besoin de leur propre dossier
# et ceux qui peuvent s'exécuter depuis la racine
scripts = [
    ("Agent 1 - Collecte", "agent_1_collecte_nettoyage/arxiv_fetcher.py", "agent_1_collecte_nettoyage"),
    ("Agent 1 - Nettoyage", "agent_1_collecte_nettoyage/data_cleaner.py", "agent_1_collecte_nettoyage"),
    ("Agent 2 - LDA", "agent_2_topic_modeling/lda_model.py", "."),  # ← exécute depuis racine
    ("Agent 2 - BERTopic", "agent_2_topic_modeling/bertopic_model.py", "."),
    ("Agent 2 - Dynamic Topics", "agent_2_topic_modeling/dynamic_topic_modeling.py", "."),
    ("Agent 3 - Analyse temporelle", "agent_3_analyse_temporelle_predictive/temporal_analysis.py", "."),
    ("Agent 3 - Detection tendances", "agent_3_analyse_temporelle_predictive/trend_detection.py", "."),
    ("Agent 3 - Predictions", "agent_3_analyse_temporelle_predictive/prediction_models.py", "."),
    ("Agent 4 - Synthese", "agent_4_synthese_llm/report_generator.py", "agent_4_synthese_llm"),
    ("Agent 5 - Export Power BI", "export_for_powerbi.py", "."),
]

# Logs des exécutions
log_file = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

print("="*50)
print("🚀 PIPELINE MULTI-AGENTS - VEILLE IA")
print("="*50)
print(f"📝 Logs sauvegardés dans : {log_file}\n")

# Ouvrir le fichier de log
with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"Pipeline exécuté le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("="*50 + "\n\n")
    
    success_count = 0
    fail_count = 0
    
    for nom, script, working_dir in scripts:
        print(f"🚀 {nom}...")
        log.write(f"\n🚀 {nom} - {script}\n")
        log.write("-"*30 + "\n")
        
        # Vérifier si le script existe
        if not os.path.exists(script):
            msg = f"❌ Fichier introuvable : {script}"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
            continue
        
        try:
            # Exécuter le script dans son dossier de travail
            result = subprocess.run(
                [sys.executable, script],
                cwd=working_dir,  # ⭐ Change le dossier de travail
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            # Enregistrer la sortie
            log.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                log.write(f"STDERR:\n{result.stderr}\n")
            
            if result.returncode != 0:
                msg = f"❌ Erreur dans {nom} (code: {result.returncode})"
                print(msg)
                log.write(msg + "\n")
                fail_count += 1
            else:
                msg = f"✅ {nom} terminé avec succès"
                print(msg)
                log.write(msg + "\n")
                success_count += 1
                
        except subprocess.TimeoutExpired:
            msg = f"❌ Timeout dans {nom} (> 1 heure)"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
        except Exception as e:
            msg = f"❌ Exception dans {nom} : {str(e)}"
            print(msg)
            log.write(msg + "\n")
            fail_count += 1
    
    # Résumé final
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DE L'EXÉCUTION")
    print("="*50)
    print(f"✅ Succès : {success_count}/{len(scripts)}")
    print(f"❌ Échecs : {fail_count}/{len(scripts)}")
    print(f"📁 Logs : {log_file}")
    
    log.write("\n" + "="*50 + "\n")
    log.write(f"RÉSUMÉ : {success_count} succès, {fail_count} échecs\n")
    log.write("="*50 + "\n")

print("\n" + "="*50)
if fail_count == 0:
    print("🎉 TOUS LES AGENTS ONT TERMINÉ AVEC SUCCÈS")
    print("📊 Tu peux maintenant lancer le dashboard !")
else:
    print("⚠️ Certains agents ont échoué. Vérifie les logs.")
print("="*50)

# Optionnel : lancer le dashboard
open_dashboard = input("\n🚀 Lancer le dashboard Streamlit ? (o/n): ").lower()
if open_dashboard == 'o':
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard2.py"])