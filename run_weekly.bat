@echo off
chcp 65001 > nul
cd /d C:\Users\youss\OneDrive\Desktop\projet_PFE

echo ========================================
echo VEILLE HEBDOMADAIRE - %date% %time%
echo ========================================

:: Activer l'environnement virtuel
call venv\Scripts\activate.bat

:: Agent 1 - Collecte
echo [1/7] Collecte des articles...
python agent_1_collecte_nettoyage\arxiv_fetcher.py
if errorlevel 1 goto error

:: Agent 1 - Nettoyage
echo [2/7] Nettoyage des articles...
python agent_1_collecte_nettoyage\data_cleaner.py
if errorlevel 1 goto error

:: Agent 2 - LDA
echo [3/7] Modélisation LDA...
python agent_2_topic_modeling\lda_model.py
if errorlevel 1 goto error

:: Agent 2 - BERTopic
echo [4/7] Modélisation BERTopic...
python agent_2_topic_modeling\bertopic_model.py
if errorlevel 1 goto error

:: Agent 2 - Dynamic Topics
echo [5/7] Topics dynamiques...
python agent_2_topic_modeling\dynamic_topic_modeling.py
if errorlevel 1 goto error

:: Agent 3 - Analyse et Prédictions
echo [6/7] Analyse temporelle et prédictions...
python agent_3_analyse_temporelle_predictive\temporal_analysis.py
python agent_3_analyse_temporelle_predictive\trend_detection.py
python agent_3_analyse_temporelle_predictive\prediction_models.py
if errorlevel 1 goto error

:: Agent 4 - Rapport
echo [7/7] Génération du rapport...
python agent_4_synthese_llm\report_generator.py
if errorlevel 1 goto error

echo ========================================
echo ✅ VEILLE TERMINEE - %date% %time%
echo ========================================
goto end

:error
echo ========================================
echo ❌ ERREUR - Veuillez vérifier les logs
echo ========================================

:end
pause