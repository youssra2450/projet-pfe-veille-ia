@echo off
echo ========================================
echo NETTOYAGE COMPLET DU PROJET
echo ========================================
echo.

echo [1/5] Suppression des donnees traitees...
rmdir /s /q data\processed 2>nul
rmdir /s /q data\raw 2>nul
rmdir /s /q data\models 2>nul
rmdir /s /q data\reports 2>nul
rmdir /s /q data\cache 2>nul
echo    OK

echo [2/5] Suppression des logs...
rmdir /s /q logs 2>nul
echo    OK

echo [3/5] Suppression des modèles entraînés...
rmdir /s /q agent_2_topic_modeling\__pycache__ 2>nul
rmdir /s /q agent_3_analyse_temporelle_predictive\__pycache__ 2>nul
rmdir /s /q agent_4_synthese_llm\__pycache__ 2>nul
echo    OK

echo [4/5] Recréation des dossiers...
mkdir data\processed 2>nul
mkdir data\raw 2>nul
mkdir data\models 2>nul
mkdir data\reports 2>nul
mkdir data\cache 2>nul
mkdir logs 2>nul
echo    OK

echo [5/5] Nettoyage termine !
echo.
echo Pour reexecuter tous les agents:
echo   python run_pipeline.py
echo.
pause