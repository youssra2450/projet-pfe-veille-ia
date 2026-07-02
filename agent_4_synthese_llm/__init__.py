"""
Agent 4 - Synthese et Interpretation
Package pour la generation de rapports de veille technologique

Modules:
- report_generator: Generation du rapport complet
- confidence_scorer: Calcul du score de confiance
"""

from .report_generator import (
    Config,
    load_real_data,
    validate_data,
    compute_statistical_insights,
    calculate_confidence_score,
    generate_llm_interpretation,
    generate_full_report,
    save_report,
    main as generate_report
)

from .confidence_scorer import (
    ConfidenceFactors,
    ConfidenceScorer,
    ConfidenceTracker,
    compute_confidence_from_dataframe
)

__version__ = "2.0.0"
__author__ = "Laboratoire de Recherche en IA"

__all__ = [
    # Report generator
    'Config',
    'load_real_data',
    'validate_data',
    'compute_statistical_insights',
    'calculate_confidence_score',
    'generate_llm_interpretation',
    'generate_full_report',
    'save_report',
    'generate_report',
    
    # Confidence scorer
    'ConfidenceFactors',
    'ConfidenceScorer',
    'ConfidenceTracker',
    'compute_confidence_from_dataframe'
]