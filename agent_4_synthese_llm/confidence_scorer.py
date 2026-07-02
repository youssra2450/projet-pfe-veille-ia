"""
Module de scoring de confiance - Niveau Master Recherche
Evaluation objective de la fiabilite des analyses
Base sur des metriques quantitatives et des facteurs ponderes
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ConfidenceFactors:
    """Facteurs objectifs de confiance"""
    data_volume_score: float = 0.0
    temporal_coverage_score: float = 0.0
    topic_diversity_score: float = 0.0
    growth_quality_score: float = 0.0
    coherence_score: float = 0.0
    
    def overall(self, weights: Dict[str, float]) -> float:
        """Calcule le score global pondere"""
        total = 0.0
        for factor, weight in weights.items():
            value = getattr(self, factor, 0.0)
            total += value * weight
        return min(1.0, max(0.0, total))


class ConfidenceScorer:
    """
    Calcule un score de confiance objectif pour chaque analyse.
    Approche : ponderation de 5 facteurs bases sur des metriques quantitatives.
    
    Facteurs :
    1. Volume de donnees (25%) - Nombre d'articles
    2. Couverture temporelle (20%) - Nombre de periodes
    3. Diversite des topics (20%) - Nombre de topics emergents
    4. Qualite des croissances (20%) - Ecart-type des croissances
    5. Coherence globale (15%) - Verification croisee des metriques
    """
    
    def __init__(self):
        self.weights = {
            "data_volume_score": 0.25,
            "temporal_coverage_score": 0.20,
            "topic_diversity_score": 0.20,
            "growth_quality_score": 0.20,
            "coherence_score": 0.15
        }
    
    def compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcule le score de confiance global et par facteur.
        
        Args:
            data: Dictionnaire contenant les metriques d'analyse
            
        Returns:
            Dictionnaire avec score global, niveau, interpretation et facteurs
        """
        
        factors = ConfidenceFactors()
        
        # 1. Volume de donnees (0-1)
        n_articles = data.get('n_articles', 0)
        if n_articles >= 5000:
            factors.data_volume_score = 1.0
        elif n_articles >= 2000:
            factors.data_volume_score = 0.8
        elif n_articles >= 1000:
            factors.data_volume_score = 0.6
        elif n_articles >= 500:
            factors.data_volume_score = 0.4
        elif n_articles >= 100:
            factors.data_volume_score = 0.2
        else:
            factors.data_volume_score = 0.1
        
        # 2. Couverture temporelle (0-1)
        n_periods = data.get('n_periods', 0)
        if n_periods >= 26:  # 6 mois
            factors.temporal_coverage_score = 1.0
        elif n_periods >= 13:  # 3 mois
            factors.temporal_coverage_score = 0.7
        elif n_periods >= 8:
            factors.temporal_coverage_score = 0.5
        elif n_periods >= 4:
            factors.temporal_coverage_score = 0.3
        else:
            factors.temporal_coverage_score = 0.1
        
        # 3. Diversite des topics (0-1)
        n_topics = data.get('n_topics', 0)
        if n_topics >= 30:
            factors.topic_diversity_score = 1.0
        elif n_topics >= 15:
            factors.topic_diversity_score = 0.8
        elif n_topics >= 10:
            factors.topic_diversity_score = 0.6
        elif n_topics >= 5:
            factors.topic_diversity_score = 0.4
        else:
            factors.topic_diversity_score = 0.2
        
        # 4. Qualite des croissances (0-1)
        std_growth = data.get('growth_std', 0)
        if std_growth > 0:
            # Moins de variation = plus de confiance
            factors.growth_quality_score = max(0, min(1, 1 - std_growth / 500))
        else:
            factors.growth_quality_score = 0.5
        
        # 5. Coherence globale (0-1)
        # Verification de la coherence entre les metriques
        if n_articles > 1000 and n_topics > 10:
            factors.coherence_score = 0.9
        elif n_articles > 500 and n_topics > 5:
            factors.coherence_score = 0.7
        elif n_articles > 100 and n_topics > 3:
            factors.coherence_score = 0.5
        else:
            factors.coherence_score = 0.3
        
        # Score global
        overall = factors.overall(self.weights)
        
        # Niveau de confiance
        if overall >= 0.80:
            confidence_level = "Eleve"
            interpretation = "Resultats robustes, tendances fiables"
        elif overall >= 0.60:
            confidence_level = "Moyen"
            interpretation = "Resultats indicatifs, a confirmer"
        elif overall >= 0.40:
            confidence_level = "Faible"
            interpretation = "Tendances preliminaires"
        else:
            confidence_level = "Insuffisant"
            interpretation = "Donnees insuffisantes"
        
        return {
            'overall_score': round(overall, 3),
            'confidence_level': confidence_level,
            'interpretation': interpretation,
            'factors': asdict(factors),
            'details': {
                'n_articles': n_articles,
                'n_topics': n_topics,
                'n_periods': n_periods,
                'growth_std': std_growth
            }
        }
    
    def compute_batch(self, data_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calcule les scores de confiance pour plusieurs analyses.
        
        Args:
            data_list: Liste de dictionnaires contenant les metriques
            
        Returns:
            DataFrame avec les scores pour chaque analyse
        """
        results = []
        for i, data in enumerate(data_list):
            score = self.compute(data)
            results.append({
                'analysis_id': data.get('analysis_id', i),
                'overall_score': score['overall_score'],
                'confidence_level': score['confidence_level'],
                'data_volume_score': score['factors']['data_volume_score'],
                'temporal_coverage_score': score['factors']['temporal_coverage_score'],
                'topic_diversity_score': score['factors']['topic_diversity_score'],
                'growth_quality_score': score['factors']['growth_quality_score'],
                'coherence_score': score['factors']['coherence_score'],
                'n_articles': score['details']['n_articles'],
                'n_topics': score['details']['n_topics']
            })
        
        return pd.DataFrame(results)


class ConfidenceTracker:
    """
    Suit l'evolution des scores de confiance dans le temps.
    Permet de mesurer l'amelioration du systeme.
    """
    
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.metrics_dir / "confidence_history.json"
    
    def log(self, scores: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Enregistre un score dans l'historique.
        
        Args:
            scores: Dictionnaire des scores de confiance
            metadata: Metadonnees supplementaires (optionnel)
        """
        history = self.load_history()
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'scores': scores,
            'metadata': metadata or {}
        }
        
        history.append(entry)
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def load_history(self) -> list:
        """Charge l'historique des scores."""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def get_trend(self) -> str:
        """
        Analyse la tendance des scores de confiance.
        
        Returns:
            'amelioration', 'degradation' ou 'stable'
        """
        history = self.load_history()
        if len(history) < 2:
            return "stable"
        
        # Derniers 3 scores
        recent_scores = [h['scores'].get('overall_score', 0) for h in history[-3:]]
        # Scores precedents (hors derniers 3)
        older_scores = [h['scores'].get('overall_score', 0) for h in history[:-3]]
        
        recent_avg = np.mean(recent_scores) if recent_scores else 0
        older_avg = np.mean(older_scores) if older_scores else 0
        
        if recent_avg > older_avg + 0.1:
            return "amelioration"
        elif recent_avg < older_avg - 0.1:
            return "degradation"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un resume de l'historique des scores.
        
        Returns:
            Dictionnaire avec statistiques sur l'historique
        """
        history = self.load_history()
        
        if not history:
            return {
                'n_entries': 0,
                'mean_score': None,
                'max_score': None,
                'min_score': None,
                'trend': 'stable'
            }
        
        scores = [h['scores'].get('overall_score', 0) for h in history]
        
        return {
            'n_entries': len(history),
            'mean_score': round(np.mean(scores), 3),
            'max_score': round(np.max(scores), 3),
            'min_score': round(np.min(scores), 3),
            'std_score': round(np.std(scores), 3),
            'trend': self.get_trend(),
            'first_entry': history[0]['timestamp'],
            'last_entry': history[-1]['timestamp']
        }
    
    def export_history(self, format: str = 'csv') -> str:
        """
        Exporte l'historique dans differents formats.
        
        Args:
            format: 'csv' ou 'json'
            
        Returns:
            Chemin du fichier exporte
        """
        history = self.load_history()
        
        if not history:
            return None
        
        if format == 'csv':
            df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    'overall_score': h['scores'].get('overall_score', 0),
                    'confidence_level': h['scores'].get('confidence_level', ''),
                    **h['scores'].get('factors', {})
                }
                for h in history
            ])
            
            export_path = self.metrics_dir / "confidence_history.csv"
            df.to_csv(export_path, index=False)
            return str(export_path)
        
        else:  # json
            export_path = self.metrics_dir / "confidence_history_export.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            return str(export_path)
    
    def plot_trend(self, save_path: Optional[Path] = None):
        """
        Genere un graphique de l'evolution des scores de confiance.
        
        Args:
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        try:
            import matplotlib.pyplot as plt
            
            history = self.load_history()
            
            if not history:
                print("Aucune donnee historique disponible")
                return
            
            dates = [pd.to_datetime(h['timestamp']) for h in history]
            scores = [h['scores'].get('overall_score', 0) for h in history]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(dates, scores, marker='o', linewidth=2, markersize=4)
            ax.axhline(y=0.8, color='green', linestyle='--', label='Seuil eleve (0.8)')
            ax.axhline(y=0.6, color='orange', linestyle='--', label='Seuil moyen (0.6)')
            ax.axhline(y=0.4, color='red', linestyle='--', label='Seuil faible (0.4)')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Score de confiance')
            ax.set_title('Evolution du score de confiance')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Graphique sauvegarde : {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            print("Matplotlib non disponible pour le graphique")


def compute_confidence_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les scores de confiance a partir d'un DataFrame.
    
    Args:
        df: DataFrame avec colonnes 'n_articles', 'n_topics', 'n_periods', 'growth_std'
        
    Returns:
        DataFrame avec les colonnes de scores ajoutees
    """
    scorer = ConfidenceScorer()
    
    results = []
    for _, row in df.iterrows():
        data = {
            'n_articles': row.get('n_articles', 0),
            'n_topics': row.get('n_topics', 0),
            'n_periods': row.get('n_periods', 0),
            'growth_std': row.get('growth_std', 0)
        }
        score = scorer.compute(data)
        results.append(score)
    
    # Ajouter les scores au DataFrame
    df_result = df.copy()
    df_result['confidence_score'] = [r['overall_score'] for r in results]
    df_result['confidence_level'] = [r['confidence_level'] for r in results]
    
    return df_result


if __name__ == "__main__":
    # Test du module
    print("=" * 60)
    print("TEST DU MODULE CONFIDENCE SCORER")
    print("=" * 60)
    
    # Test avec des donnees realistes
    test_data = {
        'n_articles': 8656,
        'n_topics': 747,
        'n_periods': 33,
        'growth_std': 150.5
    }
    
    scorer = ConfidenceScorer()
    result = scorer.compute(test_data)
    
    print("\nDonnees d'entree :")
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    print("\nResultat :")
    print(f"  Score global: {result['overall_score']:.1%}")
    print(f"  Niveau: {result['confidence_level']}")
    print(f"  Interpretation: {result['interpretation']}")
    print("\n  Facteurs :")
    for factor, score in result['factors'].items():
        print(f"    {factor}: {score:.1%}")
    
    # Test du tracker
    print("\n" + "=" * 60)
    print("TEST DU CONFIDENCE TRACKER")
    print("=" * 60)
    
    tracker = ConfidenceTracker(Path("data/reports/confidence"))
    tracker.log(result, metadata={"version": "1.0", "source": "test"})
    
    summary = tracker.get_summary()
    print(f"\nResume historique:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nTendance: {tracker.get_trend()}")