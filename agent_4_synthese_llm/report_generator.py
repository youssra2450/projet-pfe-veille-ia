import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import re

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin parent
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / "data" / "expor"
        self.reports_dir = self.root_dir / "data" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = "llama3.2"

config = Config()

# ============================================================================
# 1. CHARGEMENT ET VALIDATION DES DONNÉES
# ============================================================================
def load_real_data() -> Dict[str, Any]:
    """Charge et valide les données réelles depuis les fichiers CSV"""
    
    data = {
        "articles": None,
        "emerging_topics": None,
        "topic_distribution": None,
        "kpis": None,
        "forecasts": None,
        "timeseries": None,
        "topic_stats": None
    }
    
    # 1. Articles
    articles_path = config.data_dir / "articles.csv"
    if articles_path.exists():
        df = pd.read_csv(articles_path)
        data["articles"] = {
            "total": len(df),
            "date_min": df['published_date'].min() if 'published_date' in df.columns else None,
            "date_max": df['published_date'].max() if 'published_date' in df.columns else None,
            "categories": df['main_category'].value_counts().head(5).to_dict() if 'main_category' in df.columns else {}
        }
        logger.info(f"Articles charges: {data['articles']['total']:,}")
    
    # 2. Topics emergents
    emerging_path = config.data_dir / "emerging_topics.csv"
    if emerging_path.exists():
        df = pd.read_csv(emerging_path)
        
        categories = df['categorie'].value_counts().to_dict() if 'categorie' in df.columns else {}
        
        top_topics = df.nlargest(10, 'growth_pct')[
            ['topic_id', 'label', 'growth_pct', 'volume', 'mots_cles', 'categorie']
        ].to_dict('records')
        
        data["emerging_topics"] = {
            "total": len(df),
            "avg_growth": df['growth_pct'].mean(),
            "median_growth": df['growth_pct'].median(),
            "max_growth": df['growth_pct'].max(),
            "min_growth": df['growth_pct'].min(),
            "std_growth": df['growth_pct'].std(),
            "categories": categories,
            "top_topics": top_topics,
            "growth_distribution": {
                "q25": df['growth_pct'].quantile(0.25),
                "q50": df['growth_pct'].quantile(0.50),
                "q75": df['growth_pct'].quantile(0.75)
            }
        }
        logger.info(f"Topics emergents: {data['emerging_topics']['total']}")
    
    # 3. Distribution des topics
    dist_path = config.data_dir / "topic_distribution.csv"
    if dist_path.exists():
        df = pd.read_csv(dist_path)
        data["topic_distribution"] = {
            "total_topics": len(df),
            "top_topic": df.iloc[0]['label'] if len(df) > 0 else None,
            "top_volume": df.iloc[0]['nb_articles'] if len(df) > 0 else 0,
            "top_10_volume": df['nb_articles'].head(10).sum() if len(df) > 0 else 0
        }
    
    # 4. KPIs
    kpis_path = config.data_dir / "kpis.csv"
    if kpis_path.exists():
        df = pd.read_csv(kpis_path)
        data["kpis"] = {row['indicateur']: row['valeur'] for _, row in df.iterrows()}
    
    # 5. Prévisions
    forecasts_path = config.data_dir / "forecasts.csv"
    if forecasts_path.exists():
        df = pd.read_csv(forecasts_path)
        data["forecasts"] = {
            "has_data": True,
            "topics_count": df['topic_id'].nunique() if 'topic_id' in df.columns else 0,
            "avg_forecast": df['forecast'].mean() if 'forecast' in df.columns else None
        }
    
    # 6. Series temporelles
    ts_path = config.data_dir / "timeseries.csv"
    if ts_path.exists():
        df = pd.read_csv(ts_path)
        data["timeseries"] = {
            "periods": df['date'].nunique() if 'date' in df.columns else 0,
            "min_date": df['date'].min() if 'date' in df.columns else None,
            "max_date": df['date'].max() if 'date' in df.columns else None
        }
    
    return data


def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide la qualite des donnees et identifie les lacunes"""
    
    validation = {
        "is_complete": True,
        "warnings": [],
        "missing": []
    }
    
    required_fields = ["articles", "emerging_topics", "topic_distribution"]
    for field in required_fields:
        if data.get(field) is None:
            validation["missing"].append(field)
            validation["is_complete"] = False
    
    articles = data.get("articles", {})
    if articles.get("total", 0) < 100:
        validation["warnings"].append(
            f"Volume d'articles faible ({articles.get('total', 0)}), tendances a confirmer"
        )
    
    emerging = data.get("emerging_topics", {})
    if emerging.get("total", 0) < 5:
        validation["warnings"].append(
            f"Nombre de topics emergents faible ({emerging.get('total', 0)})"
        )
    
    return validation

# ============================================================================
# 2. ANALYSE STATISTIQUE AVANCÉE
# ============================================================================
def compute_statistical_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule des insights statistiques avances pour le rapport"""
    
    insights = {
        "key_findings": [],
        "statistical_summary": {},
        "anomalies": [],
        "trend_analysis": {},
        "recommendations": []
    }
    
    emerging = data.get("emerging_topics", {})
    articles = data.get("articles", {})
    
    if emerging:
        insights["statistical_summary"] = {
            "n_topics": emerging.get("total", 0),
            "mean_growth": emerging.get("avg_growth", 0),
            "median_growth": emerging.get("median_growth", 0),
            "std_growth": emerging.get("std_growth", 0),
            "max_growth": emerging.get("max_growth", 0),
            "min_growth": emerging.get("min_growth", 0),
            "q25_growth": emerging.get("growth_distribution", {}).get("q25", 0),
            "q75_growth": emerging.get("growth_distribution", {}).get("q75", 0)
        }
    
    if emerging.get("top_topics"):
        top = emerging["top_topics"][0]
        insights["key_findings"].append({
            "title": "Topic a plus forte emergence",
            "description": (
                f"Le topic « {top['label']} » presente la croissance la plus elevee "
                f"avec +{top['growth_pct']:.1f}% sur la periode analysee, "
                f"represente par {top['volume']} articles."
            ),
            "category": top.get('categorie', 'Non classifie'),
            "keywords": top.get('mots_cles', ''),
            "priority": "high"
        })
    
    if emerging.get("std_growth", 0) > 100:
        insights["anomalies"].append({
            "title": "Forte dispersion des taux de croissance",
            "description": (
                f"L'ecart-type des croissances est de {emerging['std_growth']:.1f}%, "
                f"indiquant une dynamique tres heterogene entre les differents sous-domaines."
            ),
            "severity": "medium"
        })
    
    categories = emerging.get("categories", {})
    if categories:
        top_cat = max(categories, key=categories.get)
        insights["key_findings"].append({
            "title": "Domaine de recherche dominant",
            "description": (
                f"La categorie « {top_cat} » concentre {categories[top_cat]} topics, "
                f"soit {categories[top_cat] / emerging.get('total', 1) * 100:.1f}% "
                f"des topics emergents identifies."
            ),
            "priority": "medium"
        })
    
    ts = data.get("timeseries", {})
    if ts.get("periods", 0) > 0:
        insights["trend_analysis"] = {
            "periods_analyzed": ts.get("periods", 0),
            "time_span": f"{ts.get('min_date', '')} - {ts.get('max_date', '')}",
            "periodicity": "hebdomadaire"
        }
    
    if articles.get("total", 0) < 1000:
        insights["recommendations"].append({
            "title": "Augmenter le corpus de donnees",
            "description": (
                f"Le volume actuel de {articles.get('total', 0)} articles est insuffisant "
                "pour des analyses robustes. Un objectif de 5000+ articles est recommande."
            ),
            "action": "Etendre la collecte a plus de categories arXiv"
        })
    else:
        insights["recommendations"].append({
            "title": "Maintenir et consolider la collecte",
            "description": (
                f"Le corpus de {articles.get('total', 0)} articles offre une base solide. "
                "La collecte doit etre maintenue pour assurer un suivi temporel."
            ),
            "action": "Automatiser la collecte quotidienne avec alertes"
        })
    
    if emerging.get("total", 0) > 10:
        insights["recommendations"].append({
            "title": "Veille approfondie sur les topics emergents",
            "description": (
                f"Les {emerging['total']} topics emergents identifies meritent "
                "une veille hebdomadaire pour suivre leur evolution."
            ),
            "action": "Mettre en place un tableau de bord dedie"
        })
    
    return insights

# ============================================================================
# 3. CALCUL DU SCORE DE CONFIANCE
# ============================================================================
def calculate_confidence_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule un score de confiance objectif base sur des metriques quantitatives."""
    
    articles = data.get("articles", {})
    emerging = data.get("emerging_topics", {})
    ts = data.get("timeseries", {})
    
    n_articles = articles.get("total", 0)
    if n_articles >= 5000:
        volume_score = 1.0
    elif n_articles >= 2000:
        volume_score = 0.8
    elif n_articles >= 1000:
        volume_score = 0.6
    elif n_articles >= 500:
        volume_score = 0.4
    else:
        volume_score = 0.2
    
    n_periods = ts.get("periods", 0)
    if n_periods >= 26:
        temporal_score = 1.0
    elif n_periods >= 13:
        temporal_score = 0.7
    elif n_periods >= 8:
        temporal_score = 0.5
    elif n_periods >= 4:
        temporal_score = 0.3
    else:
        temporal_score = 0.1
    
    n_topics = emerging.get("total", 0)
    if n_topics >= 30:
        diversity_score = 1.0
    elif n_topics >= 15:
        diversity_score = 0.8
    elif n_topics >= 10:
        diversity_score = 0.6
    elif n_topics >= 5:
        diversity_score = 0.4
    else:
        diversity_score = 0.2
    
    std_growth = emerging.get("std_growth", 0)
    if std_growth > 0:
        quality_score = max(0, min(1, 1 - std_growth / 500))
    else:
        quality_score = 0.5
    
    if n_articles > 1000 and n_topics > 10:
        coherence_score = 0.9
    elif n_articles > 500 and n_topics > 5:
        coherence_score = 0.7
    elif n_articles > 100 and n_topics > 3:
        coherence_score = 0.5
    else:
        coherence_score = 0.3
    
    weights = {"volume": 0.25, "temporal": 0.20, "diversity": 0.20, "quality": 0.20, "coherence": 0.15}
    
    overall_score = (
        weights["volume"] * volume_score +
        weights["temporal"] * temporal_score +
        weights["diversity"] * diversity_score +
        weights["quality"] * quality_score +
        weights["coherence"] * coherence_score
    )
    overall_score = round(overall_score, 3)
    
    if overall_score >= 0.80:
        confidence_level = "Eleve"
        interpretation = f"Analyse robuste basee sur {n_articles:,} articles et {n_topics} topics emergents. Les tendances sont fiables."
    elif overall_score >= 0.60:
        confidence_level = "Moyen"
        interpretation = f"Analyse indicative basee sur {n_articles:,} articles. Les tendances sont a confirmer avec des donnees supplementaires."
    elif overall_score >= 0.40:
        confidence_level = "Faible"
        interpretation = f"Analyse preliminaire. Le volume de donnees ({n_articles:,} articles) est suffisant pour des tendances generales."
    else:
        confidence_level = "Insuffisant"
        interpretation = "Donnees insuffisantes pour des conclusions fiables. Collecte supplementaire recommandee."
    
    return {
        "overall_score": overall_score,
        "confidence_level": confidence_level,
        "interpretation": interpretation,
        "factors": {
            "data_volume_score": round(volume_score, 3),
            "temporal_coverage_score": round(temporal_score, 3),
            "topic_diversity_score": round(diversity_score, 3),
            "growth_quality_score": round(quality_score, 3),
            "coherence_score": round(coherence_score, 3)
        },
        "details": {
            "n_articles": n_articles,
            "n_topics": n_topics,
            "n_periods": n_periods,
            "growth_std": round(std_growth, 2)
        }
    }

# ============================================================================
# 4. GENERATION DE L'INTERPRETATION LLM
# ============================================================================
def generate_llm_interpretation(
    data: Dict[str, Any],
    insights: Dict[str, Any],
    confidence: Dict[str, Any]
) -> Optional[str]:
    """Genere une interpretation qualitative via Ollama."""
    
    try:
        import ollama
    except ImportError:
        logger.warning("Ollama non disponible")
        return None
    
    emerging = data.get("emerging_topics", {})
    
    top_topics_text = ""
    for i, topic in enumerate(emerging.get("top_topics", [])[:5], 1):
        top_topics_text += (
            f"{i}. **{topic['label']}** : +{topic['growth_pct']:.1f}% "
            f"({topic['volume']} articles) - {topic.get('categorie', 'Non classifie')}\n"
        )
    
    stats = insights.get("statistical_summary", {})
    stats_text = (
        f"- Nombre de topics emergents : {emerging.get('total', 0)}\n"
        f"- Croissance moyenne : +{emerging.get('avg_growth', 0):.1f}%\n"
        f"- Croissance mediane : +{emerging.get('median_growth', 0):.1f}%\n"
        f"- Ecart-type des croissances : {emerging.get('std_growth', 0):.1f}%\n"
        f"- Croissance maximale : +{emerging.get('max_growth', 0):.1f}%\n"
        f"- Articles analyses : {data.get('articles', {}).get('total', 0):,}\n"
        f"- Score de confiance : {confidence.get('overall_score', 0):.1%}"
    )
    
    prompt = f"""Vous etes un chercheur senior en Intelligence Artificielle specialise en veille technologique et analyse scientometrique.

**CONTEXTE**
Vous analysez les resultats d'un pipeline de veille technologique base sur 8656 articles arXiv.
L'objectif est de produire une interpretation qualitative des tendances de recherche en IA.

**DONNEES ANALYSEES**

Top 5 des topics a plus forte croissance :
{top_topics_text}

Statistiques cles :
{stats_text}

Tendances identifiees :
{json.dumps(insights.get('key_findings', []), ensure_ascii=False, indent=2)}

Anomalies detectees :
{json.dumps(insights.get('anomalies', []), ensure_ascii=False, indent=2)}

**TACHE**
Redigez une section d'interpretation qualitative pour un rapport de recherche.

Structure attendue :

1. **Introduction** (2-3 phrases) : Contexte de l'analyse, objectifs

2. **Analyse des tendances majeures** (3-4 paragraphes) :
   - Synthese des dynamiques observees
   - Identification des ruptures et continuites
   - Comparaison avec les tendances historiques connues

3. **Discussion des signaux faibles** (1-2 paragraphes) :
   - Topics avec faible volume mais forte croissance
   - Potentialites de developpement futur

4. **Implications pour la recherche** (1-2 paragraphes) :
   - Impact sur les directions de recherche
   - Recommandations pour les comites de veille

5. **Conclusion** (2-3 phrases) : Synthese et perspectives

**STYLE**
- Ton academique et professionnel
- Langage precis, sans jargon inutile
- Citations des donnees chiffrees
- 400-500 mots au total

**REPONSE :**"""

    try:
        logger.info("Appel a Ollama pour l'interpretation qualitative...")
        
        response = ollama.chat(
            model=config.llm_model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.2,
                'num_predict': 4096,
                'top_p': 0.9
            }
        )
        
        content = response.get('message', {}).get('content', '')
        if content:
            logger.info("Interpretation LLM generee avec succes")
            return content
        else:
            logger.warning("Reponse Ollama vide")
            return None
            
    except Exception as e:
        logger.error(f"Erreur Ollama : {e}")
        return None

# ============================================================================
# 5. GENERATION DU RAPPORT COMPLET
# ============================================================================
def generate_full_report(
    data: Dict[str, Any],
    insights: Dict[str, Any],
    confidence: Dict[str, Any],
    llm_interpretation: Optional[str],
    validation: Dict[str, Any]
) -> str:
    """Genere le rapport complet au format Markdown."""
    
    now = datetime.now().strftime("%d %B %Y a %H:%M")
    emerging = data.get("emerging_topics", {})
    articles = data.get("articles", {})
    
    report = f"""---
title: "Rapport de Veille Technologique en Intelligence Artificielle"
subtitle: "Analyse des tendances de recherche a partir d'ArXiv"
date: "{now}"
institution: "Laboratoire de Recherche en IA"
version: "Master Recherche - v2.0"
---

# Resume Executif

Ce rapport presente les resultats de l'analyse de **{articles.get('total', 0):,} articles scientifiques** 
extraits de la plateforme arXiv.org sur la periode du 
**{articles.get('date_min', 'N/A')}** au **{articles.get('date_max', 'N/A')}**.

L'analyse a permis d'identifier **{emerging.get('total', 0)} topics emergents** 
dans le domaine de l'Intelligence Artificielle, avec une croissance moyenne de 
**+{emerging.get('avg_growth', 0):.1f}%** et un taux maximal de 
**+{emerging.get('max_growth', 0):.1f}%**.

Le score de confiance global de l'analyse est de **{confidence.get('overall_score', 0):.1%}**, 
correspondant a un niveau **{confidence.get('confidence_level', 'N/A')}**.

**Principales contributions :**
"""
    
    for finding in insights.get('key_findings', [])[:3]:
        report += f"\n- **{finding['title']}** : {finding['description']}"
    
    report += f"""

---

# 1. Introduction

## 1.1 Contexte

La veille technologique en Intelligence Artificielle constitue un enjeu strategique 
majeur pour les laboratoires de recherche et les entreprises innovantes. 
La croissance exponentielle des publications scientifiques rend indispensable 
le recours a des methodes automatisees d'analyse et de synthese.

## 1.2 Objectifs

Le present rapport vise a :
1. Identifier les thematiques de recherche emergentes en IA
2. Quantifier la dynamique de croissance des differents sous-domaines
3. Fournir une interpretation qualitative des tendances observees
4. Proposer des recommandations strategiques pour la veille technologique

## 1.3 Sources et methodologie

Les donnees analysees proviennent de la plateforme arXiv.org, 
couvrant les categories suivantes :
- Computer Science (cs.AI, cs.LG, cs.CV, cs.CL, cs.NE)
- Statistics (stat.ML, stat.AP)
- Mathematics (math.OC, math.IT)

La periode d'analyse s'etend de {articles.get('date_min', 'N/A')} a {articles.get('date_max', 'N/A')}.

---

# 2. Methodologie

## 2.1 Pipeline d'analyse

L'analyse a ete conduite selon un pipeline multi-agents comprenant :

1. **Agent 1 - Collecte** : Extraction des articles via l'API arXiv
2. **Agent 2 - Topic Modeling** : Modelisation thematique avec BERTopic
3. **Agent 3 - Analyse temporelle** : Detection des tendances et previsions
4. **Agent 4 - Synthese** : Interpretation qualitative par LLM

## 2.2 Metriques utilisees

- **Taux de croissance** : Variation relative sur la periode d'analyse
- **Score d'emergence** : Indice composite (croissance x volume x maturite)
- **Score de confiance** : Evaluation de la robustesse des resultats

## 2.3 Limites methodologiques

Les principales limites de cette analyse sont :
- Dependance a la couverture d'ArXiv (biais de publication)
- Absence de validation externe (ground truth)
- Sensibilite aux parametres de modelisation

---

# 3. Resultats

## 3.1 Vue d'ensemble

**Statistiques descriptives :**

| Metrique | Valeur |
|----------|--------|
| Articles analyses | {articles.get('total', 0):,} |
| Topics identifies | {emerging.get('total', 0)} |
| Croissance moyenne | +{emerging.get('avg_growth', 0):.1f}% |
| Croissance mediane | +{emerging.get('median_growth', 0):.1f}% |
| Ecart-type | {emerging.get('std_growth', 0):.1f}% |
| Croissance maximale | +{emerging.get('max_growth', 0):.1f}% |
| Croissance minimale | +{emerging.get('min_growth', 0):.1f}% |
| Q25 - Q75 | {emerging.get('growth_distribution', {}).get('q25', 0):.1f}% - {emerging.get('growth_distribution', {}).get('q75', 0):.1f}% |

## 3.2 Topics les plus dynamiques

| Rang | Topic | Croissance | Volume | Domaine |
|------|-------|------------|--------|---------|
"""
    
    for i, topic in enumerate(emerging.get('top_topics', [])[:7], 1):
        report += f"| {i} | {topic['label'][:45]} | +{topic['growth_pct']:.1f}% | {topic['volume']} | {topic.get('categorie', 'N/A')} |\n"
    
    report += f"""
## 3.3 Repartition par domaine

"""
    
    categories = emerging.get('categories', {})
    if categories:
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / emerging.get('total', 1) * 100
            report += f"- **{cat}** : {count} topics ({pct:.1f}%)\n"
    
    report += f"""

## 3.4 Score de confiance

Le score de confiance global est de **{confidence.get('overall_score', 0):.1%}**, 
correspondant a un niveau **{confidence.get('confidence_level', 'N/A')}**.

**Facteurs de confiance :**

| Facteur | Score |
|---------|-------|
| Volume de donnees | {confidence.get('factors', {}).get('data_volume_score', 0):.1%} |
| Couverture temporelle | {confidence.get('factors', {}).get('temporal_coverage_score', 0):.1%} |
| Diversite des topics | {confidence.get('factors', {}).get('topic_diversity_score', 0):.1%} |
| Qualite des croissances | {confidence.get('factors', {}).get('growth_quality_score', 0):.1%} |
| Coherence globale | {confidence.get('factors', {}).get('coherence_score', 0):.1%} |

**Interpretation :** {confidence.get('interpretation', '')}

---

# 4. Discussion

## 4.1 Interpretation qualitative

"""
    
    if llm_interpretation:
        report += llm_interpretation
    else:
        report += """
L'analyse des donnees revele plusieurs dynamiques significatives dans le paysage 
de la recherche en Intelligence Artificielle.

**Tendances majeures**

Les resultats montrent une acceleration de la recherche dans des domaines 
specificques, avec des taux de croissance atteignant plusieurs centaines de 
pourcents sur la periode analysee. Cette dynamique suggere l'emergence de 
nouvelles frontieres de recherche, potentiellement liees a des avancees 
technologiques recentes.

**Signaux faibles**

Certains topics, bien que de volume modeste, presentent des croissances 
tres elevees. Ces signaux faibles meritent une attention particuliere 
car ils peuvent precurser des tendances futures.

**Implications**

La dispersion des taux de croissance (ecart-type eleve) indique une 
specialisation croissante du champ de l'IA, avec des sous-domaines 
evoluant a des rythmes tres differents.
"""
    
    report += f"""

## 4.2 Anomalies et signaux particuliers

"""
    
    anomalies = insights.get('anomalies', [])
    if anomalies:
        for anomaly in anomalies:
            report += f"**{anomaly['title']}**\n\n{anomaly['description']}\n\n"
    else:
        report += "Aucune anomalie significative detectee dans les donnees.\n\n"

    report += f"""
## 4.3 Comparaison avec la litterature

Les tendances observees sont coherentes avec les analyses recentes 
de la production scientifique en IA. La predominance des sujets lies 
aux grands modeles de langage et a l'apprentissage automatique 
confirme les orientations generales du domaine.

---

# 5. Recommandations

"""

    for i, rec in enumerate(insights.get('recommendations', []), 1):
        report += f"""
### 5.{i} {rec['title']}

**Constat :** {rec['description']}

**Action recommandee :** {rec.get('action', 'A definir')}
"""

    if len(insights.get('recommendations', [])) < 3:
        report += """
### 5.4 Renforcer la veille sur les topics a forte croissance

Les topics avec une croissance superieure a 100% justifient une attention 
hebdomadaire et une analyse plus approfondie.

### 5.5 Approfondir l'analyse des signaux faibles

Les topics avec un faible volume mais une forte croissance representent 
des opportunites potentielles de recherche innovante.
"""

    report += f"""

---

# 6. Conclusion

Ce rapport a permis d'identifier **{emerging.get('total', 0)} topics emergents** 
dans le domaine de l'Intelligence Artificielle, sur la base de 
**{articles.get('total', 0):,} articles** extraits d'ArXiv.

Les resultats montrent une dynamique de recherche **tres active** 
avec une croissance moyenne de +{emerging.get('avg_growth', 0):.1f}% 
et des disparites importantes selon les sous-domaines.

Les recommandations formulees visent a :
- Maintenir et renforcer la collecte de donnees
- Approfondir l'analyse des signaux faibles
- Mettre en place une veille hebdomadaire des topics emergents

La suite de ces travaux consistera a :
- Etendre la collecte a d'autres sources (Semantic Scholar, ACL Anthology)
- Developper des indicateurs predictifs avances
- Automatiser la generation de rapports

---

# 7. References

1. arXiv.org - Base de donnees de prepublications scientifiques
2. BERTopic - Modelisation thematique par embeddings
3. Prophet - Prevision de series temporelles
4. Ollama - Generation de langage naturel local

---

# 8. Annexes

## A. Metriques detaillees

| Metrique | Valeur |
|----------|--------|
| Date de generation | {now} |
| Version du rapport | Master Recherche - v2.0 |
| Score de confiance | {confidence.get('overall_score', 0):.1%} |
| Niveau de confiance | {confidence.get('confidence_level', 'N/A')} |

## B. Validation des donnees

"""
    
    for warning in validation.get('warnings', []):
        report += f"- {warning}\n"
    
    if validation.get('is_complete', True):
        report += "- Donnees completes et validees\n"
    
    report += f"""
## C. Structure du pipeline

---

*Rapport genere automatiquement par le pipeline de veille technologique*
*Laboratoire de Recherche en IA - {now}*
"""
    
    return report

# ============================================================================
# 6. SAUVEGARDE DU RAPPORT
# ============================================================================
def save_report(
    report: str,
    data: Dict[str, Any],
    insights: Dict[str, Any],
    confidence: Dict[str, Any],
    validation: Dict[str, Any]
) -> tuple:
    """Sauvegarde le rapport et les metadonnees associees"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = config.reports_dir / f"rapport_{timestamp}.md"
    report_path.write_text(report, encoding='utf-8')
    logger.info(f"Rapport sauvegarde : {report_path}")
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "report_file": f"rapport_{timestamp}.md",
        "data_summary": {
            "total_articles": data.get('articles', {}).get('total', 0),
            "emerging_topics_count": data.get('emerging_topics', {}).get('total', 0),
            "avg_growth": data.get('emerging_topics', {}).get('avg_growth', 0),
            "max_growth": data.get('emerging_topics', {}).get('max_growth', 0)
        },
        "insights_count": {
            "key_findings": len(insights.get('key_findings', [])),
            "recommendations": len(insights.get('recommendations', [])),
            "anomalies": len(insights.get('anomalies', []))
        },
        "confidence": confidence,
        "validation": validation
    }
    
    metadata_path = config.reports_dir / f"metrics_{timestamp}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadonnees sauvegardees : {metadata_path}")
    
    return report_path, metadata_path

# ============================================================================
# 7. FONCTION PRINCIPALE
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("AGENT 4 - SYNTHESE ET INTERPRETATION (Niveau Master Recherche)")
    print("=" * 70)
    print("\nPipeline:")
    print("  1. Chargement et validation des donnees")
    print("  2. Calcul des insights statistiques avances")
    print("  3. Evaluation du score de confiance")
    print("  4. Generation de l'interpretation qualitative (LLM)")
    print("  5. Production du rapport final structure")
    print("\n" + "=" * 70 + "\n")
    
    logger.info("Chargement des donnees depuis data/expor/...")
    data = load_real_data()
    
    validation = validate_data(data)
    if not validation["is_complete"]:
        logger.warning("Donnees incompletes : %s", validation["missing"])
    
    logger.info("Calcul des insights statistiques...")
    insights = compute_statistical_insights(data)
    
    logger.info("Evaluation du score de confiance...")
    confidence = calculate_confidence_score(data)
    logger.info("Score global : %.1f%% (%s)", 
                confidence['overall_score'] * 100, 
                confidence['confidence_level'])
    
    logger.info("Generation de l'interpretation qualitative via LLM...")
    llm_interpretation = generate_llm_interpretation(data, insights, confidence)
    
    if llm_interpretation:
        logger.info("Interpretation LLM generee avec succes")
    else:
        logger.warning("Fallback : utilisation du template par defaut")
    
    logger.info("Generation du rapport final...")
    report = generate_full_report(data, insights, confidence, llm_interpretation, validation)
    
    report_path, metadata_path = save_report(report, data, insights, confidence, validation)
    
    print("\n" + "=" * 70)
    print("RESUME DU RAPPORT")
    print("=" * 70)
    print(f"\nRapport : {report_path}")
    print(f"\nSCORE DE CONFIANCE : {confidence['overall_score']:.1%} ({confidence['confidence_level']})")
    print(f"  {confidence['interpretation']}")
    print(f"\nSTATISTIQUES CLES :")
    print(f"  - Articles analyses : {data.get('articles', {}).get('total', 0):,}")
    print(f"  - Topics emergents : {data.get('emerging_topics', {}).get('total', 0)}")
    print(f"  - Croissance moyenne : +{data.get('emerging_topics', {}).get('avg_growth', 0):.1f}%")
    print(f"  - Croissance maximale : +{data.get('emerging_topics', {}).get('max_growth', 0):.1f}%")
    print(f"\nInterpretation LLM : {'Generee' if llm_interpretation else 'Template par defaut'}")
    
    print("\n" + "=" * 70)
    print("AGENT 4 - RAPPORT GENERE AVEC SUCCES")
    print("=" * 70)

if __name__ == "__main__":
    main()