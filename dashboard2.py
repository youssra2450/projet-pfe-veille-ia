"""
dashboard.py
Tableau de bord intelligent de veille technologique IA
Pipeline multi-agents — Analyse des tendances de recherche

Structure en trois piliers demandés par l'encadrant :
  1. Visualisation       -> onglet "Visualisation"
  2. Interprétation auto  -> onglet "Interprétation automatique"
                             (règles instantanées + Agent 4 / Ollama)
  3. Aide à la décision   -> onglet "Aide à la décision"
                             (priorisation par règles + Agent 4)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import json
import re

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE & THÈME SOMBRE
# ==============================================================================
st.set_page_config(
    page_title="Veille Scientifique - arXiv Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0A0F1D;
        color: #F3F4F6;
    }
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #94A3B8 !important;
    }
    .pbi-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .kpi-num-title {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-main-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F3F4F6;
        margin-bottom: 8px;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F3F4F6;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .graph-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F3F4F6;
        margin-bottom: 2px;
    }
    .graph-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 15px;
    }
    .insight-card {
        background-color: #111827;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #E2E8F0;
    }
    .insight-title {
        font-weight: 700;
        margin-bottom: 6px;
        display: block;
    }
    .reco-card {
        background-color: #111827;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #E2E8F0;
    }
    .reco-title {
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        border-radius: 12px;
        padding: 1px 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
    }
    .llm-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .confidence-high   { color: #10B981; font-weight: bold; }
    .confidence-medium { color: #F59E0B; font-weight: bold; }
    .confidence-low    { color: #EF4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CHEMINS & CONSTANTES
# ==============================================================================
DATA_DIR = Path("data/powerbi")
REPORTS_DIR = Path("data/reports")

COULEUR_INSIGHT = {
    "info": "#3B82F6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}
COULEUR_PRIO = {"urgent": "#EF4444", "normal": "#F59E0B", "ok": "#10B981"}
LABEL_PRIO = {
    "urgent": "Priorité haute",
    "normal": "Priorité moyenne",
    "ok": "À surveiller",
}


# ==============================================================================
# 3. FONCTIONS UTILITAIRES
# ==============================================================================
def get_kpi(df_kpis, nom, defaut=0.0):
    if df_kpis is None:
        return defaut
    ligne = df_kpis[df_kpis["indicateur"] == nom]
    if ligne.empty:
        return defaut
    return ligne.iloc[0]["valeur"]


def classer_domaine(label, mots_cles):
    texte = f"{label} {mots_cles}".lower()

    regles = [
        ("Vision & Multimodal",
         ["vision", "image", "multimodal", "vlm", "visual", "reconstruction", "scene", "geometry", "video", "generation"]),
        ("Agents & Renforcement",
         ["agent", "reinforcement", "reward", "rl", "rlvr", "rollout", "orchestrat", "policy", "execution"]),
        ("Sécurité IA",
         ["safety", "adversarial", "attack", "robust", "guard", "vulnerab", "security"]),
        ("Calcul efficace",
         ["spike", "neuromorphic", "energy", "efficient", "edge", "quantum", "channel", "states"]),
        ("NLP & Recherche d'information",
         ["retrieval", "rag", "language", "nlp", "token", "text", "document"]),
        ("Optimisation",
         ["evolutionary", "optimization", "pareto", "swarm", "moea", "nsga", "riemannian", "matrix", "low-rank"]),
        ("Machine Learning",
         ["learning", "neural", "network", "deep", "classification", "regression", "clustering", "gaussian", "mixture", "bayesian", "bandit", "regret"]),
        ("Mathématiques & Théorie",
         ["extremal", "algebraic", "statistics", "causal", "structural", "discovery", "entropy", "information"]),
        ("Applications",
         ["clinical", "medical", "students", "feedback", "software", "engineering", "lidar", "perception", "autonomous", "driving"]),
    ]

    for domaine, mots in regles:
        if any(m in texte for m in mots):
            return domaine
    return "Autre"


def classer_priorite(growth, moyenne, ecart):
    if ecart == 0:
        return "normal"
    z = (growth - moyenne) / ecart
    if z > 1:
        return "urgent"
    elif z > 0:
        return "normal"
    return "ok"


# ==============================================================================
# 4. CHARGEMENT DES DONNÉES
# ==============================================================================
@st.cache_data(ttl=3600)
def load_all_data():
    """Charge toutes les données depuis les fichiers CSV exportés."""
    try:
        df_emerging = pd.read_csv(DATA_DIR / "emerging_topics.csv")
        df_articles = pd.read_csv(DATA_DIR / "articles.csv")
        df_articles_topics = pd.read_csv(DATA_DIR / "articles_topics.csv")
        df_kpis = pd.read_csv(DATA_DIR / "kpis.csv")
        df_topic_dist = pd.read_csv(DATA_DIR / "topic_distribution.csv")
    except FileNotFoundError as e:
        st.error(
            f"Fichier introuvable : {e.filename}\n\n"
            "Lancez d'abord `python export_for_powerbi.py`."
        )
        return None

    # Série temporelle (optionnelle)
    try:
        df_ts = pd.read_csv(DATA_DIR / "timeseries.csv")
        df_ts["date"] = pd.to_datetime(df_ts["date"])
    except FileNotFoundError:
        df_ts = None

    # --- Enrichissement de df_emerging -------------------------------
    df_articles_topics["topic_id"] = df_articles_topics["topic_id"].astype(int)
    df_emerging["topic_id"] = df_emerging["topic_id"].astype(int)

    # ✅ CORRECTION : Utiliser la colonne volume déjà présente dans emerging_topics.csv
    if 'volume' in df_emerging.columns:
        # La colonne existe déjà (dans le CSV), on la garde
        df_emerging["volume"] = df_emerging["volume"].astype(int)
    else:
        # Sinon on la calcule
        volumes = df_articles_topics["topic_id"].value_counts().reset_index()
        volumes.columns = ["topic_id", "volume"]
        df_emerging = df_emerging.merge(volumes, on="topic_id", how="left")
        df_emerging["volume"] = df_emerging["volume"].fillna(0).astype(int)

    df_emerging["categorie"] = df_emerging.apply(
        lambda r: classer_domaine(r["label"], str(r.get("mots_cles", ""))),
        axis=1
    )

    avg_growth = df_emerging["growth_pct"].mean() if len(df_emerging) else 0
    max_growth = df_emerging["growth_pct"].max() if len(df_emerging) else 0
    std_growth = df_emerging["growth_pct"].std() if len(df_emerging) > 1 else 0
    if np.isnan(std_growth):
        std_growth = 0

    return {
        "emerging": df_emerging,
        "emerging_full": df_emerging,
        "articles": df_articles,
        "articles_topics": df_articles_topics,
        "kpis": df_kpis,
        "topic_dist": df_topic_dist,
        "timeseries": df_ts,
        "avg_growth": avg_growth,
        "max_growth": max_growth,
        "std_growth": std_growth,
    }


def load_agent4_report():
    """Charge le dernier rapport généré par l'Agent 4."""
    if not REPORTS_DIR.exists():
        return None

    metrics_files = [
        item for item in REPORTS_DIR.iterdir()
        if item.is_file() and item.name.startswith("metrics_") and item.name.endswith(".json")
    ]
    if not metrics_files:
        return None

    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(latest_metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        st.warning(f"Erreur lecture métriques : {e}")
        return None

    report_filename = metrics.get("report_file", "")
    if not report_filename:
        report_files = [
            item for item in REPORTS_DIR.iterdir()
            if item.is_file() and item.name.startswith("rapport_") and item.name.endswith(".md")
        ]
        if not report_files:
            return None
        report_file = max(report_files, key=lambda x: x.stat().st_mtime)
    else:
        report_file = REPORTS_DIR / report_filename
        if not report_file.exists() or not report_file.is_file():
            return None

    try:
        content = report_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            report_content = parts[2].strip() if len(parts) >= 3 else content
        else:
            report_content = content
    except Exception as e:
        st.warning(f"Erreur lecture rapport : {e}")
        return None

    return {
        "confidence": metrics.get("confidence", {}),
        "report": report_content,
        "generated_at": metrics.get("generated_at", ""),
        "report_file": str(report_file),
    }


def extract_sections_from_report(report_text):
    sections = {"executive_summary": "", "recommendations": "", "limitations": ""}

    if "## 1. Résumé Exécutif" in report_text:
        parts = report_text.split("## 1. Résumé Exécutif")
        if len(parts) > 1:
            end = parts[1].find("## 2.")
            sections["executive_summary"] = parts[1][:end].strip() if end > 0 else parts[1].strip()

    if "## 4. Recommandations" in report_text:
        parts = report_text.split("## 4. Recommandations")
        if len(parts) > 1:
            end = parts[1].find("## 5.")
            sections["recommendations"] = parts[1][:end].strip() if end > 0 else parts[1].strip()
    elif "## 6. Recommandations" in report_text:
        parts = report_text.split("## 6. Recommandations")
        if len(parts) > 1:
            end = parts[1].find("## 7.")
            sections["recommendations"] = parts[1][:end].strip() if end > 0 else parts[1].strip()

    if "## 5. Limitations" in report_text:
        parts = report_text.split("## 5. Limitations")
        if len(parts) > 1:
            end = parts[1].find("## 6.") if "## 6." in parts[1] else parts[1].find("## Annexe")
            sections["limitations"] = parts[1][:end].strip() if end > 0 else parts[1].strip()
    elif "## 7. Limitations" in report_text:
        parts = report_text.split("## 7. Limitations")
        if len(parts) > 1:
            end = parts[1].find("## Annexe")
            sections["limitations"] = parts[1][:end].strip() if end > 0 else parts[1].strip()

    return sections


# ==============================================================================
# 5. INTERPRÉTATION AUTOMATIQUE PAR RÈGLES
# ==============================================================================
def generer_insights_regles(data):
    insights = []
    df_dist = data["topic_dist"]
    df_emerging = data["emerging_full"]
    df_kpis = data["kpis"]

    if df_dist is not None and len(df_dist):
        top = df_dist.sort_values("nb_articles", ascending=False).iloc[0]
        total_articles = df_dist['nb_articles'].sum() if 'nb_articles' in df_dist.columns else 1
        proportion = (top['nb_articles'] / total_articles) * 100
        insights.append({
            "type": "info",
            "titre": "Thème dominant du corpus",
            "texte": f"Le topic « {top['label']} » regroupe {int(top['nb_articles'])} articles, soit {proportion:.1f} % du corpus analysé.",
        })

    if len(df_emerging):
        top_e = df_emerging.sort_values("growth_pct", ascending=False).iloc[0]
        insights.append({
            "type": "success",
            "titre": "Signal d'émergence le plus fort",
            "texte": f"« {top_e['label']} » affiche la croissance la plus élevée du corpus (+{top_e['growth_pct']:.0f} %), avec {int(top_e['volume'])} articles déjà recensés. Mots-clés associés : {top_e['mots_cles']}.",
        })

        moy = data["avg_growth"]
        ecart = data["std_growth"]
        insights.append({
            "type": "info",
            "titre": "Dynamique générale des topics émergents",
            "texte": f"Les {len(df_emerging)} topics émergents affichent une croissance moyenne de +{moy:.0f} % (écart-type : {ecart:.0f} points).",
        })

    outliers = get_kpi(df_kpis, "Outliers BERTopic", None)
    n_topics = get_kpi(df_kpis, "Topics BERTopic", None)
    if outliers is not None and n_topics is not None:
        niveau = "warning" if outliers > 25 else "info"
        insights.append({
            "type": niveau,
            "titre": "Granularité du topic modeling",
            "texte": f"BERTopic identifie {int(n_topics)} topics distincts, avec un taux d'outliers de {outliers:.1f} %.",
        })

    score_conf = get_kpi(df_kpis, "Score de confiance", None)
    if score_conf is not None:
        niveau = "success" if score_conf >= 75 else "warning" if score_conf >= 50 else "danger"
        insights.append({
            "type": niveau,
            "titre": "Fiabilité de l'analyse temporelle",
            "texte": f"Le score de confiance global est de {score_conf:.1f} %.",
        })

    return insights


def generer_recommandations_regles(data):
    df_emerging = data["emerging_full"]
    if df_emerging.empty:
        return []

    moyenne = data["avg_growth"]
    ecart = data["std_growth"]

    recos = []
    for _, row in df_emerging.sort_values("growth_pct", ascending=False).iterrows():
        recos.append({
            "priorite": classer_priorite(row["growth_pct"], moyenne, ecart),
            "label": row["label"],
            "categorie": row["categorie"],
            "growth_pct": row["growth_pct"],
            "volume": int(row["volume"]),
            "mots_cles": row["mots_cles"],
        })
    return recos


def generer_actions_systeme(data):
    df_kpis = data["kpis"]
    df_emerging = data["emerging_full"]
    actions = []

    outliers = get_kpi(df_kpis, "Outliers BERTopic", None)
    if outliers is not None and outliers > 25:
        actions.append({
            "priorite": "urgent",
            "titre": "Réduire le taux d'outliers BERTopic",
            "texte": f"Le taux d'outliers ({outliers:.1f} %) est élevé. Augmenter la taille du corpus ou réduire min_cluster_size dans HDBSCAN améliorerait la couverture thématique.",
        })

    n_periodes = get_kpi(df_kpis, "Périodes analysées", None)
    if n_periodes is not None and n_periodes < 6:
        actions.append({
            "priorite": "urgent",
            "titre": "Enrichir la couverture temporelle",
            "texte": f"Seules {int(n_periodes)} période(s) sont disponibles. Le calcul fiable des taux de croissance nécessite au moins 6 périodes.",
        })

    if len(df_emerging):
        top_e = df_emerging.sort_values("growth_pct", ascending=False).iloc[0]
        actions.append({
            "priorite": "ok",
            "titre": f"Surveiller en priorité : {top_e['label']}",
            "texte": f"Avec +{top_e['growth_pct']:.0f} % de croissance, ce topic justifie une veille rapprochée.",
        })

    return actions


# ==============================================================================
# 6. FONCTIONS D'AFFICHAGE
# ==============================================================================
def display_kpis(data, agent4_report=None):
    df_kpis = data["kpis"]
    df_emerging = data["emerging"]

    n_articles = int(get_kpi(df_kpis, "Articles analysés", len(data["articles"])))
    n_topics_bertopic = int(get_kpi(df_kpis, "Topics BERTopic", 0))
    n_emerging_total = int(get_kpi(df_kpis, "Topics émergents", len(data["emerging_full"])))
    n_emerging_affiches = len(df_emerging)

    if agent4_report and agent4_report.get("confidence"):
        score_conf = agent4_report["confidence"].get("overall_score", 0) * 100
        source_conf = "Agent 4 (LLM)"
    else:
        score_conf = get_kpi(df_kpis, "Score de confiance", 0)
        source_conf = "Calcul global"

    avg_growth = df_emerging["growth_pct"].mean() if len(df_emerging) else 0
    max_growth = df_emerging["growth_pct"].max() if len(df_emerging) else 0

    cartes = [
        ("ARTICLES ANALYSÉS", f"{n_articles:,}", "📖 Corpus complet"),
        ("TOPICS BERTOPIC", f"{n_topics_bertopic}", "🧩 Topic modeling"),
        ("TOPICS ÉMERGENTS", f"{n_emerging_affiches}/{n_emerging_total}", "📈 Affichés / détectés"),
        ("CROISSANCE MOYENNE", f"+{avg_growth:.0f}%", "📊 Sur la sélection"),
        ("CROISSANCE MAX", f"+{max_growth:.0f}%", "🚀 Topic le plus dynamique"),
        ("SCORE DE CONFIANCE", f"{score_conf:.1f}%", f"🎯 {source_conf}"),
    ]

    cols = st.columns(6)
    for col, (titre, val, sub) in zip(cols, cartes):
        with col:
            st.markdown(f"""
            <div class="pbi-card">
                <div class="kpi-num-title">{titre}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def display_topic_distribution(data):
    df_dist = data["topic_dist"]
    if df_dist is None or df_dist.empty:
        st.info("topic_distribution.csv non disponible.")
        return

    top = df_dist.sort_values("nb_articles", ascending=True).tail(15)
    couleurs = top["est_emergent"].map({True: "#10B981", False: "#3B82F6"})

    fig = go.Figure(go.Bar(
        x=top["nb_articles"],
        y=top["label"],
        orientation="h",
        marker_color=couleurs,
        text=top["nb_articles"],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=11),
        hovertemplate="<b>%{y}</b><br>%{x} articles<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        margin=dict(l=10, r=40, t=10, b=10),
        height=420,
        xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#94A3B8", size=10)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("🟢 Topic émergent détecté par l'Agent 3 · 🔵 Topic stable")


def display_correlation_chart(data):
    df = data["emerging"]
    if df is None or df.empty:
        st.info("Aucun topic émergent ne correspond aux filtres sélectionnés.")
        return

    fig = px.scatter(
        df,
        x="growth_pct",
        y="volume",
        size="volume" if df["volume"].sum() > 0 else None,
        text="label",
        color="categorie",
        hover_name="label",
        hover_data={"growth_pct": ":.0f", "volume": True, "mots_cles": True},
        size_max=40,
        labels={"growth_pct": "Croissance (%)", "volume": "Volume d'articles", "categorie": "Domaine"},
    )

    moyenne = data["avg_growth"]
    fig.add_vline(x=moyenne, line_dash="dash", line_color="#94A3B8", annotation_text=f"Moyenne (+{moyenne:.0f}%)")

    fig.update_traces(textposition="top center", textfont=dict(color="#E2E8F0", size=10))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1F2937"),
        yaxis=dict(showgrid=True, gridcolor="#1F2937"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def display_emerging_table(data):
    df = data["emerging"]
    if df is None or df.empty:
        st.info("Aucun topic à afficher.")
        return

    df_table = df[["topic_id", "label", "growth_pct", "volume", "categorie", "mots_cles"]].copy()
    df_table = df_table.sort_values("growth_pct", ascending=False)
    df_table.columns = ["ID", "Topic", "Croiss. %", "Volume", "Domaine", "Mots-clés"]

    st.dataframe(df_table, use_container_width=True, height=300, hide_index=True)


def display_timeseries_chart(data):
    df_ts = data["timeseries"]
    df_emerging = data["emerging"]

    if df_ts is None or df_ts.empty:
        st.info("timeseries.csv non disponible.")
        return
    if df_emerging.empty:
        st.info("Aucun topic sélectionné.")
        return

    topic_ids = df_emerging["topic_id"].tolist()
    df_ts_f = df_ts[df_ts["topic_id"].isin(topic_ids)].copy()
    if df_ts_f.empty:
        st.info("Aucune donnée temporelle.")
        return

    label_map = dict(zip(df_emerging["topic_id"], df_emerging["label"]))
    df_ts_f["label"] = df_ts_f["topic_id"].map(label_map)

    fig = px.line(df_ts_f, x="date", y="proportion_pct", color="label", markers=True)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0", height=380,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def display_rule_insights(data):
    insights = generer_insights_regles(data)
    if not insights:
        st.info("Pas assez de données pour générer des observations.")
        return

    for ins in insights:
        couleur = COULEUR_INSIGHT.get(ins["type"], "#10B981")
        st.markdown(f"""
        <div class="insight-card" style="border-left-color: {couleur}">
            <span class="insight-title" style="color: {couleur}">{ins['titre']}</span>
            {ins['texte']}
        </div>
        """, unsafe_allow_html=True)


def display_agent4_report(agent4_data):
    if agent4_data is None:
        st.info("ℹ️ Aucun rapport Agent 4 trouvé.")
        return

    confidence = agent4_data.get("confidence", {})
    score = confidence.get("overall_score", 0)
    level = confidence.get("confidence_level", "Inconnu")
    interpretation = confidence.get("interpretation", "")

    st.markdown(f"""
    <div class="pbi-card">
        <div class="kpi-main-title">🎯 Score de confiance global (Agent 4)</div>
        <div class="kpi-val">{score:.1%}</div>
        <div class="kpi-sub">Niveau : {level}</div>
        <div class="kpi-sub">{interpretation}</div>
    </div>
    """, unsafe_allow_html=True)

    sections = extract_sections_from_report(agent4_data.get("report", ""))
    if sections["executive_summary"]:
        st.markdown("### 📋 Résumé exécutif")
        st.markdown(f'<div class="llm-box">{sections["executive_summary"]}</div>', unsafe_allow_html=True)


def display_rule_recommendations(data):
    recos = generer_recommandations_regles(data)
    if not recos:
        st.info("Aucun topic émergent à classer.")
        return

    for r in recos[:10]:
        couleur = COULEUR_PRIO[r["priorite"]]
        label_badge = LABEL_PRIO[r["priorite"]]
        st.markdown(f"""
        <div class="reco-card" style="border-left-color: {couleur}">
            <span class="reco-title" style="color: {couleur}">{r['label']}</span>
            <span class="badge" style="background:{couleur}22; color:{couleur};">{label_badge}</span><br>
            {r['categorie']} · +{r['growth_pct']:.0f}% · {r['volume']} articles<br>
            <span style="color:#94A3B8;">{r['mots_cles']}</span>
        </div>
        """, unsafe_allow_html=True)


def display_agent4_decision(agent4_data):
    if agent4_data is None:
        st.info("ℹ️ Aucun rapport Agent 4 trouvé.")
        return

    confidence = agent4_data.get("confidence", {})
    score = confidence.get("overall_score", 0)
    level = confidence.get("confidence_level", "Inconnu")

    if score >= 0.7:
        st.success(f"✅ **Niveau de confiance : {level}**\n\n{confidence.get('interpretation', '')}")
    elif score >= 0.5:
        st.warning(f"⚠️ **Niveau de confiance : {level}**")
    else:
        st.error(f"❌ **Niveau de confiance : {level}**")


# ==============================================================================
# 7. SIDEBAR
# ==============================================================================
def display_sidebar(data):
    with st.sidebar:
        st.markdown("<h2 style='color:#F3F4F6; font-size:1.4rem;'>📊 FILTRES</h2>", unsafe_allow_html=True)
        st.write("---")

        df_emerging = data["emerging_full"]
        domaines = sorted(df_emerging["categorie"].unique().tolist())
        selected_domaines = st.multiselect("Domaines de recherche", domaines, default=domaines)

        max_growth = float(data["max_growth"]) if data["max_growth"] > 0 else 100.0
        min_growth = st.slider("Croissance minimale (%)", min_value=0, max_value=int(max_growth) + 1, value=0, step=10)

        return selected_domaines, min_growth


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    data = load_all_data()
    if data is None:
        st.stop()

    n_articles = int(get_kpi(data["kpis"], "Articles analysés", 0))
    n_categories = int(get_kpi(data["kpis"], "Catégories arXiv", 0))
    duree = int(get_kpi(data["kpis"], "Durée collecte", 0))

    st.markdown(
        f"<h1 style='text-align: center; color: #F3F4F6; font-size: 1.8rem;'>📡 VEILLE TECHNOLOGIQUE IA</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; color: #94A3B8;'>Pipeline multi-agents · {n_articles:,} articles · {n_categories} catégories · {duree} jours</p>",
        unsafe_allow_html=True
    )

    agent4_report = load_agent4_report()
    selected_domaines, min_growth = display_sidebar(data)

    df_filtered = data["emerging_full"].copy()
    if selected_domaines:
        df_filtered = df_filtered[df_filtered["categorie"].isin(selected_domaines)]
    df_filtered = df_filtered[df_filtered["growth_pct"] >= min_growth]
    data["emerging"] = df_filtered

    display_kpis(data, agent4_report)
    st.markdown("---")

    tab_viz, tab_interp, tab_decision = st.tabs(["📊 VISUALISATION", "🤖 INTERPRÉTATION AUTOMATIQUE", "💡 AIDE À LA DÉCISION"])

    with tab_viz:
        with st.container():
            st.markdown('<div class="pbi-card">', unsafe_allow_html=True)
            st.markdown('<p class="graph-title">🧩 Distribution thématique</p>')
            display_topic_distribution(data)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="pbi-card">', unsafe_allow_html=True)
                st.markdown('<p class="graph-title">🎯 Croissance vs volume</p>')
                display_correlation_chart(data)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="pbi-card">', unsafe_allow_html=True)
                st.markdown('<p class="graph-title">📋 Topics émergents</p>')
                display_emerging_table(data)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab_interp:
        st.markdown("### 🧠 Analyse instantanée (règles)")
        display_rule_insights(data)
        st.markdown("---")
        st.markdown("### 🤖 Analyse approfondie — Agent 4")
        display_agent4_report(agent4_report)

    with tab_decision:
        st.markdown("### 🎯 Priorisation des topics")
        display_rule_recommendations(data)
        st.markdown("---")
        st.markdown("### 🤖 Recommandations Agent 4")
        display_agent4_decision(agent4_report)

    st.markdown("---")
    st.markdown("<center style='color:#64748B;'>Dashboard multi-agents | Agent 1 → Agent 2 → Agent 3 → Agent 4</center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()