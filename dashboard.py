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
# 1. CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Veille Technologique IA - arXiv",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path fill='%2310B981' d='M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3zm8 0A1.5 1.5 0 0 1 10.5 1h3A1.5 1.5 0 0 1 15 2.5v3A1.5 1.5 0 0 1 13.5 7h-3A1.5 1.5 0 0 1 9 5.5v-3zm-8 8A1.5 1.5 0 0 1 2.5 9h3A1.5 1.5 0 0 1 7 10.5v3A1.5 1.5 0 0 1 5.5 15h-3A1.5 1.5 0 0 1 1 13.5v-3zm8 0A1.5 1.5 0 0 1 10.5 9h3A1.5 1.5 0 0 1 15 10.5v3A1.5 1.5 0 0 1 13.5 15h-3A1.5 1.5 0 0 1 9 13.5v-3z'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles personnalises + Bootstrap Icons
st.markdown("""
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
* { font-family: 'Inter', -apple-system, sans-serif; }

/* Theme sombre professionnel */
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

/* Cartes KPI */
.kpi-card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-top: 3px solid #10B981;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.15s;
}

.kpi-card:hover {
    border-color: #10B981;
    transform: translateY(-2px);
}

.kpi-label {
    font-size: 0.7rem;
    color: #94A3B8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.kpi-label i { color: #10B981; font-size: 0.85rem; }

.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #F3F4F6;
    line-height: 1.3;
    margin: 6px 0 2px 0;
}

.kpi-sub {
    font-size: 0.76rem;
    color: #64748B;
}

/* Sections */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F3F4F6;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #1F2937;
    display: flex;
    align-items: center;
    gap: 9px;
}
.section-title i { color: #10B981; font-size: 1.05rem; }
.section-title .count-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    height: 22px;
    padding: 0 7px;
    border-radius: 999px;
    background: #1F2937;
    color: #E2E8F0;
    font-size: 0.7rem;
    font-weight: 800;
    margin-left: 4px;
}

/* Insights */
.insight-card {
    background-color: #111827;
    border-left: 4px solid #10B981;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    color: #E2E8F0;
    font-size: 0.9rem;
    line-height: 1.5;
}
.insight-card .insight-title {
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* Recommandations */
.reco-card {
    background-color: #111827;
    border-left: 4px solid #F59E0B;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    color: #E2E8F0;
    font-size: 0.9rem;
    line-height: 1.5;
}
.reco-card .reco-title {
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* Badges de priorite */
.badge-priority {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 12px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-left: 8px;
}

/* Box LLM */
.llm-box {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 20px;
    color: #E2E8F0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Progress bar */
.progress-bar-bg {
    background: #1F2937;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: #10B981;
}

/* Timeline cards */
.timeline-card {
    background: #111827;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
}
.timeline-label { font-size: 0.8rem; color: #94A3B8; }
.timeline-growth { font-size: 1.2rem; font-weight: 700; color: #10B981; }
.timeline-vol { font-size: 0.7rem; color: #64748B; }

/* Table detaillee */
.detail-table-wrap {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 4px 14px 14px 14px;
    overflow-x: auto;
}
table.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
table.detail-table thead th {
    text-align: left;
    color: #94A3B8;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    font-weight: 700;
    padding: 10px 8px;
    border-bottom: 1px solid #1F2937;
}
table.detail-table tbody td {
    padding: 10px 8px;
    border-bottom: 1px solid #1A2233;
    color: #E2E8F0;
    vertical-align: middle;
}
table.detail-table tbody tr:hover { background-color: #15203299; }
.dt-topic-name { font-weight: 700; color: #F3F4F6; }
.dt-keywords { color: #94A3B8; font-size: 0.78rem; }

.growth-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.78rem;
    white-space: nowrap;
}
.trend-pill {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.76rem;
    white-space: nowrap;
}
.conf-cell {
    display: inline-block;
    min-width: 38px;
    text-align: center;
    padding: 4px 0;
    border-radius: 6px;
    font-weight: 800;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CHEMINS & CONSTANTES
# ==============================================================================
DATA_DIR = Path("data/expor")
REPORTS_DIR = Path("data/reports")

COULEUR_PRIO = {
    "urgent": "#EF4444",
    "normal": "#F59E0B",
    "ok": "#10B981"
}
ICON_PRIO = {
    "urgent": "bi-exclamation-octagon-fill",
    "normal": "bi-exclamation-triangle-fill",
    "ok": "bi-check-circle-fill",
}
LABEL_PRIO = {
    "urgent": "Priorite haute",
    "normal": "Priorite moyenne",
    "ok": "A surveiller"
}

# ==============================================================================
# 3. FONCTIONS UTILITAIRES
# ==============================================================================
def bi(name: str, cls: str = "") -> str:
    if not name.startswith("bi-"):
        name = f"bi-{name}"
    classes = f"bi {name} {cls}".strip()
    return f'<i class="{classes}"></i>'


def section_title(icon_class: str, text: str, count: int = None):
    badge = f'<span class="count-badge">{count}</span>' if count is not None else ""
    st.markdown(
        f'<div class="section-title">{bi(icon_class)}{text}{badge}</div>',
        unsafe_allow_html=True,
    )


def get_kpi(df_kpis, nom, defaut=0.0):
    if df_kpis is None:
        return defaut
    ligne = df_kpis[df_kpis["indicateur"] == nom]
    if ligne.empty:
        return defaut
    return ligne.iloc[0]["valeur"]


def growth_pill_style(pct: float) -> tuple:
    if pct >= 300:
        return "#DC2626", "#FFFFFF"
    elif pct >= 200:
        return "#F97316", "#FFFFFF"
    elif pct >= 100:
        return "#FBBF24", "#1F2937"
    elif pct >= 50:
        return "#FDE68A", "#1F2937"
    return "#334155", "#E2E8F0"


def growth_pill_html(pct: float) -> str:
    bg, fg = growth_pill_style(pct)
    return f'<span class="growth-pill" style="background:{bg}; color:{fg};">+{pct:.0f}%</span>'


def trend_from_growth(pct: float) -> tuple:
    if pct >= 300:
        return "Explosive", "#DC2626", "#FEE2E2"
    elif pct >= 200:
        return "Forte", "#EA580C", "#FFEDD5"
    elif pct >= 100:
        return "Moderee", "#2563EB", "#DBEAFE"
    elif pct >= 40:
        return "Croissante", "#059669", "#D1FAE5"
    return "Stable", "#64748B", "#E2E8F0"


def trend_pill_html(pct: float) -> str:
    label, color, bg = trend_from_growth(pct)
    return f'<span class="trend-pill" style="background:{bg}; color:{color};">{label}</span>'


def interpolate_color(value, vmin, vmax, c_low=(254, 240, 138), c_high=(220, 38, 38)):
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(c_low[0] + t * (c_high[0] - c_low[0]))
    g = int(c_low[1] + t * (c_high[1] - c_low[1]))
    b = int(c_low[2] + t * (c_high[2] - c_low[2]))
    return f"rgb({r},{g},{b})"


def confidence_heuristic(volume: float, growth_pct: float, vol_max: float) -> float:
    vol_score = min(volume / vol_max, 1.0) if vol_max > 0 else 0.0
    stab_score = 1.0 / (1.0 + max(growth_pct - 100, 0) / 300)
    score = 55 + 30 * vol_score + 15 * stab_score
    return max(0, min(100, score))


def confidence_bounds(scores: list) -> tuple:
    if not scores:
        return 60, 95
    lo, hi = min(scores), max(scores)
    if hi - lo < 5:
        lo, hi = lo - 2.5, hi + 2.5
    return lo, hi


def confidence_cell_html(score: float, vmin: float, vmax: float) -> str:
    bg = interpolate_color(score, vmin, vmax)
    fg = "#1F2937" if score < (vmin + vmax) / 2 else "#FFFFFF"
    return f'<span class="conf-cell" style="background:{bg}; color:{fg};">{score:.0f}</span>'


# ==============================================================================
# 4. CHARGEMENT DES DONNEES
# ==============================================================================
@st.cache_data(ttl=3600)
def load_all_data():
    try:
        df_emerging = pd.read_csv(DATA_DIR / "emerging_topics.csv")
        df_articles = pd.read_csv(DATA_DIR / "articles.csv")
        df_articles_topics = pd.read_csv(DATA_DIR / "articles_topics.csv")
        df_kpis = pd.read_csv(DATA_DIR / "kpis.csv")
        df_topic_dist = pd.read_csv(DATA_DIR / "topic_distribution.csv")
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e.filename}\n\nLancez `python export.py`.")
        return None

    try:
        df_ts = pd.read_csv(DATA_DIR / "timeseries.csv")
        df_ts["date"] = pd.to_datetime(df_ts["date"])
    except FileNotFoundError:
        df_ts = None

    df_articles_topics["topic_id"] = df_articles_topics["topic_id"].astype(int)
    df_emerging["topic_id"] = df_emerging["topic_id"].astype(int)

    if "volume" in df_emerging.columns:
        df_emerging["volume"] = df_emerging["volume"].fillna(0).astype(int)
    else:
        vols = df_articles_topics["topic_id"].value_counts().reset_index()
        vols.columns = ["topic_id", "volume"]
        df_emerging = df_emerging.merge(vols, on="topic_id", how="left")
        df_emerging["volume"] = df_emerging["volume"].fillna(0).astype(int)

    df_emerging["label_display"] = df_emerging["label"]

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
    if not REPORTS_DIR.exists():
        return None

    metrics_files = [f for f in REPORTS_DIR.iterdir()
                      if f.is_file() and f.name.startswith("metrics_") and f.name.endswith(".json")]
    if not metrics_files:
        return None

    latest = max(metrics_files, key=lambda x: x.stat().st_mtime)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        return None

    report_filename = metrics.get("report_file", "")
    generated_at = metrics.get("generated_at", "")

    if not report_filename:
        rpts = [f for f in REPORTS_DIR.iterdir()
                if f.is_file() and f.name.startswith("rapport_") and f.name.endswith(".md")]
        if not rpts:
            return None
        report_file = max(rpts, key=lambda x: x.stat().st_mtime)
    else:
        report_file = REPORTS_DIR / report_filename
        if not report_file.exists():
            return None

    try:
        content = report_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            report_content = parts[2].strip() if len(parts) >= 3 else content
        else:
            report_content = content
    except Exception:
        return None

    return {
        "confidence": metrics.get("confidence", {}),
        "report": report_content,
        "generated_at": generated_at,
        "report_file": str(report_file),
    }


def extract_sections(report_text: str) -> dict:
    """
    Extrait les sections principales du rapport.
    """
    sections = {"executive_summary": "", "recommendations": "", "limitations": ""}
    
    if not report_text:
        return sections
    
    # Chercher le resume executif
    resume_match = re.search(
        r'(?:#+\s*Resume Executif|#+\s*Résumé Exécutif)(.*?)(?=#+\s*1\.\s*Introduction|#+\s*1\.\s*Introduction|##\s*1\.\s*Introduction)',
        report_text,
        re.DOTALL | re.IGNORECASE
    )
    if resume_match:
        sections["executive_summary"] = resume_match.group(1).strip()
    
    # Chercher les recommandations
    rec_match = re.search(
        r'(?:#+\s*Recommandations|#+\s*5\.\s*Recommandations|##\s*5\.\s*Recommandations)(.*?)(?=#+\s*6\.\s*Conclusion|#+\s*6\.\s*Conclusion|##\s*6\.\s*Conclusion|$|#+\s*References|#+\s*Annexes)',
        report_text,
        re.DOTALL | re.IGNORECASE
    )
    if rec_match:
        sections["recommendations"] = rec_match.group(1).strip()
    else:
        rec_match2 = re.search(
            r'(?:5\.\s*Recommandations|5\.\s*RECOMMANDATIONS)(.*?)(?=6\.|$|#+\s*6\.)',
            report_text,
            re.DOTALL | re.IGNORECASE
        )
        if rec_match2:
            sections["recommendations"] = rec_match2.group(1).strip()
    
    # Chercher les limitations
    lim_match = re.search(
        r'(?:#+\s*Limitations|#+\s*6\.\s*Limitations|##\s*6\.\s*Limitations)(.*?)(?=#+\s*|$)',
        report_text,
        re.DOTALL | re.IGNORECASE
    )
    if lim_match:
        sections["limitations"] = lim_match.group(1).strip()
    
    return sections


# ==============================================================================
# 5. AFFICHAGE DES KPIS
# ==============================================================================
def display_kpis(data: dict, agent4_report=None):
    df_kpis = data["kpis"]
    df_emerging = data["emerging"]

    n_articles = int(get_kpi(df_kpis, "Articles analyses", len(data["articles"])))
    n_topics_bert = int(get_kpi(df_kpis, "Topics BERTopic", 0))
    n_emerging = len(df_emerging)
    n_topics_total = int(get_kpi(df_kpis, "Topics emergents", len(data["emerging_full"])))

    if agent4_report and agent4_report.get("confidence"):
        score_conf = agent4_report["confidence"].get("overall_score", 0) * 100
        source_conf = "Agent 4 - LLM"
    else:
        score_conf = get_kpi(df_kpis, "Score de confiance", 0)
        source_conf = "Calcul global"

    avg_growth = df_emerging["growth_pct"].mean() if len(df_emerging) else 0
    max_growth = df_emerging["growth_pct"].max() if len(df_emerging) else 0

    kpis = [
        ("bi-file-earmark-text-fill", "Articles analyses", f"{n_articles:,}", "Corpus complet"),
        ("bi-diagram-3-fill", "Topics BERTopic", f"{n_topics_bert}", "Topic modeling"),
        ("bi-graph-up-arrow", "Topics emergents", f"{n_emerging}/{n_topics_total}", "Detectes / Total"),
        ("bi-bar-chart-fill", "Croissance moyenne", f"+{avg_growth:.0f} %", "Sur la selection"),
        ("bi-rocket-takeoff-fill", "Croissance max", f"+{max_growth:.0f} %", "Topic le plus dynamique"),
        ("bi-shield-check", "Score de confiance", f"{score_conf:.1f} %", source_conf),
    ]

    cols = st.columns(6)
    for col, (icon, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{bi(icon)}{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 6. TABLE DETAILLEE
# ==============================================================================
def display_detailed_topics_table(data: dict, max_rows: int = 15):
    df = data["emerging"]
    if df.empty:
        st.info("Aucun topic a afficher.")
        return

    df_sorted = df.sort_values("growth_pct", ascending=False).head(max_rows).copy()
    vol_max = df["volume"].max() if df["volume"].max() > 0 else 1

    conf_scores = [
        confidence_heuristic(int(r["volume"]), float(r["growth_pct"]), vol_max)
        for _, r in df_sorted.iterrows()
    ]
    conf_vmin, conf_vmax = confidence_bounds(conf_scores)

    html_parts = []
    
    html_parts.append("""
    <style>
    .detail-table-wrap {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 4px 14px 14px 14px;
        overflow-x: auto;
        margin: 10px 0;
    }
    table.detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        font-family: -apple-system, system-ui, sans-serif;
    }
    table.detail-table thead th {
        text-align: left;
        color: #94A3B8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 700;
        padding: 10px 8px;
        border-bottom: 2px solid #1F2937;
    }
    table.detail-table tbody td {
        padding: 10px 8px;
        border-bottom: 1px solid #1A2233;
        color: #E2E8F0;
        vertical-align: middle;
    }
    table.detail-table tbody tr:hover {
        background-color: #15203299;
    }
    .dt-topic-name { font-weight: 700; color: #F3F4F6; }
    .dt-keywords { color: #94A3B8; font-size: 0.78rem; }
    .text-center { text-align: center; }
    .growth-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.78rem;
        white-space: nowrap;
    }
    .trend-pill {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.76rem;
        white-space: nowrap;
    }
    .conf-cell {
        display: inline-block;
        min-width: 38px;
        text-align: center;
        padding: 4px 0;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
    }
    </style>
    """)
    
    html_parts.append('<div class="detail-table-wrap">')
    html_parts.append('<table class="detail-table">')
    html_parts.append("""
    <thead>
        <tr>
            <th>Topic</th>
            <th>Mots-cles</th>
            <th class="text-center">Articles</th>
            <th class="text-center">Croissance</th>
            <th>Tendance</th>
            <th class="text-center">Confiance*</th>
        </tr>
    </thead>
    <tbody>
    """)
    
    for (_, row), conf in zip(df_sorted.iterrows(), conf_scores):
        growth = float(row["growth_pct"])
        volume = int(row["volume"])
        
        bg, fg = growth_pill_style(growth)
        label, color, bg_trend = trend_from_growth(growth)
        conf_bg = interpolate_color(conf, conf_vmin, conf_vmax)
        conf_fg = "#1F2937" if conf < (conf_vmin + conf_vmax) / 2 else "#FFFFFF"
        
        html_parts.append(f"""
        <tr>
            <td class="dt-topic-name">{row['label_display']}</td>
            <td class="dt-keywords">{row.get('mots_cles', '')}</td>
            <td class="text-center">{volume}</td>
            <td class="text-center"><span class="growth-pill" style="background:{bg}; color:{fg};">+{growth:.0f}%</span></td>
            <td><span class="trend-pill" style="background:{bg_trend}; color:{color};">{label}</span></td>
            <td class="text-center"><span class="conf-cell" style="background:{conf_bg}; color:{conf_fg};">{conf:.0f}</span></td>
        </tr>
        """)
    
    html_parts.append("""
    </tbody>
    </table>
    </div>
    """)
    
    full_html = "".join(html_parts)
    st.components.v1.html(full_html, height=450, scrolling=True)
    
    st.caption(
        "* Indice de fiabilite calcule localement (volume + stabilite de la "
        "croissance) -- distinct du score de confiance global de l'Agent 4."
    )


def display_emerging_table_compact(data: dict, max_rows: int = 8):
    df = data["emerging"]
    if df.empty:
        st.info("Aucun topic a afficher.")
        return
    df_table = (
        df.sort_values("growth_pct", ascending=False)
        .head(max_rows)
        [["label_display", "growth_pct", "volume", "categorie"]]
        .copy()
    )
    df_table.columns = ["Topic", "Croiss. %", "Volume", "Domaine"]
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "Croiss. %": st.column_config.NumberColumn(format="+%.0f%%"),
        },
    )


def display_topic_distribution(data: dict):
    df_dist = data["topic_dist"]
    if df_dist is None or df_dist.empty:
        st.info("topic_distribution.csv non disponible.")
        return

    top = df_dist.sort_values("nb_articles", ascending=True).tail(15)
    colors = top["est_emergent"].map({True: "#10B981", False: "#3B82F6"})

    fig = go.Figure(go.Bar(
        x=top["nb_articles"],
        y=top["label"],
        orientation="h",
        marker_color=colors,
        text=top["nb_articles"],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=10),
        hovertemplate="<b>%{y}</b><br>%{x} articles<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        height=420,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#94A3B8", size=9)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Vert : topic emergent detecte par l'Agent 3 | Bleu : topic stable")


def display_correlation_chart(data: dict):
    df = data["emerging"]
    if df.empty:
        st.info("Aucun topic ne correspond aux filtres selectionnes.")
        return

    fig = px.scatter(
        df, x="growth_pct", y="volume",
        size="volume" if df["volume"].sum() > 0 else None,
        text="label_display",
        color="categorie",
        hover_name="label_display",
        hover_data={"growth_pct": ":.0f", "volume": True, "mots_cles": True},
        size_max=40,
        labels={"growth_pct": "Croissance (%)", "volume": "Volume", "categorie": "Domaine"},
    )
    moyenne = df["growth_pct"].mean()
    fig.add_vline(x=moyenne, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Moy. +{moyenne:.0f} %")
    fig.update_traces(textposition="top center", textfont=dict(color="#E2E8F0", size=9))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color="#94A3B8")),
        xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def display_timeseries_chart(data: dict):
    df_ts = data["timeseries"]
    df_emerging = data["emerging"]
    if df_ts is None or df_ts.empty or df_emerging.empty:
        return

    topic_ids = df_emerging["topic_id"].tolist()
    df_ts_f = df_ts[df_ts["topic_id"].isin(topic_ids)].copy()
    if df_ts_f.empty:
        return

    label_map = dict(zip(df_emerging["topic_id"], df_emerging["label_display"]))
    df_ts_f["label"] = df_ts_f["topic_id"].map(label_map)

    fig = px.line(
        df_ts_f, x="date", y="proportion_pct", color="label", markers=True,
        labels={"date": "Periode", "proportion_pct": "Part du corpus (%)", "label": "Topic"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        height=380,
        legend=dict(font=dict(color="#94A3B8")),
        xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def display_forecast_chart(data: dict):
    try:
        df_forecast = pd.read_csv(DATA_DIR / "forecasts.csv")
        df_emerging = data["emerging"]
        if df_forecast is None or df_forecast.empty or df_emerging.empty:
            return

        top_topics = df_emerging.head(5)["topic_id"].tolist()
        df_fc = df_forecast[df_forecast["topic_id"].isin(top_topics)].copy()
        if df_fc.empty:
            return

        label_map = dict(zip(df_emerging["topic_id"], df_emerging["label_display"]))
        df_fc["label"] = df_fc["topic_id"].map(label_map)
        df_fc["date"] = pd.to_datetime(df_fc["date"])

        fig = px.line(
            df_fc, x="date", y="forecast_pct", color="label", markers=True,
            labels={"forecast_pct": "Prevision (%)", "date": "Date", "label": "Topic"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E2E8F0",
            height=350,
            legend=dict(font=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#94A3B8")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass


def display_domain_map(data: dict):
    df = data["emerging"]
    if df.empty:
        return
    stats = (
        df.groupby("categorie")
        .agg(Articles=("volume", "sum"),
             Croissance=("growth_pct", "mean"),
             Topics=("topic_id", "count"))
        .reset_index()
        .rename(columns={"categorie": "Domaine"})
        .sort_values("Articles", ascending=False)
    )
    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Domaine": "Domaine",
            "Articles": "Articles",
            "Croissance": st.column_config.NumberColumn("Croissance moy.", format="%.1f %%"),
            "Topics": "Topics",
        },
    )


def display_search_topics(data: dict):
    df = data["emerging_full"]
    if df.empty:
        return

    section_title("bi-search", "Recherche de topics")
    term = st.text_input(
        "Rechercher par mot-cle",
        placeholder="Ex: video, quantum, agent...",
        key="search_topics",
        label_visibility="collapsed",
    )
    if term:
        results = df[
            df["label"].str.contains(term, case=False, na=False)
            | df["mots_cles"].str.contains(term, case=False, na=False)
        ]
        if not results.empty:
            st.dataframe(
                results[["label_display", "growth_pct", "volume", "categorie"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "label_display": "Topic",
                    "growth_pct": st.column_config.NumberColumn("Croissance", format="+%.0f %%"),
                    "volume": "Volume",
                    "categorie": "Domaine",
                },
            )
        else:
            st.info("Aucun topic trouve pour cette recherche.")


def display_timeline(data: dict):
    df = data["emerging"]
    if df.empty:
        return
    section_title("bi-calendar3", "Chronologie des emergences")
    st.caption("Classement par croissance relative — 5 premiers topics")
    cols = st.columns(min(len(df), 5))
    for idx, (_, row) in enumerate(df.head(5).iterrows()):
        pct_bar = min(row["growth_pct"] / 2, 100)
        with cols[idx % len(cols)]:
            st.markdown(f"""
            <div class="timeline-card">
                <div class="timeline-label">{row['label_display'][:22]}</div>
                <div class="timeline-growth">+{row['growth_pct']:.0f} %</div>
                <div class="timeline-vol">{row['volume']} articles</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width:{pct_bar}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 7. INSIGHTS ET RECOMMANDATIONS
# ==============================================================================
def generate_rule_insights(data: dict) -> list:
    insights = []
    df_emerging = data["emerging"]
    df_kpis = data["kpis"]

    if len(df_emerging):
        top = df_emerging.sort_values("growth_pct", ascending=False).iloc[0]
        insights.append({
            "type": "success",
            "icon": "bi-graph-up-arrow",
            "title": "Signal d'emergence le plus fort",
            "text": f"« {top['label_display']} » affiche la croissance la plus elevee (+{top['growth_pct']:.0f} %), avec {int(top['volume'])} articles."
        })
        moy = df_emerging["growth_pct"].mean()
        insights.append({
            "type": "info",
            "icon": "bi-activity",
            "title": "Dynamique generale",
            "text": f"Les {len(df_emerging)} topics affiches presentent une croissance moyenne de +{moy:.0f} %."
        })

    score_conf = get_kpi(df_kpis, "Score de confiance", None)
    if score_conf is not None:
        insights.append({
            "type": "success" if score_conf >= 75 else "warning" if score_conf >= 50 else "danger",
            "icon": "bi-shield-check",
            "title": "Fiabilite de l'analyse",
            "text": f"Le score de confiance global est de {score_conf:.1f} %."
        })

    return insights


def display_rule_insights(data: dict):
    insights = generate_rule_insights(data)
    if not insights:
        st.info("Donnees insuffisantes.")
        return

    type_colors = {"info": "#3B82F6", "success": "#10B981", "warning": "#F59E0B", "danger": "#EF4444"}
    for ins in insights:
        c = type_colors.get(ins["type"], "#10B981")
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{c};">
            <div class="insight-title" style="color:{c};">{bi(ins['icon'])}{ins['title']}</div>
            {ins['text']}
        </div>
        """, unsafe_allow_html=True)


def generate_rule_recommendations(data: dict) -> list:
    df_emerging = data["emerging"]
    if df_emerging.empty:
        return []

    moyenne = df_emerging["growth_pct"].mean()
    ecart = df_emerging["growth_pct"].std() if len(df_emerging) > 1 else 0

    recos = []
    for _, row in df_emerging.sort_values("growth_pct", ascending=False).iterrows():
        if ecart == 0:
            priority = "normal"
        else:
            z = (row["growth_pct"] - moyenne) / ecart
            priority = "urgent" if z > 1 else "normal" if z > 0 else "ok"

        recos.append({
            "priority": priority,
            "label": row["label_display"],
            "growth_pct": row["growth_pct"],
            "volume": int(row["volume"]),
            "mots_cles": row.get("mots_cles", ""),
        })
    return recos


def display_rule_recommendations(data: dict):
    recos = generate_rule_recommendations(data)
    if not recos:
        st.info("Aucun topic emergent.")
        return

    for r in recos:
        c = COULEUR_PRIO[r["priority"]]
        icon = ICON_PRIO[r["priority"]]
        label = LABEL_PRIO[r["priority"]]
        st.markdown(f"""
        <div class="reco-card" style="border-left-color:{c};">
            <div class="reco-title" style="color:{c};">
                {bi(icon)}{r['label']}
                <span class="badge-priority" style="background:{c}22;color:{c};border:1px solid {c};">{label}</span>
            </div>
            <div style="color:#94A3B8;font-size:0.85rem;margin-top:4px;">
                +{r['growth_pct']:.0f} % &middot; {r['volume']} articles
            </div>
            <div style="color:#64748B;font-size:0.8rem;">{r['mots_cles']}</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 8. AGENT 4 - RAPPORT (VERSION UNIFIEE - CORRIGEE)
# ==============================================================================
def display_agent4_report(agent4_data, show_full: bool = True):
    """
    Affiche le rapport Agent 4 de maniere unifiee.
    
    Args:
        agent4_data: Donnees du rapport
        show_full: Si True, affiche le rapport complet; Sinon, seulement les recommandations
    """
    if agent4_data is None:
        st.info("Aucun rapport Agent 4 trouve.")
        if st.button("Generer le rapport (Ollama)"):
            import subprocess, sys
            with st.spinner("Generation en cours..."):
                res = subprocess.run(
                    [sys.executable, "agent_4_synthese_llm/report_generator.py"],
                    capture_output=True,
                )
                if res.returncode == 0:
                    st.success("Rapport genere avec succes.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Erreur : {res.stderr.decode()}")
        return

    confidence = agent4_data.get("confidence", {})
    score = confidence.get("overall_score", 0)
    level = confidence.get("confidence_level", "Inconnu")
    interp = confidence.get("interpretation", "")
    color = "#10B981" if score >= 0.7 else "#F59E0B" if score >= 0.5 else "#EF4444"
    icon = "bi-shield-fill-check" if score >= 0.7 else "bi-shield-fill-exclamation" if score >= 0.5 else "bi-shield-fill-x"

    # Afficher le score de confiance (toujours visible)
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:{color};">
        <div class="kpi-label">{bi(icon)}Score de confiance global - Agent 4</div>
        <div class="kpi-value" style="color:{color};">{score:.1%}</div>
        <div class="kpi-sub">Niveau : {level}</div>
        <div class="kpi-sub">{interp}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Extraire les sections
    sections = extract_sections(agent4_data.get("report", ""))
    
    if show_full:
        # Mode complet : afficher tout le rapport
        report_content = agent4_data.get("report", "")
        if report_content:
            if report_content.startswith("---"):
                parts = report_content.split("---", 2)
                if len(parts) >= 3:
                    report_content = parts[2].strip()
            st.markdown(f'<div class="llm-box">{report_content}</div>', unsafe_allow_html=True)
    else:
        # Mode resume : afficher SEULEMENT les recommandations
        if sections["recommendations"]:
            rec_text = sections["recommendations"]
            st.markdown(f'<div class="llm-box" style="border-left:4px solid #F59E0B;background:#1E293B;padding:16px;border-radius:8px;">{rec_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Aucune recommandation detaillee disponible dans le rapport.")

    st.caption(f"Rapport genere le : {agent4_data.get('generated_at', 'date inconnue')}")


# ==============================================================================
# 8b. AGENT 4 - RESUME STRATEGIQUE
# ==============================================================================
def display_agent4_enhanced_report(agent4_data, data: dict):
    """
    Affiche un resume strategique base sur le rapport Agent 4
    """
    if agent4_data is None:
        return

    confidence = agent4_data.get("confidence", {})
    score = confidence.get("overall_score", 0)
    level = confidence.get("confidence_level", "N/A")
    sections = extract_sections(agent4_data.get("report", ""))
    df_emerging = data["emerging"]

    if score >= 0.7:
        st.success(f"Niveau de confiance : **{level}** ({score:.0%})")
    elif score >= 0.5:
        st.warning(f"Niveau de confiance : **{level}** ({score:.0%})")
    else:
        st.error(f"Niveau de confiance : **{level}** ({score:.0%})")

    st.markdown("---")

    if not df_emerging.empty:
        top = df_emerging.iloc[0]
        st.markdown(f"""
        <div class="llm-box">
            <div style="margin-bottom:8px;">
                {bi('bi-arrow-up-right-circle-fill')}
                <strong>Theme en tete :</strong> {top['label_display']} (+{top['growth_pct']:.0f} %)
            </div>
            <div style="margin-bottom:8px;">
                {bi('bi-file-earmark-text')}
                <strong>Volume :</strong> {top['volume']} articles
            </div>
            <div style="margin-bottom:12px;">
                {bi('bi-tags-fill')}
                <strong>Mots-cles :</strong> {top.get('mots_cles', '')}
            </div>
            <div style="border-top:1px solid #334155; padding-top:10px; color:#94A3B8;">
                {bi('bi-lightbulb')}
                Ce domaine montre une forte acceleration. Une veille hebdomadaire est recommandee.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if top["volume"] < 30:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#EF4444">
                <div class="insight-title" style="color:#EF4444;">
                    {bi('bi-exclamation-triangle-fill')} Risque d'instabilite
                </div>
                Ce topic ne compte que {top['volume']} articles. La tendance doit etre interpretee avec prudence.
            </div>
            """, unsafe_allow_html=True)

        if top["growth_pct"] > 50:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#10B981">
                <div class="insight-title" style="color:#10B981;">
                    {bi('bi-lightbulb-fill')} Opportunite identifiee
                </div>
                Croissance continue au-dessus du seuil d'emergence. Potentiel eleve pour des projets futurs.
            </div>
            """, unsafe_allow_html=True)

    
# ==============================================================================
# 9. SIDEBAR
# ==============================================================================
def display_sidebar(data: dict):
    with st.sidebar:
        st.markdown(
            f'<h2 style="color:#F3F4F6;font-size:1.2rem;">{bi("bi-sliders")}Filtres</h2>',
            unsafe_allow_html=True,
        )
        st.write("---")

        df = data["emerging_full"]
        domaines = sorted(df["categorie"].unique().tolist())
        selected = st.multiselect("Domaines de recherche", domaines, default=domaines)

        max_g = float(data["max_growth"]) if data["max_growth"] > 0 else 100.0
        min_g = st.slider("Croissance minimale (%)", 0, int(max_g) + 1, 0, step=10)

        st.write("---")
        st.caption("Ces filtres s'appliquent aux trois onglets.")
        st.write("---")
        if st.button("Actualiser les donnees"):
            st.cache_data.clear()
            st.rerun()

    return selected, min_g


# ==============================================================================
# 10. POINT D'ENTREE
# ==============================================================================
def main():
    data = load_all_data()
    if data is None:
        st.stop()


    n_articles = int(get_kpi(data["kpis"], "Articles analysés", 0))
    n_categories = int(get_kpi(data["kpis"], "Catégories arXiv", 0))
    duree = int(get_kpi(data["kpis"], "Durée collecte", 0))

    st.markdown("""
    <h1 style="text-align:center;color:#F3F4F6;font-size:1.75rem;margin-bottom:4px;font-weight:800;">
        Veille Technologique IA - arXiv
    </h1>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center;color:#94A3B8;margin-bottom:20px;font-size:0.9rem;'>"
        f"Pipeline multi-agents &nbsp;·&nbsp; {n_articles:,} articles &nbsp;·&nbsp; "
        f"{n_categories} categories arXiv &nbsp;·&nbsp; {duree} jours &nbsp;·&nbsp; "
        f"Mise a jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>",
        unsafe_allow_html=True,
    )

    agent4_report = load_agent4_report()
    selected_domaines, min_g = display_sidebar(data)

    df_filtered = data["emerging_full"].copy()
    if selected_domaines:
        df_filtered = df_filtered[df_filtered["categorie"].isin(selected_domaines)]
    df_filtered = df_filtered[df_filtered["growth_pct"] >= min_g]
    data["emerging"] = df_filtered

    display_kpis(data, agent4_report)
    st.markdown("---")

    tab_viz, tab_interp, tab_decision = st.tabs([
        "Visualisation",
        "Interpretation automatique",
        "Aide a la decision",
    ])

    # Onglet 1 - Visualisation
    with tab_viz:
        section_title("bi-bar-chart-line-fill", "Distribution thematique")
        display_topic_distribution(data)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            section_title("bi-scatter-chart", "Croissance vs Volume")
            display_correlation_chart(data)
        with col2:
            section_title("bi-table", "Topics emergents")
            display_emerging_table_compact(data, max_rows=8)

        st.markdown("---")
        section_title(
            "bi-grid-3x3-gap-fill",
            "Table detaillee des topics emergents",
            count=len(data["emerging"]),
        )
        display_detailed_topics_table(data, max_rows=15)

        st.markdown("---")
        section_title("bi-graph-up", "Evolution temporelle")
        display_timeseries_chart(data)

        st.markdown("---")
        section_title("bi-eye-fill", "Previsions - horizon 6 mois")
        display_forecast_chart(data)

        st.markdown("---")
        section_title("bi-globe-americas", "Carte des domaines IA")
        display_domain_map(data)

        st.markdown("---")
        display_timeline(data)

        st.markdown("---")
        display_search_topics(data)

    # Onglet 2 - Interpretation automatique (Rapport COMPLET + INSIGHTS)
    with tab_interp:
        st.markdown("---")
        section_title("bi-robot", "Synthese approfondie - Agent 4")
        st.caption("Rapport complet genere par le LLM local (llama3.2) a partir du meme corpus.")
        display_agent4_report(agent4_report, show_full=True)

    # Onglet 3 - Aide a la decision (Recommandations seulement)
    with tab_decision:
        section_title("bi-funnel-fill", "Priorisation par regles")
        st.caption("Classement selon l'ecart a la croissance moyenne.")
        display_rule_recommendations(data)

        st.markdown("---")
        section_title("bi-robot", "Recommandations - Agent 4")
        st.caption("Basees sur l'analyse complete et le score de confiance.")
        display_agent4_report(agent4_report, show_full=False)

        st.markdown("---")
        section_title("bi-bar-chart-steps", "Resume strategique")
        display_agent4_enhanced_report(agent4_report, data)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#475569;font-size:0.78rem;'>"
        "Agent 1 (collecte) &rarr; Agent 2 (BERTopic) &rarr; Agent 3 (predictions) &rarr; Agent 4 (synthese LLM)</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()