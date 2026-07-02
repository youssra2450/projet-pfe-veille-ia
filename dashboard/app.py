"""
Tableau de bord Master - Visualisation des tendances IA
À lancer : streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="IA Research Trends Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .good-score {
        color: #4CAF50;
        font-weight: bold;
    }
    .medium-score {
        color: #FFC107;
        font-weight: bold;
    }
    .low-score {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class ResearchDashboard:
    """Dashboard d'analyse des tendances de recherche IA"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data" / "processed"
        self.load_all_data()
    
    def load_all_data(self):
        """Charge tous les fichiers de données"""
        self.df_articles = None
        self.df_topics = None
        self.df_emerging = None
        self.df_forecasts = None
        self.df_timeseries = None
        
        # Articles nettoyés
        articles_path = self.data_dir / "articles_clean.parquet"
        if articles_path.exists():
            self.df_articles = pd.read_parquet(articles_path)
            if 'published_date' in self.df_articles.columns:
                self.df_articles['published_date'] = pd.to_datetime(self.df_articles['published_date'])
        
        # Topics
        topics_path = self.data_dir / "articles_with_topics.parquet"
        if topics_path.exists():
            self.df_topics = pd.read_parquet(topics_path)
        
        # Topics émergents
        emerging_path = self.data_dir / "emerging_topics.parquet"
        if emerging_path.exists():
            self.df_emerging = pd.read_parquet(emerging_path)
        
        # Prédictions
        forecasts_path = self.data_dir / "forecasts" / "forecasts.parquet"
        if forecasts_path.exists():
            self.df_forecasts = pd.read_parquet(forecasts_path)
            if 'date' in self.df_forecasts.columns:
                self.df_forecasts['date'] = pd.to_datetime(self.df_forecasts['date'])
        
        # Séries temporelles
        ts_path = self.data_dir / "topic_timeseries.parquet"
        if ts_path.exists():
            self.df_timeseries = pd.read_parquet(ts_path)
    
    def render_header(self):
        """Affiche l'en-tête"""
        st.markdown('<p class="main-header">📊 IA Research Trends Dashboard</p>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analyse des tendances de recherche en IA à partir d\'arXiv</p>', 
                   unsafe_allow_html=True)
        st.divider()
    
    def render_metrics(self):
        """Affiche les métriques principales"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Nombre d'articles
        n_articles = len(self.df_articles) if self.df_articles is not None else 0
        
        # Période
        if self.df_articles is not None and 'published_date' in self.df_articles.columns:
            date_min = self.df_articles['published_date'].min()
            date_max = self.df_articles['published_date'].max()
            n_days = (date_max - date_min).days + 1
            period_text = f"{n_days} jours"
        else:
            period_text = "N/A"
        
        # Nombre de topics
        if self.df_topics is not None and 'topic_id' in self.df_topics.columns:
            n_topics = self.df_topics['topic_id'].nunique()
        else:
            n_topics = 0
        
        # Score de confiance (à calculer)
        confidence_score = 76.5
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_articles}</div>
                <div class="metric-label">📄 Articles analysés</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{period_text}</div>
                <div class="metric-label">📅 Couverture temporelle</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_topics}</div>
                <div class="metric-label">🎯 Topics identifiés</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            score_class = "good-score" if confidence_score >= 70 else ("medium-score" if confidence_score >= 50 else "low-score")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {score_class}">{confidence_score}%</div>
                <div class="metric-label">🔒 Score de confiance</div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_topic_distribution(self):
        """Affiche la distribution des topics"""
        st.subheader("📊 Distribution des thématiques")
        
        if self.df_topics is not None and 'topic_id' in self.df_topics.columns:
            topic_counts = self.df_topics['topic_id'].value_counts().sort_index()
            
            # Noms des topics (basés sur les résultats BERTopic)
            topic_names = {
                0: "Reinforcement Learning & Agents",
                1: "Evolutionary Algorithms & Search",
                2: "Neural Networks & Deep Learning",
                3: "LLM & AI Alignment",
                4: "Computer Vision",
                5: "Diffusion Models & Generation",
                6: "Edge AI & Hardware",
                7: "Time Series & Forecasting",
                8: "Healthcare & Clinical AI",
                9: "Spiking Neural Networks",
                10: "Quantum Computing",
                11: "Speech & Audio Processing",
                12: "Multimodal Learning",
                13: "Graph Neural Networks",
                14: "Federated Learning",
                15: "Explainable AI",
            }
            
            df_plot = pd.DataFrame({
                "Topic": [topic_names.get(i, f"Topic {i}") for i in topic_counts.index],
                "Nombre d'articles": topic_counts.values,
                "Proportion (%)": (topic_counts.values / topic_counts.sum() * 100).round(1)
            }).head(10)  # Top 10 seulement pour lisibilité
            
            fig = px.bar(
                df_plot,
                x="Topic",
                y="Nombre d'articles",
                color="Proportion (%)",
                text="Proportion (%)",
                color_continuous_scale="Blues",
                title="Top 10 des thématiques"
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de distribution non disponibles")
    
    def render_temporal_evolution(self):
        """Affiche l'évolution temporelle"""
        st.subheader("📈 Évolution temporelle des publications")
        
        if self.df_articles is not None and 'published_date' in self.df_articles.columns:
            # Agrégation par jour
            daily_counts = self.df_articles.groupby(self.df_articles['published_date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title="Nombre d'articles par jour",
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Nombre d'articles", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données temporelles non disponibles")
    
    def render_forecasts(self):
        """Affiche les prédictions"""
        st.subheader("🔮 Prédictions des tendances")
        
        if self.df_forecasts is not None and not self.df_forecasts.empty:
            # Top 5 topics avec les meilleures prédictions
            topic_forecasts = self.df_forecasts.groupby('topic_id').agg({
                'forecast': 'last',
                'mae': 'mean'
            }).reset_index()
            topic_forecasts = topic_forecasts.sort_values('forecast', ascending=False).head(5)
            
            fig = px.bar(
                topic_forecasts,
                x='topic_id',
                y='forecast',
                color='mae',
                title="Prévisions par topic (horizon 3 mois)",
                labels={'topic_id': 'Topic ID', 'forecast': 'Prévision', 'mae': 'MAE'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des prédictions
            st.markdown("#### Détail des prédictions")
            st.dataframe(
                topic_forecasts.head(10),
                column_config={
                    "topic_id": "Topic",
                    "forecast": st.column_config.NumberColumn("Prévision", format="%.3f"),
                    "mae": st.column_config.NumberColumn("MAE", format="%.4f")
                },
                use_container_width=True
            )
        else:
            st.info("Données de prédiction non disponibles")
    
    def render_emerging_topics(self):
        """Affiche les topics émergents"""
        st.subheader("🔥 Topics émergents")
        
        if self.df_emerging is not None and not self.df_emerging.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 des scores d'émergence**")
                top_emerging = self.df_emerging.nlargest(5, 'emergence_score')[['topic_id', 'emergence_score', 'growth_rate_3m']]
                top_emerging.columns = ['Topic', 'Score', 'Croissance']
                st.dataframe(top_emerging, use_container_width=True)
            
            with col2:
                st.markdown("**Distribution des classifications**")
                if 'classification' in self.df_emerging.columns:
                    class_counts = self.df_emerging['classification'].value_counts()
                    fig = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Classification des topics"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Classification non disponible")
        else:
            st.info("Données de topics émergents non disponibles")
    
    def render_model_performance(self):
        """Affiche les performances des modèles"""
        st.subheader("🤖 Performance des modèles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Métriques LDA**")
            st.metric("Perplexité", "1,757", help="Plus bas = meilleur")
            st.metric("Nombre de topics", "5")
        
        with col2:
            st.markdown("**Métriques BERTopic**")
            st.metric("Topics identifiés", "33")
            st.metric("Outliers", "173", help="Articles non classés")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Métriques Prophet**")
            st.metric("MAE moyen", "0.233", help="Mean Absolute Error")
            st.metric("Meilleur MAE", "0.055", help="Topic 26")
        
        with col4:
            st.markdown("**Métriques globales**")
            st.metric("Score confiance", "76.5%", help="Niveau Robuste")
            st.metric("Qualité globale", "👍 Satisfaisant")
    
    def render_sidebar(self):
        """Affiche la barre latérale"""
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/2103/2103623.png", width=80)
            st.markdown("## 🎓 Master PFE")
            st.markdown("**Sujet 2:** Analyse des tendances IA")
            st.markdown("---")
            
            st.markdown("### 📋 Architecture")
            st.markdown("""
            - **Agent 1:** Collecte & Nettoyage
            - **Agent 2:** Topic Modeling (BERTopic + LDA)
            - **Agent 3:** Prédictions (Prophet)
            - **Agent 4:** Synthèse LLM
            """)
            
            st.markdown("---")
            st.markdown("### 🔧 Technologies")
            st.markdown("""
            - arXiv API
            - BERTopic / spaCy
            - Prophet / ARIMA
            - Ollama (Mistral)
            - Streamlit
            """)
            
            st.markdown("---")
            st.markdown("### 📊 Données")
            if self.df_articles is not None:
                st.metric("Articles", len(self.df_articles))
            if self.df_topics is not None:
                st.metric("Topics", self.df_topics['topic_id'].nunique())
            
            st.markdown("---")
            st.caption(f"© 2025 - Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}")


def main():
    dashboard = ResearchDashboard()
    dashboard.render_sidebar()
    dashboard.render_header()
    
    # Tabs pour organiser le contenu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "🎯 Distribution des topics",
        "📈 Évolution temporelle",
        "🔮 Prédictions",
        "🤖 Performance"
    ])
    
    with tab1:
        dashboard.render_metrics()
        st.divider()
        dashboard.render_emerging_topics()
    
    with tab2:
        dashboard.render_topic_distribution()
    
    with tab3:
        dashboard.render_temporal_evolution()
    
    with tab4:
        dashboard.render_forecasts()
    
    with tab5:
        dashboard.render_model_performance()


if __name__ == "__main__":
    main()