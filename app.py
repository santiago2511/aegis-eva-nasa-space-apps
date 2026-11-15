#!/usr/bin/env python3
"""
Aegis-EVA: Astronaut Health & Predictive Telemetry System
Equipe: Galactic Engineering BR-US | NASA Space Apps Challenge
Dashboard Executivo em Streamlit + Plotly: Centro de Controle de Missão e Telemetria Preditiva EVA
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Inclusão do diretório do projeto no sys.path para importações modulares
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.nasa_api import NasaDonkiClient, quick_risk_assessment
from src.biometrics_engine import BiometricsEngine, NASA_THRESHOLDS
from src.anomaly_detector import AnomalyDetector

# Configuração da Página
st.set_page_config(
    page_title="Aegis-EVA | Mission Control Telemetry",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização Customizada Dark Aeroespacial NASA
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #070b19 0%, #03060f 90%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #8da2c0;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-card {
        background: rgba(14, 23, 42, 0.75);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .alert-critical {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25) 0%, rgba(153, 27, 27, 0.1) 100%);
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .alert-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.25) 0%, rgba(180, 83, 9, 0.1) 100%);
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .alert-info {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-left: 5px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def load_nasa_weather():
    """Consulta em cache da API NASA DONKI para alta velocidade e sem gargalos de rate limit."""
    return quick_risk_assessment()


def main():
    # Inicialização de Motores
    biometrics_engine = BiometricsEngine()
    detector = AnomalyDetector(rolling_window=10, z_threshold=2.5)
    nasa_weather = load_nasa_weather()

    # SIDEBAR: Controle Operacional de Missão
    with st.sidebar:
        st.markdown("<h2 style='font-family: Orbitron; color: #00d2ff;'>🛰️ AEGIS-EVA</h2>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-top:-10px;'>GALACTIC ENGINEERING BR-US</div>", unsafe_allow_html=True)
        st.divider()

        st.subheader("👨‍🚀 Seleção de Tripulante")
        crew_list = biometrics_engine.get_crew_list()
        crew_options = {c["id"]: f"{c['name']} ({c['role'].split('/')[0].strip()})" for c in crew_list}
        selected_astro_id = st.selectbox(
            "Astronauta em Foco:",
            options=list(crew_options.keys()),
            format_func=lambda x: crew_options[x],
        )
        selected_astro = biometrics_engine.get_astronaut_by_id(selected_astro_id)

        st.divider()
        st.subheader("🧪 Injetor de Cenários / Demonstração")
        scenario_map = {
            "NOMINAL": "🟢 Nominal / EVA Estável",
            "HYPOXIA": "🚨 Hipóxia Fisiológica (Queda SpO2)",
            "HYPERCAPNIA": "⚠️ Hipercapnia Crítica (Acúmulo CO2)",
            "DEPRESSURIZATION": "💥 Despressurização Aguda do Traje",
            "THERMAL": "🔥 Estresse Térmico / Falha LCVG",
            "RADIATION": "☀️ Surto de Radiação Solar (SPE)",
        }
        selected_scenario = st.selectbox(
            "Simular Condição Médica/Traje:",
            options=list(scenario_map.keys()),
            format_func=lambda x: scenario_map[x],
        )

        st.subheader("⏱️ Janela Temporal da Telemetria")
        time_window = st.slider("Histórico recente (minutos):", min_value=15, max_value=90, value=45, step=15)

        st.divider()
        st.subheader("📡 NASA DONKI Status")
        risk_color = {
            "NOMINAL": "#10b981",
            "MODERATE": "#f59e0b",
            "HIGH": "#f97316",
            "CRITICAL": "#ef4444",
        }.get(nasa_weather.get("risk_level", "NOMINAL"), "#00d2ff")

        st.markdown(
            f"""
            <div style='background: rgba(15, 23, 42, 0.8); border: 1px solid {risk_color}; border-radius: 8px; padding: 10px;'>
                <div style='font-size: 0.8rem; color: #94a3b8;'>CLIMA ESPACIAL:</div>
                <div style='font-size: 1.1rem; font-weight: bold; color: {risk_color};'>{nasa_weather.get('risk_level')}</div>
                <div style='font-size: 0.8rem;'>Solar Flare: <b>{nasa_weather.get('max_flare_class')}</b> | Kp: <b>{nasa_weather.get('max_kp_index')}</b></div>
                <div style='font-size: 0.8rem;'>Multiplicador Rad: <b>{nasa_weather.get('radiation_multiplier')}x</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CABEÇALHO PRINCIPAL
    st.markdown("<div class='main-header'>AEGIS-EVA // ASTRONAUT TELEMETRY & HEALTH</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Centro de Comando Aeroespacial • Monitoramento Biomédico e Diagnóstico Preditivo em Tempo Real</div>", unsafe_allow_html=True)

    # GERAÇÃO E PROCESSAMENTO DA TELEMETRIA
    rad_mult = nasa_weather.get("radiation_multiplier", 1.0)
    if selected_scenario == "RADIATION":
        rad_mult = max(rad_mult, 8.5)

    df_telemetry = biometrics_engine.generate_historical_dataframe(
        astronaut_id=selected_astro_id,
        minutes=time_window,
        freq_seconds=30,
        scenario=selected_scenario,
        radiation_multiplier=rad_mult,
    )

    df_analyzed, alerts, ahi_score = detector.evaluate_telemetry_series(df_telemetry, nasa_weather)
    latest_pt = df_analyzed.iloc[-1]
    prev_pt = df_analyzed.iloc[-2] if len(df_analyzed) > 1 else latest_pt

    # PAINEL EXECUTIVO DE KPIS VITAIS
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Cores dinâmicas para AHI
    ahi_color = "#10b981" if ahi_score >= 80 else ("#f59e0b" if ahi_score >= 60 else "#ef4444")

    with col1:
        st.metric(
            label="❤️ Frequência Cardíaca",
            value=f"{latest_pt['heart_rate']:.0f} BPM",
            delta=f"{latest_pt['heart_rate'] - prev_pt['heart_rate']:+.1f}",
        )
    with col2:
        st.metric(
            label="🫁 Oximetria (SpO2)",
            value=f"{latest_pt['spo2']:.1f} %",
            delta=f"{latest_pt['spo2'] - prev_pt['spo2']:+.1f} %",
        )
    with col3:
        st.metric(
            label="🫧 Pressão do Traje",
            value=f"{latest_pt['suit_pressure']:.2f} kPa",
            delta=f"{latest_pt['suit_pressure'] - prev_pt['suit_pressure']:+.2f}",
        )
    with col4:
        st.metric(
            label="🌫️ CO2 no Traje",
            value=f"{latest_pt['co2_level']:.2f} mmHg",
            delta=f"{latest_pt['co2_level'] - prev_pt['co2_level']:+.2f}",
        )
    with col5:
        st.metric(
            label="☢️ Taxa Radiação",
            value=f"{latest_pt['radiation_rate']:.3f} mSv/h",
            delta=f"{latest_pt['radiation_rate'] - prev_pt['radiation_rate']:+.3f}",
        )
    with col6:
        st.metric(
            label="🛡️ Aegis Health (AHI)",
            value=f"{ahi_score:.1f} / 100",
            delta=f"{ahi_score - 100.0:+.1f}",
        )

    st.markdown("---")

    # SEÇÃO DE ALERTAS CRÍTICOS E DIAGNÓSTICO PREDITIVO
    st.subheader("🚨 Painel de Diagnóstico Preditivo e Alertas Operacionais (MCC)")
    
    crit_cols = st.columns(len(alerts) if len(alerts) <= 3 else 3)
    for idx, alert in enumerate(alerts):
        c_idx = idx % len(crit_cols)
        sev_class = "alert-critical" if alert["severity"] == "CRITICAL" else ("alert-warning" if alert["severity"] == "WARNING" else "alert-info")
        icon = "🔴" if alert["severity"] == "CRITICAL" else ("🟡" if alert["severity"] == "WARNING" else "🟢")
        ttc_text = f" • <b>TTC (Tempo até Criticalidade):</b> {alert['ttc_minutes']} min" if alert.get("ttc_minutes", 999) < 900 else ""
        
        with crit_cols[c_idx]:
            st.markdown(
                f"""
                <div class='{sev_class}'>
                    <div style='font-size: 0.85rem; font-weight: bold; text-transform: uppercase;'>{icon} {alert['system']} // [{alert['severity']}]</div>
                    <div style='font-size: 1.05rem; font-weight: 700; margin: 4px 0;'>{alert['title']}</div>
                    <div style='font-size: 0.85rem; color: #cbd5e1;'>{alert['detail']}{ttc_text}</div>
                    <div style='font-size: 0.85rem; color: #38bdf8; margin-top: 6px;'><b>Diretriz:</b> {alert['recommendation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # GRÁFICOS INTERATIVOS DE SÉRIES TEMPORAIS COM PLOTLY
    st.subheader("📈 Telemetria Multivariada com Faixas de Segurança Normativas (NASA-STD-3001)")

    tab_vitals, tab_suit, tab_radiation, tab_zscore = st.tabs([
        "🫀 Fisiologia & Oximetria",
        "👨‍🚀 Atmosfera e Pressão do Traje",
        "☀️ Radiação & Clima Espacial",
        "📊 Análise Matemática (Z-Score & Derivadas)",
    ])

    with tab_vitals:
        fig1 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Frequência Cardíaca (BPM) com Limites de Esforço", "Saturação de Oxigênio (SpO2 %) com Zona Crítica de Hipóxia"),
        )
        # Frequência Cardíaca
        fig1.add_trace(
            go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["heart_rate"], mode="lines+markers", name="Frequência Cardíaca (BPM)", line=dict(color="#00d2ff", width=2.5)),
            row=1, col=1,
        )
        # Limiares de FC
        fig1.add_hline(y=100, line_dash="dot", line_color="#f59e0b", annotation_text="Limiar Alerta (100 BPM)", row=1, col=1)
        fig1.add_hline(y=130, line_dash="dash", line_color="#ef4444", annotation_text="Limiar Crítico (130 BPM)", row=1, col=1)

        # SpO2
        fig1.add_trace(
            go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["spo2"], mode="lines+markers", name="SpO2 (%)", line=dict(color="#10b981", width=2.5)),
            row=2, col=1,
        )
        fig1.add_hline(y=95.0, line_dash="dot", line_color="#f59e0b", annotation_text="Atenção SpO2 (95%)", row=2, col=1)
        fig1.add_hline(y=90.0, line_dash="dash", line_color="#ef4444", annotation_text="Hipóxia Grave (90%)", row=2, col=1)

        fig1.update_layout(height=450, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

    with tab_suit:
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Pressão Interna do Traje EMU (kPa)", "Nível de Dióxido de Carbono - CO2 (mmHg)"),
        )
        # Pressão
        fig2.add_trace(
            go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["suit_pressure"], mode="lines", name="Pressão Traje (kPa)", line=dict(color="#38bdf8", width=2.5)),
            row=1, col=1,
        )
        fig2.add_hline(y=26.5, line_dash="dash", line_color="#ef4444", annotation_text="Perigo Despressurização (26.5 kPa)", row=1, col=1)

        # CO2
        fig2.add_trace(
            go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["co2_level"], mode="lines", name="CO2 Traje (mmHg)", line=dict(color="#fbbf24", width=2.5)),
            row=2, col=1,
        )
        fig2.add_hline(y=5.5, line_dash="dot", line_color="#f59e0b", annotation_text="Aviso LiOH (5.5 mmHg)", row=2, col=1)
        fig2.add_hline(y=8.0, line_dash="dash", line_color="#ef4444", annotation_text="Hipercapnia Severa (8.0 mmHg)", row=2, col=1)

        fig2.update_layout(height=450, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_radiation:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=df_analyzed["datetime"],
                y=df_analyzed["radiation_rate"],
                mode="lines+markers",
                name="Taxa de Radiação (mSv/h)",
                line=dict(color="#ec4899", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(236, 72, 153, 0.15)",
            )
        )
        fig3.add_hline(y=0.25, line_dash="dot", line_color="#f59e0b", annotation_text="Limiar Alerta Dosimétrico (0.25 mSv/h)")
        fig3.add_hline(y=0.80, line_dash="dash", line_color="#ef4444", annotation_text="Evacuação Imediata EVA (0.80 mSv/h)")
        fig3.update_layout(
            title=f"Perfil de Exposição à Radiação em EVA (Fator Solar NASA: {nasa_weather.get('risk_level')})",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab_zscore:
        fig4 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Z-Score Móvel (Detecção de Desvio Padrão)", "Derivadas de Primeira Ordem (dY/dt - Tendência de Variação)"),
        )
        # Z-Scores
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["heart_rate_zscore"], name="Z-Score FC", line=dict(color="#00d2ff")), row=1, col=1)
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["co2_level_zscore"], name="Z-Score CO2", line=dict(color="#fbbf24")), row=1, col=1)
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["suit_pressure_zscore"], name="Z-Score Pressão", line=dict(color="#ef4444")), row=1, col=1)
        fig4.add_hline(y=2.5, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig4.add_hline(y=-2.5, line_dash="dash", line_color="#ef4444", row=1, col=1)

        # Derivadas
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["spo2_d_dt"], name="d(SpO2)/dt (%/min)", line=dict(color="#10b981")), row=2, col=1)
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["co2_level_d_dt"], name="d(CO2)/dt (mmHg/min)", line=dict(color="#f59e0b")), row=2, col=1)
        fig4.add_trace(go.Scatter(x=df_analyzed["datetime"], y=df_analyzed["suit_pressure_d_dt"], name="d(P)/dt (kPa/min)", line=dict(color="#ec4899")), row=2, col=1)

        fig4.update_layout(height=480, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig4, use_container_width=True)

    # VISÃO GERAL DA TRIPULAÇÃO E LOG EXPORTÁVEL
    st.subheader("👥 Telemetria Consolidada da Tripulação e Exportação")
    
    col_crew, col_export = st.columns([3, 1])

    with col_crew:
        crew_snapshot = biometrics_engine.get_crew_instant_telemetry(radiation_multiplier=rad_mult)
        df_crew = pd.DataFrame(crew_snapshot)[[
            "astronaut_id", "astronaut_name", "role", "heart_rate", "spo2", "suit_pressure", "co2_level", "radiation_rate", "battery_pct"
        ]]
        df_crew.columns = ["ID", "Astronauta", "Função", "FC (BPM)", "SpO2 (%)", "Pressão (kPa)", "CO2 (mmHg)", "Radiação (mSv/h)", "Bateria (%)"]
        st.dataframe(df_crew, use_container_width=True, hide_index=True)

    with col_export:
        st.markdown("<b>Exportar Registro Técnico:</b>", unsafe_allow_html=True)
        csv_data = df_analyzed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar Telemetria em CSV",
            data=csv_data,
            file_name=f"aegis_eva_telemetry_{selected_astro_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info("Log auditável para Flight Surgeons e Engenharia de Suporte da NASA.")


if __name__ == "__main__":
    main()
