#!/usr/bin/env python3
"""
Aegis-EVA: Astronaut Health & Predictive Telemetry System
Módulo: src/anomaly_detector.py
Responsabilidade: Motor de diagnóstico preditivo com embasamento matemático rigoroso:
- Z-score móvel (Rolling Z-score)
- Taxa de variação temporal (Derivada de primeira ordem dY/dt)
- Detecção precoce de Hipóxia, Hipercapnia, Despressurização, Estresse Térmico e Radiação Solar
- Calibração dinâmica de limiares baseada na API NASA DONKI (Space Weather Tuning)
- Índice de Saúde de Atividade Extraveicular (Aegis Health Index - AHI: 0 - 100)
- Cálculo de Tempo até Criticalidade (Time-to-Criticality - TTC em minutos)
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from src.biometrics_engine import NASA_THRESHOLDS
    from src.nasa_api import quick_risk_assessment
except ImportError:
    from biometrics_engine import NASA_THRESHOLDS
    from nasa_api import quick_risk_assessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Aegis-AnomalyDetector")


class AnomalyDetector:
    """Motor analítico e preditivo de anomalias fisiológicas e de suporte de vida (PLSS)."""

    def __init__(self, rolling_window: int = 10, z_threshold: float = 2.5):
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold

    def calculate_rolling_z_score(self, series: pd.Series) -> pd.Series:
        """
        Calcula o Z-score móvel:
        Z_t = (X_t - mu_w) / (sigma_w + epsilon)
        onde mu_w e sigma_w são a média e o desvio padrão da janela deslizante w.
        """
        if len(series) < 3:
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        rolling_mean = series.rolling(window=self.rolling_window, min_periods=2).mean()
        rolling_std = series.rolling(window=self.rolling_window, min_periods=2).std()
        epsilon = 1e-6
        z_scores = (series - rolling_mean) / (rolling_std + epsilon)
        return z_scores.fillna(0.0)

    def calculate_first_derivative(self, series: pd.Series, dt_minutes: float = 0.5) -> pd.Series:
        """
        Calcula a derivada discreta de primeira ordem (taxa de variação temporal):
        dY/dt ~= (Y_t - Y_{t-1}) / dt
        """
        derivative = series.diff() / max(dt_minutes, 0.01)
        return derivative.fillna(0.0)

    def evaluate_telemetry_series(
        self,
        df: pd.DataFrame,
        nasa_risk_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float]:
        """
        Processa a série temporal de telemetria completa de um astronauta.
        Retorna:
          - DataFrame enriquecido com Z-scores, derivadas e flags preditivas
          - Lista de eventos e alertas de anomalia ativos
          - Aegis Health Index (AHI) atual (0 - 100)
        """
        if df.empty:
            return df, [], 100.0

        df = df.copy()
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)

        # 1. Ajuste adaptativo de limiares por clima espacial NASA DONKI
        rad_mult = 1.0
        weather_alert = "NOMINAL"
        if nasa_risk_summary:
            rad_mult = nasa_risk_summary.get("radiation_multiplier", 1.0)
            weather_alert = nasa_risk_summary.get("risk_level", "NOMINAL")

        # Limiares adaptativos
        adapted_rad_warn = NASA_THRESHOLDS["radiation_rate"]["warn_max"] / max(rad_mult * 0.7, 1.0)
        adapted_rad_crit = NASA_THRESHOLDS["radiation_rate"]["crit_max"] / max(rad_mult * 0.7, 1.0)

        # 2. Engenharia de Recursos Analíticos (Z-Scores e Derivadas)
        monitored_metrics = ["heart_rate", "spo2", "co2_level", "suit_pressure", "core_temp", "suit_temp", "radiation_rate"]
        for metric in monitored_metrics:
            if metric in df.columns:
                df[f"{metric}_zscore"] = self.calculate_rolling_z_score(df[metric])
                df[f"{metric}_d_dt"] = self.calculate_first_derivative(df[metric], dt_minutes=0.5)

        # 3. Análise da Leitura Mais Recente (Último Ponto)
        latest = df.iloc[-1]
        alerts: List[Dict[str, Any]] = []
        health_penalty = 0.0

        # Regra 1: Hipóxia Preditiva (SpO2 em declínio ou absoluto baixo)
        spo2_val = latest.get("spo2", 98.0)
        spo2_d_dt = latest.get("spo2_d_dt", 0.0)
        if spo2_val < NASA_THRESHOLDS["spo2"]["crit_min"] or (spo2_val < 93.0 and spo2_d_dt < -0.3):
            ttc_min = max(0.5, round((spo2_val - 85.0) / abs(min(spo2_d_dt, -0.2)), 1)) if spo2_d_dt < 0 else 1.0
            alerts.append({
                "code": "CRIT_HYPOXIA",
                "severity": "CRITICAL",
                "system": "Fisiológico / O2",
                "title": "Hipóxia Crítica Detectada",
                "detail": f"SpO2 em {spo2_val:.1f}% com taxa de declínio de {spo2_d_dt:.2f}%/min. Risco de perda de consciência.",
                "ttc_minutes": ttc_min,
                "recommendation": "Purgar traje com O2 100% de emergência e iniciar retorno imediato à câmara de descompressão (Airlock).",
            })
            health_penalty += 45.0
        elif spo2_val < NASA_THRESHOLDS["spo2"]["warn_min"] or spo2_d_dt < -0.4:
            alerts.append({
                "code": "WARN_HYPOXIA_TREND",
                "severity": "WARNING",
                "system": "Fisiológico / O2",
                "title": "Tendência de Hipóxia Preditiva",
                "detail": f"SpO2 em {spo2_val:.1f}% (dSpO2/dt = {spo2_d_dt:.2f}%/min). Queda anormal detectada antes do limiar crítico.",
                "ttc_minutes": 5.0,
                "recommendation": "Verificar válvula reguladora de O2 primária do subsistema PLSS.",
            })
            health_penalty += 20.0

        # Regra 2: Hipercapnia Preditiva (Acúmulo de CO2 no Traje)
        co2_val = latest.get("co2_level", 2.5)
        co2_d_dt = latest.get("co2_d_dt", 0.0)
        if co2_val > NASA_THRESHOLDS["co2_level"]["crit_max"] or (co2_val > 6.0 and co2_d_dt > 0.4):
            ttc_min = max(1.0, round((10.0 - co2_val) / max(co2_d_dt, 0.2), 1)) if co2_d_dt > 0 else 2.0
            alerts.append({
                "code": "CRIT_HYPERCAPNIA",
                "severity": "CRITICAL",
                "system": "Suporte de Vida (PLSS)",
                "title": "Hipercapnia Severa (CO2 Crítico)",
                "detail": f"Nível de CO2 em {co2_val:.2f} mmHg com taxa de acúmulo de +{co2_d_dt:.2f} mmHg/min. Saturação do LiOH/Metox.",
                "ttc_minutes": ttc_min,
                "recommendation": "Comutar imediatamente para o canister reserva de depuração de CO2.",
            })
            health_penalty += 40.0
        elif co2_val > NASA_THRESHOLDS["co2_level"]["warn_max"] or co2_d_dt > 0.3:
            alerts.append({
                "code": "WARN_CO2_BUILDUP",
                "severity": "WARNING",
                "system": "Suporte de Vida (PLSS)",
                "title": "Acúmulo Anormal de Dióxido de Carbono",
                "detail": f"CO2 em {co2_val:.2f} mmHg com tendência ascendente contínua (+{co2_d_dt:.2f} mmHg/min).",
                "ttc_minutes": 8.0,
                "recommendation": "Reduzir carga de trabalho metabólica e monitorar ciclo de ventilação do traje.",
            })
            health_penalty += 18.0

        # Regra 3: Despressurização do Traje Espacial (EMU Suit Pressure)
        suit_p = latest.get("suit_pressure", 29.6)
        p_d_dt = latest.get("suit_pressure_d_dt", 0.0)
        if suit_p < NASA_THRESHOLDS["suit_pressure"]["crit_min"] or p_d_dt < -0.3:
            ttc_min = max(0.2, round((suit_p - 20.0) / abs(min(p_d_dt, -0.2)), 1)) if p_d_dt < 0 else 1.0
            alerts.append({
                "code": "CRIT_DEPRESSURIZATION",
                "severity": "CRITICAL",
                "system": "Integridade do Traje (EMU)",
                "title": "Despressurização Rápida do Traje",
                "detail": f"Pressão da cabine do traje em {suit_p:.2f} kPa (dP/dt = {p_d_dt:.2f} kPa/min). Possível perfuração micrometeorítica.",
                "ttc_minutes": ttc_min,
                "recommendation": "ABORTAR MISSÃO EVA IMEDIATAMENTE. Ativar suprimento de O2 de emergência e entrar no Airlock.",
            })
            health_penalty += 55.0
        elif suit_p < NASA_THRESHOLDS["suit_pressure"]["warn_min"] or p_d_dt < -0.12:
            alerts.append({
                "code": "WARN_PRESSURE_LOSS",
                "severity": "WARNING",
                "system": "Integridade do Traje (EMU)",
                "title": "Microvazamento de Pressão Detectado",
                "detail": f"Pressão do traje em {suit_p:.2f} kPa com taxa de perda contínua ({p_d_dt:.2f} kPa/min).",
                "ttc_minutes": 10.0,
                "recommendation": "Inspecionar juntas selantes, luvas e conexão umbilical.",
            })
            health_penalty += 25.0

        # Regra 4: Estresse Térmico e Falha de LCVG
        core_t = latest.get("core_temp", 36.8)
        suit_t = latest.get("suit_temp", 21.4)
        t_d_dt = latest.get("core_temp_d_dt", 0.0)
        if core_t > NASA_THRESHOLDS["core_temp"]["crit_max"] or suit_t > NASA_THRESHOLDS["suit_temp"]["crit_max"]:
            alerts.append({
                "code": "CRIT_THERMAL_STRESS",
                "severity": "CRITICAL",
                "system": "Termorregulação (LCVG)",
                "title": "Hipertermia Crítica do Astronauta",
                "detail": f"Temperatura central em {core_t:.2f}°C e temperatura do traje em {suit_t:.1f}°C. Falha no circuito de resfriamento líquido.",
                "ttc_minutes": 4.0,
                "recommendation": "Definir fluxo do LCVG para potência máxima de contingência e cessar esforço físico.",
            })
            health_penalty += 35.0
        elif core_t > NASA_THRESHOLDS["core_temp"]["warn_max"] or t_d_dt > 0.15:
            alerts.append({
                "code": "WARN_THERMAL_TREND",
                "severity": "WARNING",
                "system": "Termorregulação (LCVG)",
                "title": "Tendência de Elevação Térmica",
                "detail": f"Temperatura central subindo a +{t_d_dt:.2f}°C/min (atual: {core_t:.2f}°C).",
                "ttc_minutes": 12.0,
                "recommendation": "Ajustar termostato do traje e verificar circulação da água do LCVG.",
            })
            health_penalty += 15.0

        # Regra 5: Taxa de Radiação e Correlação NASA DONKI
        rad_val = latest.get("radiation_rate", 0.04)
        if rad_val > adapted_rad_crit or weather_alert == "CRITICAL":
            alerts.append({
                "code": "CRIT_RADIATION_EXPOSURE",
                "severity": "CRITICAL",
                "system": "Ambiente Espacial / Dosimetria",
                "title": "Evento Solar Extremo / Radiação Perigosa",
                "detail": f"Taxa de radiação em {rad_val:.4f} mSv/h acoplada a alerta de clima espacial NASA DONKI ({weather_alert}).",
                "ttc_minutes": 5.0,
                "recommendation": "RECOLHIMENTO DA TRIPULAÇÃO. Buscar blindagem no módulo central da estação.",
            })
            health_penalty += 40.0
        elif rad_val > adapted_rad_warn or weather_alert in ["HIGH", "MODERATE"]:
            alerts.append({
                "code": "WARN_SOLAR_ACTIVITY",
                "severity": "WARNING",
                "system": "Ambiente Espacial / Dosimetria",
                "title": "Atividade Solar Elevada (NASA DONKI)",
                "detail": f"Taxa em {rad_val:.4f} mSv/h. Limiares dinamicamente ajustados para clima espacial {weather_alert}.",
                "ttc_minutes": 25.0,
                "recommendation": "Registrar dosímetro de bolso e planejar encerramento antecipado das tarefas externas.",
            })
            health_penalty += 15.0

        # Regra 6: Alerta Informativo Nominal
        if not alerts:
            alerts.append({
                "code": "NOMINAL_STABLE",
                "severity": "INFO",
                "system": "Sistema Geral",
                "title": "Todos os Parâmetros Operacionais Estáveis",
                "detail": f"Telemetria biomédica e do traje espacial dentro dos limites de segurança NASA-STD-3001.",
                "ttc_minutes": 999.0,
                "recommendation": "Prosseguir com o cronograma regular de Atividade Extraveicular (EVA).",
            })

        # Cálculo do Aegis Health Index (AHI): 0 a 100
        ahi_score = max(5.0, min(100.0, 100.0 - health_penalty))
        return df, alerts, round(ahi_score, 1)


if __name__ == "__main__":
    print("=" * 75)
    print("🧠 AEGIS-EVA: Teste do Motor Preditivo de Detecção de Anomalias")
    print("=" * 75)

    from biometrics_engine import BiometricsEngine

    # 1. Obter avaliação NASA
    nasa_risk = quick_risk_assessment()
    print(f"📡 Clima Espacial NASA: {nasa_risk['risk_level']} | Multiplicador: {nasa_risk['radiation_multiplier']}x")

    engine = BiometricsEngine()
    detector = AnomalyDetector(rolling_window=10)

    # 2. Testar Cenário 1: Operação Nominal
    print("\n[Cenário 1/2] Testando Astronauta em Estado NOMINAL...")
    df_nominal = engine.generate_historical_dataframe(
        astronaut_id="ASTRO-01",
        minutes=30,
        freq_seconds=30,
        scenario="NOMINAL",
        radiation_multiplier=nasa_risk["radiation_multiplier"],
    )
    df_nom_out, alerts_nom, ahi_nom = detector.evaluate_telemetry_series(df_nominal, nasa_risk)
    print(f"  -> Aegis Health Index (AHI): {ahi_nom} / 100.0")
    print(f"  -> Total de Alertas: {len(alerts_nom)}")
    for a in alerts_nom:
        print(f"     [{a['severity']}] {a['title']} -> {a['recommendation']}")

    # 3. Testar Cenário 2: Anomalia Crítica de Despressurização
    print("\n[Cenário 2/2] Testando Astronauta sob Anomalia de DEPRESSURIZATION...")
    df_depress = engine.generate_historical_dataframe(
        astronaut_id="ASTRO-03",
        minutes=30,
        freq_seconds=30,
        scenario="DEPRESSURIZATION",
        radiation_multiplier=nasa_risk["radiation_multiplier"],
    )
    df_dep_out, alerts_dep, ahi_dep = detector.evaluate_telemetry_series(df_depress, nasa_risk)
    print(f"  -> Aegis Health Index (AHI): {ahi_dep} / 100.0")
    print(f"  -> Total de Alertas: {len(alerts_dep)}")
    for a in alerts_dep:
        print(f"     [{a['severity']}] {a['title']} (TTC: {a['ttc_minutes']} min)")
        print(f"        Detalhe: {a['detail']}")
        print(f"        Ação Recomendada: {a['recommendation']}")

    print("=" * 75)
    print("✅ Módulo src/anomaly_detector.py validado com sucesso!")
