#!/usr/bin/env python3
"""
Aegis-EVA: Astronaut Health & Predictive Telemetry System
Módulo: src/biometrics_engine.py
Responsabilidade: Motor fisiológico e de suporte de vida (PLSS) para simulação em tempo real
e histórica de telemetria de astronautas em Atividade Extraveicular (EVA), com perfis individualizados,
ruído estocástico Gaussiano e injeção controlada de cenários críticos.
Conforme as normas da NASA (NASA-STD-3001: Space Flight Human-System Standard).
"""

import os
import random
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    from src.nasa_api import quick_risk_assessment
except ImportError:
    from nasa_api import quick_risk_assessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Aegis-Biometrics")

# Constantes Normativas Médicas da NASA (NASA-STD-3001)
NASA_THRESHOLDS = {
    "heart_rate": {"nominal_min": 60, "nominal_max": 100, "warn_max": 125, "crit_max": 150, "unit": "BPM"},
    "hrv_rmssd": {"nominal_min": 35, "nominal_max": 80, "warn_min": 25, "crit_min": 15, "unit": "ms"},
    "spo2": {"nominal_min": 95.0, "nominal_max": 100.0, "warn_min": 92.0, "crit_min": 88.0, "unit": "%"},
    "core_temp": {"nominal_min": 36.5, "nominal_max": 37.5, "warn_max": 38.3, "crit_max": 39.2, "unit": "°C"},
    "suit_temp": {"nominal_min": 18.0, "nominal_max": 24.0, "warn_max": 28.0, "crit_max": 32.0, "unit": "°C"},
    "suit_pressure": {"nominal_min": 28.5, "nominal_max": 31.0, "warn_min": 26.5, "crit_min": 24.0, "unit": "kPa"}, # EMU nominal: 4.3 psi = 29.6 kPa
    "co2_level": {"nominal_min": 1.5, "nominal_max": 3.8, "warn_max": 5.5, "crit_max": 8.0, "unit": "mmHg"},
    "radiation_rate": {"nominal_min": 0.02, "nominal_max": 0.06, "warn_max": 0.25, "crit_max": 0.80, "unit": "mSv/h"},
}

CREW_MEMBERS = [
    {
        "id": "ASTRO-01",
        "name": "CDR Alex Silva",
        "role": "Comandante de Missão / EVA Lead",
        "base_hr": 68,
        "base_hrv": 52.0,
        "base_spo2": 98.4,
        "base_temp": 36.8,
        "suit_id": "EMU-X4-Alpha",
    },
    {
        "id": "ASTRO-02",
        "name": "MS1 Dra. Elena Vance",
        "role": "Especialista de Missão / Astrobióloga",
        "base_hr": 74,
        "base_hrv": 48.0,
        "base_spo2": 98.9,
        "base_temp": 36.7,
        "suit_id": "EMU-X4-Bravo",
    },
    {
        "id": "ASTRO-03",
        "name": "FE Marcus Drake",
        "role": "Engenheiro de Voo / Suporte Técnico EVA",
        "base_hr": 71,
        "base_hrv": 45.0,
        "base_spo2": 98.0,
        "base_temp": 36.9,
        "suit_id": "EMU-X4-Charlie",
    },
]


@dataclass
class TelemetryPoint:
    timestamp: str
    astronaut_id: str
    astronaut_name: str
    heart_rate: float        # BPM
    hrv_rmssd: float         # ms
    spo2: float              # %
    core_temp: float         # °C
    suit_temp: float         # °C
    suit_pressure: float     # kPa
    co2_level: float         # mmHg
    radiation_rate: float    # mSv/h
    o2_reserve_pct: float    # %
    battery_pct: float       # %
    suit_integrity_pct: float# %
    metabolic_rate_watts: float # Watts (gasto calórico da EVA)
    scenario_tag: str        # NOMINAL, HYPOXIA, HYPERCAPNIA, DEPRESSURIZATION, THERMAL, RADIATION


class BiometricsEngine:
    """Motor fisiológico e biomédico para atividade extraveicular espacial."""

    def __init__(self, crew: Optional[List[Dict[str, Any]]] = None):
        self.crew = crew or CREW_MEMBERS

    def get_crew_list(self) -> List[Dict[str, Any]]:
        return self.crew

    def get_astronaut_by_id(self, astronaut_id: str) -> Optional[Dict[str, Any]]:
        for member in self.crew:
            if member["id"] == astronaut_id:
                return member
        return None

    def simulate_telemetry_point(
        self,
        astronaut: Dict[str, Any],
        timestamp: datetime,
        elapsed_step: int = 0,
        total_steps: int = 60,
        scenario: str = "NOMINAL",
        radiation_multiplier: float = 1.0,
    ) -> TelemetryPoint:
        """
        Calcula um ponto de telemetria altamente realista com base na fisiologia base do astronauta,
        desgaste de consumíveis da missão e cenário anômalo ativo.
        """
        # Fator de progresso temporal (0.0 no início até 1.0 no final)
        progress = elapsed_step / max(total_steps, 1)

        # Consumo de O2 e Bateria ao longo de uma EVA de 6 a 8 horas
        o2_reserve = max(20.0, 96.0 - (progress * 18.0) + random.uniform(-0.4, 0.4))
        battery = max(15.0, 94.0 - (progress * 22.0) + random.uniform(-0.3, 0.3))
        suit_integrity = 100.0

        # Dinâmica básica fisiológica com flutuação estocástica Gaussiana
        hr = float(np.random.normal(astronaut["base_hr"], 3.0))
        hrv = float(np.random.normal(astronaut["base_hrv"], 3.5))
        spo2 = float(np.random.normal(astronaut["base_spo2"], 0.35))
        core_t = float(np.random.normal(astronaut["base_temp"], 0.12))
        suit_t = float(np.random.normal(21.4, 0.45))
        suit_p = float(np.random.normal(29.6, 0.15))
        co2 = float(np.random.normal(2.4, 0.25))
        
        # Radiação basal espacial com aplicação de clima solar DONKI
        base_rad = float(np.random.normal(0.045, 0.008))
        rad_rate = max(0.015, base_rad * radiation_multiplier)
        metabolic_watts = float(np.random.normal(240.0, 25.0)) # Atividade física em microgravidade

        # Injeção de Cenários Críticos para Demonstração Executiva
        if scenario == "HYPOXIA":
            # Redução de SpO2 progressiva, taquicardia compensatória seguida de descompensação
            decay = min(1.0, progress * 1.8)
            spo2 -= (decay * 11.0)
            hr += (decay * 38.0)
            hrv -= (decay * 25.0)
            o2_reserve -= (decay * 15.0)

        elif scenario == "HYPERCAPNIA":
            # Acúmulo de CO2 por saturação do canister de LiOH/Metox
            surge = min(1.0, progress * 2.0)
            co2 += (surge * 6.5)
            hr += (surge * 28.0)
            core_t += (surge * 0.7)
            hrv -= (surge * 18.0)

        elif scenario == "DEPRESSURIZATION":
            # Queda de pressão no traje espacial (vazamento microestrutural)
            loss = min(1.0, progress * 2.2)
            suit_p -= (loss * 6.2)
            suit_integrity -= (loss * 32.0)
            hr += (loss * 45.0) # Resposta aguda de estresse simpático
            hrv = max(10.0, hrv - (loss * 30.0))
            spo2 -= (loss * 5.0)

        elif scenario == "THERMAL":
            # Falha no circuito de arrefecimento líquido (LCVG)
            heat = min(1.0, progress * 1.8)
            suit_t += (heat * 9.5)
            core_t += (heat * 2.2)
            hr += (heat * 32.0)
            metabolic_watts += (heat * 90.0)

        elif scenario == "RADIATION":
            # Surto agudo de radiação solar (SPE - Solar Proton Event)
            burst = min(1.0, progress * 2.5)
            rad_rate += (burst * 1.45)
            hr += (burst * 15.0)

        # Clamping de segurança para valores fisiologicamente coerentes
        hr = float(np.clip(hr, 45.0, 195.0))
        hrv = float(np.clip(hrv, 8.0, 120.0))
        spo2 = float(np.clip(spo2, 75.0, 100.0))
        core_t = float(np.clip(core_t, 34.0, 41.5))
        suit_t = float(np.clip(suit_t, 10.0, 38.0))
        suit_p = float(np.clip(suit_p, 18.0, 34.0))
        co2 = float(np.clip(co2, 0.5, 14.0))
        rad_rate = float(np.clip(rad_rate, 0.01, 5.0))
        o2_reserve = float(np.clip(o2_reserve, 0.0, 100.0))
        battery = float(np.clip(battery, 0.0, 100.0))
        suit_integrity = float(np.clip(suit_integrity, 0.0, 100.0))

        return TelemetryPoint(
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            astronaut_id=astronaut["id"],
            astronaut_name=astronaut["name"],
            heart_rate=round(hr, 1),
            hrv_rmssd=round(hrv, 1),
            spo2=round(spo2, 1),
            core_temp=round(core_t, 2),
            suit_temp=round(suit_t, 2),
            suit_pressure=round(suit_p, 2),
            co2_level=round(co2, 2),
            radiation_rate=round(rad_rate, 4),
            o2_reserve_pct=round(o2_reserve, 1),
            battery_pct=round(battery, 1),
            suit_integrity_pct=round(suit_integrity, 1),
            metabolic_rate_watts=round(metabolic_watts, 1),
            scenario_tag=scenario,
        )

    def generate_historical_dataframe(
        self,
        astronaut_id: str,
        minutes: int = 45,
        freq_seconds: int = 30,
        scenario: str = "NOMINAL",
        radiation_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """
        Gera um DataFrame de séries temporais consistente para análise preditiva e gráficos Plotly.
        """
        astronaut = self.get_astronaut_by_id(astronaut_id)
        if not astronaut:
            astronaut = self.crew[0]

        total_steps = max(10, int((minutes * 60) / freq_seconds))
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(seconds=total_steps * freq_seconds)

        records: List[Dict[str, Any]] = []
        for step in range(total_steps):
            pt_time = start_time + timedelta(seconds=step * freq_seconds)
            point = self.simulate_telemetry_point(
                astronaut=astronaut,
                timestamp=pt_time,
                elapsed_step=step,
                total_steps=total_steps,
                scenario=scenario,
                radiation_multiplier=radiation_multiplier,
            )
            records.append(asdict(point))

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["timestamp"])
        return df

    def get_crew_instant_telemetry(
        self,
        radiation_multiplier: float = 1.0,
        scenarios: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Gera instantaneamente a leitura atualizada para toda a tripulação de 3 astronautas.
        """
        now = datetime.now(timezone.utc)
        scenarios = scenarios or {}
        telemetry_batch: List[Dict[str, Any]] = []

        for member in self.crew:
            m_id = member["id"]
            scenario = scenarios.get(m_id, "NOMINAL")
            point = self.simulate_telemetry_point(
                astronaut=member,
                timestamp=now,
                elapsed_step=30,
                total_steps=60,
                scenario=scenario,
                radiation_multiplier=radiation_multiplier,
            )
            data = asdict(point)
            data["suit_id"] = member["suit_id"]
            data["role"] = member["role"]
            telemetry_batch.append(data)

        return telemetry_batch


if __name__ == "__main__":
    print("=" * 75)
    print("👨‍🚀 AEGIS-EVA: Teste de Simulação Biomédica da Tripulação EVA (NASA-STD-3001)")
    print("=" * 75)
    
    # 1. Consulta Risco Solar Real
    risk_summary = quick_risk_assessment()
    rad_mult = risk_summary.get("radiation_multiplier", 1.0)
    print(f"📡 NASA Space Weather: {risk_summary['risk_level']} (Multiplicador de Radiação: {rad_mult}x)")

    # 2. Inicialização do Motor
    engine = BiometricsEngine()
    print(f"👥 Tripulação carregada: {len(engine.get_crew_list())} astronautas ativos na missão.")

    # 3. Snapshot de Telemetria Instantânea
    print("\n📊 Snapshot de Telemetria em Tempo Real:")
    snapshot = engine.get_crew_instant_telemetry(radiation_multiplier=rad_mult)
    for astro in snapshot:
        print(f"  • [{astro['astronaut_id']}] {astro['astronaut_name']} ({astro['role']}):")
        print(f"     FC: {astro['heart_rate']} BPM | SpO2: {astro['spo2']}% | T.Core: {astro['core_temp']}°C")
        print(f"     Pressão Traje: {astro['suit_pressure']} kPa | CO2: {astro['co2_level']} mmHg | Rad: {astro['radiation_rate']} mSv/h")

    # 4. Geração de Série Temporal Histórica (Simulação de Anomalia)
    print("\n📈 Gerando Série Temporal Histórica (Cenário: HYPERCAPNIA - 45 min)...")
    df_anomaly = engine.generate_historical_dataframe(
        astronaut_id="ASTRO-01",
        minutes=45,
        freq_seconds=30,
        scenario="HYPERCAPNIA",
        radiation_multiplier=rad_mult,
    )
    print(f"  -> DataFrame gerado com {len(df_anomaly)} registros de telemetria.")
    print("  -> Amostra final da telemetria sob estresse hiperbárico/CO2:")
    print(df_anomaly[["timestamp", "heart_rate", "spo2", "co2_level", "radiation_rate"]].tail(3).to_string(index=False))

    print("=" * 75)
    print("✅ Módulo src/biometrics_engine.py validado com sucesso!")
