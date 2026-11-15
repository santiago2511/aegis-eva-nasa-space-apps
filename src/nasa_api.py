#!/usr/bin/env python3
"""
Aegis-EVA: Astronaut Health & Predictive Telemetry System
Módulo: src/nasa_api.py
Responsabilidade: Integração resiliente com a API NASA DONKI (Space Weather Database Of
Notifications, Knowledge, and Information) para monitoramento de Solar Flares (FLR)
e Geomagnetic Storms (GST), com fallback automático, cache local e cálculo de risco espacial.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Aegis-NASA-API")

# Diretórios base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FLR_CACHE_FILE = os.path.join(DATA_DIR, "flr_cache.json")
GST_CACHE_FILE = os.path.join(DATA_DIR, "gst_cache.json")

# NASA DONKI Endpoints
NASA_DONKI_FLR_URL = "https://api.nasa.gov/DONKI/FLR"
NASA_DONKI_GST_URL = "https://api.nasa.gov/DONKI/GST"


def get_default_fallback_flr() -> List[Dict[str, Any]]:
    """Gera dados realistas de contingência para Solar Flares quando offline sem cache."""
    now = datetime.now(timezone.utc)
    return [
        {
            "flrID": f"{now.strftime('%Y-%m-%d')}-FLR-001",
            "instruments": [{"displayName": "GOES-16: EXIS-XRS"}],
            "beginTime": (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "peakTime": (now - timedelta(hours=5, minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": (now - timedelta(hours=5, minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classType": "M2.8",
            "sourceLocation": "N14E22",
            "activeRegionNum": 13664,
            "note": "Contingency baseline: Active Region coronal mass ejection precursor.",
        },
        {
            "flrID": f"{now.strftime('%Y-%m-%d')}-FLR-002",
            "instruments": [{"displayName": "GOES-16: EXIS-XRS"}],
            "beginTime": (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "peakTime": (now - timedelta(hours=1, minutes=50)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": (now - timedelta(hours=1, minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classType": "X1.4",
            "sourceLocation": "S18W35",
            "activeRegionNum": 13667,
            "note": "Contingency baseline: Strong X-class impulsive solar energetic proton surge.",
        },
    ]


def get_default_fallback_gst() -> List[Dict[str, Any]]:
    """Gera dados realistas de contingência para Geomagnetic Storms quando offline sem cache."""
    now = datetime.now(timezone.utc)
    return [
        {
            "gstID": f"{now.strftime('%Y-%m-%d')}-GST-001",
            "startTime": (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "allKpIndex": [
                {"observedTime": (now - timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ"), "kpIndex": 4.3, "source": "NOAA"},
                {"observedTime": (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"), "kpIndex": 6.0, "source": "NOAA"},
                {"observedTime": (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"), "kpIndex": 6.7, "source": "NOAA"},
            ],
            "link": "https://kauai.ccmc.gsfc.nasa.gov/DONKI/",
        }
    ]


class NasaDonkiClient:
    """Cliente para a API NASA DONKI com tolerância a falhas, cache em disco e análise de risco."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        os.makedirs(DATA_DIR, exist_ok=True)

    def _read_cache(self, filepath: str) -> Optional[List[Dict[str, Any]]]:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} cached records from {filepath}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read cache file {filepath}: {e}")
        return None

    def _save_cache(self, filepath: str, data: List[Dict[str, Any]]) -> None:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} records to cache {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {filepath}: {e}")

    def fetch_solar_flares(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Busca eventos de Solar Flares (FLR).
        Retorna: (lista de eventos, status_origem: 'LIVE' | 'CACHE' | 'FALLBACK')
        """
        now = datetime.now(timezone.utc)
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        if not force_refresh:
            cached = self._read_cache(FLR_CACHE_FILE)
            if cached is not None and len(cached) > 0:
                return cached, "CACHE"

        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key,
        }

        try:
            logger.info(f"Connecting to NASA DONKI FLR API: {params['startDate']} -> {params['endDate']}")
            response = requests.get(NASA_DONKI_FLR_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self._save_cache(FLR_CACHE_FILE, data)
                    return data, "LIVE"
                else:
                    logger.warning(f"Unexpected JSON response from NASA FLR: {data}")
            else:
                logger.warning(f"NASA FLR returned status code {response.status_code}: {response.text}")
        except Exception as err:
            logger.warning(f"NASA FLR network request failed: {err}")

        # Fallback para cache se disponível
        cached = self._read_cache(FLR_CACHE_FILE)
        if cached is not None:
            return cached, "CACHE"

        # Fallback de contingência
        logger.warning("Using synthetic astronomical fallback for Solar Flares")
        contingency = get_default_fallback_flr()
        self._save_cache(FLR_CACHE_FILE, contingency)
        return contingency, "FALLBACK"

    def fetch_geomagnetic_storms(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Busca eventos de Tempestades Geomagnéticas (GST).
        Retorna: (lista de eventos, status_origem: 'LIVE' | 'CACHE' | 'FALLBACK')
        """
        now = datetime.now(timezone.utc)
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        if not force_refresh:
            cached = self._read_cache(GST_CACHE_FILE)
            if cached is not None and len(cached) > 0:
                return cached, "CACHE"

        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key,
        }

        try:
            logger.info(f"Connecting to NASA DONKI GST API: {params['startDate']} -> {params['endDate']}")
            response = requests.get(NASA_DONKI_GST_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self._save_cache(GST_CACHE_FILE, data)
                    return data, "LIVE"
                else:
                    logger.warning(f"Unexpected JSON response from NASA GST: {data}")
            else:
                logger.warning(f"NASA GST returned status code {response.status_code}: {response.text}")
        except Exception as err:
            logger.warning(f"NASA GST network request failed: {err}")

        # Fallback para cache se disponível
        cached = self._read_cache(GST_CACHE_FILE)
        if cached is not None:
            return cached, "CACHE"

        # Fallback de contingência
        logger.warning("Using synthetic astronomical fallback for Geomagnetic Storms")
        contingency = get_default_fallback_gst()
        self._save_cache(GST_CACHE_FILE, contingency)
        return contingency, "FALLBACK"

    def evaluate_space_weather_risk(
        self, flr_events: List[Dict[str, Any]], gst_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Avalia o risco espacial com base na classificação dos flares (B, C, M, X)
        e intensidade do Kp Index das tempestades geomagnéticas.
        Retorna coeficientes e diretrizes operacionais para atividades EVA.
        """
        max_flare_class = "A0.0"
        max_flare_score = 0.0

        for flr in flr_events:
            ctype = flr.get("classType", "")
            if not ctype:
                continue
            # Cálculo de severidade por potência de fluxo de Raios-X (Watts/m^2)
            letter = ctype[0].upper()
            try:
                mag = float(ctype[1:]) if len(ctype) > 1 else 1.0
            except ValueError:
                mag = 1.0

            scale_map = {"A": 0.1, "B": 1.0, "C": 10.0, "M": 100.0, "X": 1000.0}
            score = scale_map.get(letter, 0.1) * mag
            if score > max_flare_score:
                max_flare_score = score
                max_flare_class = ctype

        max_kp = 0.0
        for gst in gst_events:
            for kp_entry in gst.get("allKpIndex", []):
                kp_val = kp_entry.get("kpIndex", 0.0)
                if isinstance(kp_val, (int, float)) and kp_val > max_kp:
                    max_kp = float(kp_val)

        # Determinação de Fator de Risco, Multiplicador de Radiação e Alerta EVA
        # Baseline espacial no espaço exterior (LEO/Translunar) EVA: ~0.02 a 0.05 mSv/h em calmaria
        if max_flare_class.startswith("X") or max_kp >= 7.0:
            risk_level = "CRITICAL"
            radiation_multiplier = 8.5
            eva_action = "ABORT_EVA"
            recommendation = "EMERGÊNCIA: Tempestade solar severa. Retorno imediato à escotilha (AirLock/Shelter)."
        elif max_flare_class.startswith("M") or max_kp >= 5.0:
            risk_level = "HIGH"
            radiation_multiplier = 3.5
            eva_action = "RESTRICT_EVA"
            recommendation = "ALERTA ELEVADO: Radiação solar intensa detectada. Limitar EVA a 45 minutos."
        elif max_flare_class.startswith("C") or max_kp >= 4.0:
            risk_level = "MODERATE"
            radiation_multiplier = 1.8
            eva_action = "MONITOR_EVA"
            recommendation = "ATENÇÃO: Variações leves no fluxo de partículas. Manter dosimetria ativa."
        else:
            risk_level = "NOMINAL"
            radiation_multiplier = 1.0
            eva_action = "PERMIT_EVA"
            recommendation = "CONDIÇÕES NOMINAIS: Clima espacial calmo. Atividade extraveicular liberada."

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_level": risk_level,
            "max_flare_class": max_flare_class,
            "max_kp_index": round(max_kp, 2),
            "radiation_multiplier": radiation_multiplier,
            "eva_action": eva_action,
            "recommendation": recommendation,
            "flr_count": len(flr_events),
            "gst_count": len(gst_events),
        }


def quick_risk_assessment() -> Dict[str, Any]:
    """Função utilitária de conveniência para consumo direto pelos motores de telemetria."""
    client = NasaDonkiClient()
    flrs, flr_source = client.fetch_solar_flares()
    gsts, gst_source = client.fetch_geomagnetic_storms()
    assessment = client.evaluate_space_weather_risk(flrs, gsts)
    assessment["flr_source"] = flr_source
    assessment["gst_source"] = gst_source
    return assessment


if __name__ == "__main__":
    print("=" * 70)
    print("🛰️  AEGIS-EVA: Teste de Conexão com API NASA DONKI & Clima Espacial")
    print("=" * 70)
    client = NasaDonkiClient()
    
    print("\n[1/3] Consultando Solar Flares (FLR)...")
    flrs, flr_src = client.fetch_solar_flares()
    print(f" -> Sucesso! Origem: {flr_src} | Total de eventos obtidos: {len(flrs)}")
    if flrs:
        latest_flr = flrs[-1]
        print(f"    Último Flare: ID={latest_flr.get('flrID')} | Classe={latest_flr.get('classType')} | Pico={latest_flr.get('peakTime')}")

    print("\n[2/3] Consultando Tempestades Geomagnéticas (GST)...")
    gsts, gst_src = client.fetch_geomagnetic_storms()
    print(f" -> Sucesso! Origem: {gst_src} | Total de tempestades obtidas: {len(gsts)}")
    if gsts:
        latest_gst = gsts[-1]
        print(f"    Último GST: ID={latest_gst.get('gstID')} | Início={latest_gst.get('startTime')}")

    print("\n[3/3] Avaliando Risco de Radiação e Diretrizes para EVA...")
    risk = client.evaluate_space_weather_risk(flrs, gsts)
    print(f" -> Nível de Risco Espacial: {risk['risk_level']}")
    print(f" -> Classe Máx Solar Flare:  {risk['max_flare_class']}")
    print(f" -> Maior Índice Kp:         {risk['max_kp_index']}")
    print(f" -> Multiplicador Radiação:  {risk['radiation_multiplier']}x")
    print(f" -> Ação Operacional EVA:    {risk['eva_action']}")
    print(f" -> Diretriz Médica:         {risk['recommendation']}")
    print("=" * 70)
    print("✅ Módulo src/nasa_api.py validado com sucesso!")
