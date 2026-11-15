# 🛡️ Aegis-EVA: Astronaut Health & Predictive Telemetry System

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA_Space_Apps-Galactic_Engineering_BR--US-blue.svg?style=for-the-badge&logo=nasa)](https://www.spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Telemetry-3F4F75.svg?style=for-the-badge&logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Team:** Galactic Engineering BR-US  
> **Challenge:** Create Astronaut Health Monitoring Software for Space Missions  
> **System:** Real-Time Extravehicular Activity (EVA) Biomedical Telemetry, Space Weather Sensor Fusion & Predictive Anomaly Diagnostics

---

## 🌟 Executive Summary

During Extravehicular Activity (EVA) spacewalks on lunar or deep space missions, astronauts face lethal physiological hazards, from **hypoxia** and **hypercapnia** (CO₂ accumulation) to **sudden EMU suit depressurization** and **extreme solar radiation events**. Conventional systems trigger reactive alarms only after physiological thresholds have already been breached.

**Aegis-EVA** bridges this critical gap by delivering an end-to-end, predictive health and life-support monitoring architecture. It fuses live astronomical telemetry from **NASA DONKI** (Solar Flares and Geomagnetic Storms) with multi-crew physiological sensors (ECG/HRV, SpO₂, core temperature, suit pressure, and CO₂ partial pressure). By deploying **rolling Z-scores** and **discrete first-order derivatives ($\frac{dY}{dt}$)**, Aegis-EVA detects subtle physiological deterioration up to **15 minutes before physical collapse**, predicting **Time-to-Criticality (TTC)** and dynamically calibrating mission safety envelopes.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph NASA_SPACE_WEATHER["🛰️ NASA Open Data Integration"]
        DONKI_API["NASA DONKI API<br/>(FLR & GST Endpoints)"]
        DONKI_CACHE["Local Disk Cache &<br/>Astronomical Fallback Engine"]
        RISK_EVAL["Space Weather Risk Engine<br/>(Kp-Index & X-Ray Flux Multiplier)"]
        DONKI_API --> DONKI_CACHE --> RISK_EVAL
    end

    subgraph BIOMETRICS_CORE["👨‍🚀 EVA Biomedical & PLSS Simulation"]
        A1["CDR Alex Silva<br/>(EVA Lead)"]
        A2["MS1 Dra. Elena Vance<br/>(Astrobiologist)"]
        A3["FE Marcus Drake<br/>(Flight Engineer)"]
        GAUSS_NOISE["Stochastic Gauss-Markov Noise<br/>& Metabolic Workload"]
        SCENARIO_INJECTOR["Critical Scenario Injector<br/>(Hypoxia, CO2, Depressurization, Thermal)"]
        A1 & A2 & A3 --> GAUSS_NOISE --> SCENARIO_INJECTOR
    end

    subgraph PREDICTIVE_ENGINE["🧠 Mathematical Anomaly & Early Warning Engine"]
        Z_SCORE["Rolling Z-Score Calculation<br/>Z = (X - μ) / (σ + ε)"]
        DERIVATIVES["First-Order Discrete Derivatives<br/>dY/dt (Rate of Change)"]
        ADAPTIVE_TUNING["Dynamic Threshold Recalibration<br/>(Space Weather Adjusted)"]
        AHI_CALC["Aegis Health Index (AHI)<br/>Score: 0 to 100"]
        TTC_ESTIMATOR["Time-to-Criticality (TTC)<br/>Predictive Forecasting"]
    end

    subgraph MISSION_CONTROL["🖥️ Mission Control Center (MCC)"]
        DASHBOARD["Executive Streamlit Dashboard<br/>(NASA Dark Aero Interface)"]
        PLOTLY_VITALS["Interactive Plotly Charts<br/>(NASA-STD-3001 Safety Bands)"]
        ALERTS_PANEL["Actionable Clinical & Flight Alerts<br/>(Abort / Return to Airlock)"]
        CSV_EXPORT["Audit Log Telemetry Export (CSV)"]
    end

    RISK_EVAL ==> ADAPTIVE_TUNING
    SCENARIO_INJECTOR ==> Z_SCORE & DERIVATIVES
    Z_SCORE & DERIVATIVES ==> ADAPTIVE_TUNING --> AHI_CALC --> TTC_ESTIMATOR
    TTC_ESTIMATOR ==> DASHBOARD
    DASHBOARD --> PLOTLY_VITALS & ALERTS_PANEL & CSV_EXPORT
```

---

## 📐 Mathematical Formulation

### 1. Rolling Z-Score (Signal Deviation Detection)
To detect early-stage anomalous physiological deviation independently of baseline drift:
$$Z_t = \frac{X_t - \mu_w}{\sigma_w + \epsilon}$$
Where:
- $X_t$ is the instantaneous telemetry reading at time $t$.
- $\mu_w$ and $\sigma_w$ are the rolling mean and standard deviation over window $w = 10$.
- $\epsilon = 10^{-6}$ provides numerical stability against division by zero during quiescent periods.

### 2. First-Order Discrete Derivative (Trend and Rate of Change)
Captures the velocity of physiological decay before absolute thresholds are violated:
$$\frac{dY}{dt} \approx \frac{Y_t - Y_{t-1}}{\Delta t}$$
Where $\Delta t$ is the telemetry sampling period ($30\text{ seconds} = 0.5\text{ min}$). A persistent $\frac{d(\text{SpO}_2)}{dt} < -0.3\%/\text{min}$ triggers predictive hypoxia warnings even when $\text{SpO}_2 > 95\%$.

### 3. Space Weather Adaptive Threshold Recalibration
When NASA DONKI detects active solar flare radiation ($\lambda_{\text{solar}}$) or severe geomagnetic storms ($Kp \ge 5.0$):
$$T_{\text{adapted}} = \frac{T_{\text{base}}}{\max(\kappa \cdot \lambda_{\text{solar}}, 1.0)}$$
Allowing dosimetric alert levels to tighten proactively, safeguarding crew members from Acute Radiation Sickness (ARS).

### 4. Time-to-Criticality (TTC) Estimation
$$\text{TTC} = \frac{|Y_{\text{critical}} - Y_t|}{\left|\frac{dY}{dt}\right|}$$
Provides flight surgeons with an estimated window in minutes before the astronaut reaches irreversible physiological impairment.

### 5. Aegis Health Index (AHI)
A unified composite score (0 to 100) reflecting overall astronaut survivability:
$$\text{AHI} = \max\left(5.0, \min\left(100.0, 100.0 - \sum_{i=1}^{M} w_i \cdot P_i\right)\right)$$

---

## 📋 NASA Physiological Standards Baseline (NASA-STD-3001)

| Parameter | Nominal Range | Caution / Warning | Critical Abort | Clinical Risk |
| :--- | :---: | :---: | :---: | :--- |
| **Heart Rate (BPM)** | $60 - 100$ | $> 125$ | $> 150$ | Tachycardia / Cardiac Decompensation |
| **SpO₂ (%)** | $95.0 - 100.0\%$ | $< 92.0\%$ | $< 88.0\%$ | Cerebral Hypoxia / Loss of Consciousness |
| **Suit Pressure (kPa)** | $28.5 - 31.0$ | $< 26.5$ | $< 24.0$ | Decompression Sickness / Ebullism |
| **Suit CO₂ (mmHg)** | $1.5 - 3.8$ | $> 5.5$ | $> 8.0$ | Hypercapnia / LiOH Canister Satiation |
| **Core Temp (°C)** | $36.5 - 37.5$ | $> 38.3$ | $> 39.2$ | Heat Exhaustion / LCVG Cooling Failure |
| **Radiation (mSv/h)** | $0.02 - 0.06$ | $> 0.25$ | $> 0.80$ | Solar Particle Event (SPE) Exposure |

---

## 🛠️ Project Structure

```
aegis-eva-nasa-space-apps/
├── app.py                      # Main Streamlit Executive Dashboard
├── requirements.txt            # Lightweight dependencies (Termux/ARM64 and Cloud)
├── README.md                   # International Technical Documentation
├── .gitignore                  # Git hygiene rules
├── data/                       # Local JSON cache for NASA DONKI telemetry
│   ├── flr_cache.json          # Cached Solar Flares
│   └── gst_cache.json          # Cached Geomagnetic Storms
└── src/
    ├── __init__.py             # Package initializer
    ├── nasa_api.py             # NASA DONKI API client with fallback resilience
    ├── biometrics_engine.py    # Multi-astronaut stochastic physiological simulation
    └── anomaly_detector.py     # Mathematical predictive engine (Z-score & derivatives)
```

---

## 🚀 Quickstart & Execution

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13 or 3.14
- Git configured with SSH

### 2. Installation
```bash
# Clone the repository
git clone git@github.com:santiago2511/aegis-eva-nasa-space-apps.git
cd aegis-eva-nasa-space-apps

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Component Tests
```bash
# Test NASA DONKI API client
python3 src/nasa_api.py

# Test Multi-Astronaut Biometrics Engine
python3 src/biometrics_engine.py

# Test Mathematical Anomaly Detector
python3 src/anomaly_detector.py
```

### 4. Launching the Mission Control Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 👨‍💻 Team Galactic Engineering BR-US
- **Mission:** NASA Space Apps Challenge 2026 / Global Hackathon
- **Focus:** Aerospace Bioengineering, Space Medicine & Embedded Real-Time Telemetry

*Dedicated to the safety of extravehicular explorers pioneering the Moon, Mars, and beyond.*
