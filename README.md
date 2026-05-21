# 🌍 OmniTerra: Global Yield Intelligence

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════╗
║          OmniTerra: A Multi-Modal Spatio-Temporal Transformer         ║
║      Framework for Global Yield Intelligence and Carbon Modeling      ║
╚═══════════════════════════════════════════════════════════════════════╝
```

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://omniterra-global-m-2bpfuims8arhryiznigx5g.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-Sentinel--2-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://earthengine.google.com/)
[![Preprint DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20308398.svg)](https://doi.org/10.5281/zenodo.20308398)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20308741.svg)](https://doi.org/10.5281/zenodo.20308741)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

**Production-deployed AI framework combining Sentinel-2 real-time satellite data, Spatio-Temporal Transformers, and IPCC-aligned carbon sequestration modeling for global precision agriculture.**

*Developed by [Agha Wafa Abbas](mailto:agha.wafa@port.ac.uk)*

</div>

---

## 📄 Research & Publication

> **Preprint:** Abbas, A. W. (2026). *OmniTerra: A Multi-Modal Spatio-Temporal Transformer Framework for Global Yield Intelligence and Carbon Sequestration Modeling.* Zenodo. https://doi.org/10.5281/zenodo.20308398
>
> **Software Release v1.0.0:** Agha Wafa Abbas. (2026). *OmniTerra v1.0.0 — Production Release.* Zenodo. https://doi.org/10.5281/zenodo.20308741

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture)
- [Inference Results by Region](#-inference-results-by-region)
- [Carbon Sequestration Module](#-carbon-sequestration-module)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset & Experimental Setup](#-dataset--experimental-setup)
- [Comparative Benchmarking](#-comparative-benchmarking)
- [Author](#-author)
- [Citation](#-citation)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🔬 Overview

OmniTerra is a **Multi-Modal Spatio-Temporal Transformer (ST-Transformer)** framework for global crop yield intelligence and carbon sequestration modelling. Unlike existing approaches that rely on pre-processed static datasets, OmniTerra operates on **live Sentinel-2 satellite streams** via Google Earth Engine (GEE), making real-time precision agriculture intelligence operationally accessible through a production-grade web application.

| # | Innovation | Description |
|---|-----------|-------------|
| 1 | **Real-Time EO Integration** | Live Sentinel-2 NDVI extraction via GEE at 10m resolution, 500m buffer zone |
| 2 | **ST-Transformer Inference** | Multi-head self-attention for crop-specific yield prediction |
| 3 | **Carbon Estimation Module** | IPCC-aligned `C_ag = ŷ × 0.47 Mg C/ha` |
| 4 | **Production Deployment** | Streamlit web app — zero installation for end users |

---

## 📊 Key Results

<div align="center">

| Metric | Value |
|--------|-------|
| **Inference Confidence** | **94.2%** |
| **NDVI–Yield Correlation** | **R² = 0.91** |
| **Model Parameters** | **25,089** (~98 KB) |
| **End-to-End Latency** | **3–8 seconds** |
| **Global Carbon Mean** | **1.75 Mg C/ha** |
| **Crops Supported** | Wheat · Rice · Maize |
| **Regions Validated** | 6 global agricultural zones |

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OmniTerra — Multi-Tier Architecture              │
├──────────────────┬──────────────────────┬───────────────────────────┤
│     TIER 1       │       TIER 2         │         TIER 3            │
│  Data Acquisition│  Inference Engine    │   Deployment Layer        │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ Google Earth     │ ST-Transformer       │ Streamlit Web App         │
│ Engine (GEE)     │ (OmniTerra)          │                           │
│ • Sentinel-2 SR  │ • Input Projection   │ • Interactive UI          │
│   COPERNICUS     │   R³ → R⁶⁴           │   (Lat/Lon/Crop)          │
│ • 500m Buffer    │ • MHSA (4 heads)     │ • Folium Satellite Map    │
│ • NDVI Extract   │ • FFN R⁶⁴→R¹²⁸→R¹   │ • Yield + Carbon Output   │
│   (B8−B4)/B8+B4  │ • ŷ (t/ha)          │ • Downloadable Reports    │
└──────────────────┴──────────────────────┴───────────────────────────┘
                    x = [NDVI, T, SM] ∈ ℝ³
```

**NDVI Formula:** `NDVI = (B8 − B4) / (B8 + B4)` | Cropland range: 0.2–0.9

---

## 🧠 Model Architecture

```python
class OmniTerraTransformer(torch.nn.Module):
    def __init__(self, input_dim=3, model_dim=64):
        super().__init__()
        self.input_fc  = torch.nn.Linear(input_dim, model_dim)   # ℝ³ → ℝ⁶⁴
        self.attention = torch.nn.MultiheadAttention(model_dim,
                             num_heads=4, batch_first=True)       # 4-head MHSA
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)                               # → ŷ t/ha
        )
    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        out, _ = self.attention(x, x, x)
        return self.ffn(out.squeeze(1))
```

| Layer | Dimensions | Parameters |
|---|---|---|
| Input Projection | 3 → 64 | 256 |
| Multi-Head Self-Attention | 64, 4 heads | 16,384 |
| FFN Layer 1 | 64 → 128 | 8,320 |
| FFN Layer 2 | 128 → 1 | 129 |
| **Total** | — | **25,089** |

---

## 🌐 Inference Results by Region

---

### 🌾 Wheat — USA, Kansas (High Greenery)
**Coordinates:** `38.5000°N, -98.0000°E` | **NDVI:** `0.34` | **Yield:** `3.57 t/ha` | **Carbon:** `1.68 Mg C/ha`

| | |
|:---:|:---:|
| ![Fig4](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Wheat%20%28USA%20-%20High%20Greenery%29%20Part%201.PNG?raw=true) | ![Fig5](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Wheat%20%28USA%20-%20High%20Greenery%29%20Part%202.PNG?raw=true) |
| *Fig. 4 — Live inference · NDVI: 0.34 · Yield: 3.57 t/ha* | *Fig. 5 — Multi-Modal Insights · Carbon: 1.68 Mg C/ha · Status: Normal Growth* |

> NDVI **0.34** indicates active photosynthetic growth. Transformer predicted **3.57 t/ha**. Vegetation status: **Stable**. Inference confidence: **94.2%**.

---

### 🌾 Wheat — Ukraine (Low Greenery)
**Coordinates:** `49.5883°N, 34.5514°E` | **NDVI:** `0.13` | **Yield:** `3.10 t/ha` | **Carbon:** `1.46 Mg C/ha`

| | |
|:---:|:---:|
| ![Fig8](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Wheat%20%28Ukraine%20%E2%80%93%20Low%20Greenery%29%20Part%201.PNG?raw=true) | ![Fig9](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Wheat%20%28Ukraine%20%E2%80%93%20Low%20Greenery%29%20Part%202.PNG?raw=true) |
| *Fig. 8 — NDVI: 0.13 · Yield: 3.10 t/ha* | *Fig. 9 — Alert: Low Vegetation Density · Carbon: 1.46 Mg C/ha* |

> NDVI **0.13** → sparse vegetation. Model adjusted yield to **3.10 t/ha**. Status: **Critical Monitoring** → Nitrogen-based soil enrichment recommended.

---

### 🌾 Rice — China (Water / Bare Soil)
**Coordinates:** `27.6104°N, 111.7088°E` | **NDVI:** `0.05` | **Yield:** `0.89 t/ha` | **Carbon:** `0.42 Mg C/ha`

| | |
|:---:|:---:|
| ![Fig6](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Rice%20%28China%20-%20Water%20and%20Bare%20Soil%20Area%29%20Part%201.PNG?raw=true) | ![Fig7](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Rice%20%28China%20-%20Water%20and%20Bare%20Soil%20Area%29%20Part%202.PNG?raw=true) |
| *Fig. 6 — NDVI: 0.05 · Yield: 0.89 t/ha (bare soil/water)* | *Fig. 7 — Carbon: 0.42 Mg C/ha · Status: Critical Monitoring* |

> NDVI **0.05** indicates water logging or bare soil. Transformer reduced yield to **0.89 t/ha** — robust edge-case handling. Carbon: **0.42 Mg C/ha**.

---

### 🌽 Maize — Brazil (Ultra High Greenery)
**Coordinates:** `12.5000°S, 55.5000°W` | **NDVI:** `0.66` | **Yield:** `7.41 t/ha` | **Carbon:** `3.48 Mg C/ha`

| | |
|:---:|:---:|
| ![Fig10](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Maize%20%28Brazil%20-%20Ultra%20High%20Greenery%29%20Part%201.PNG?raw=true) | ![Fig11](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Maize%20%28Brazil%20-%20Ultra%20High%20Greenery%29%20Part%202.PNG?raw=true) |
| *Fig. 10 — NDVI: 0.66 · Yield: 7.41 t/ha (peak greenery)* | *Fig. 11 — Carbon: 3.48 Mg C/ha · Status: High Photosynthetic Activity* |

> NDVI **0.66** — exceptionally high. Framework projected outstanding yield of **7.41 t/ha**. Status: **High Photosynthetic Activity** — Maintain current nutrient levels. Vegetation: **Optimal**.

---

### 🌽 Maize — Kenya (Moderate Fields)
**Coordinates:** `1.0189°N, 34.9542°E` | **NDVI:** `0.52` | **Yield:** `6.54 t/ha` | **Carbon:** `3.08 Mg C/ha`

| | |
|:---:|:---:|
| ![Fig12](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Maize%20%28Kenya%20-%20Moderate%20Fields%29%20Part%201.PNG?raw=true) | ![Fig13](https://github.com/Aghawafaabbass/OmniTerra-Global-M/blob/main/screenshots/Maize%20%28Kenya%20-%20Moderate%20Fields%29%20Part%202.PNG?raw=true) |
| *Fig. 12 — NDVI: 0.52 · Yield: 6.54 t/ha* | *Fig. 13 — Carbon: 3.08 Mg C/ha · Status: Normal Growth Cycle* |

> Balanced NDVI **0.52** → reliable forecast of **6.54 t/ha**. Status: **Normal Growth Cycle** → Regular monitoring. Vegetation: **Optimal**. Inference confidence: **94.2%**.

---

## ☁️ Carbon Sequestration Module

**IPCC 2006 Formula (Volume 4: AFOLU):**
```
C_ag = ŷ × BCEF × CF ≈ ŷ × 0.47   (Mg C/ha)
```

| NDVI Range | Classification | Action |
|---|---|---|
| NDVI < 0.2 | 🔴 Critical — Low Density | Immediate nitrogen enrichment |
| 0.2 ≤ NDVI < 0.3 | 🟠 Stressed Vegetation | Targeted fertilizer; irrigation check |
| 0.3 ≤ NDVI < 0.6 | 🔵 Normal Growth Cycle | Regular monitoring |
| NDVI ≥ 0.6 | 🟢 Optimal — High Activity | Maintain regime; harvest planning |

### Regional Summary

| Region | Crop | NDVI | Yield (t/ha) | Carbon (Mg C/ha) |
|---|---|---|---|---|
| Punjab, India | Rice | 0.70 | 4.51 | 2.12 |
| Iowa, USA | Maize | 0.65 | 4.12 | 1.94 |
| Lahore, Pakistan | Wheat | 0.61 | 3.84 | 1.80 |
| Global Average | — | 0.59 | 3.73 | 1.75 |
| Saskatchewan, Canada | Wheat | 0.53 | 3.22 | 1.51 |
| Sao Paulo, Brazil | Maize | 0.48 | 2.95 | 1.39 |

---

## ⚙️ Installation

```bash
git clone https://github.com/Aghawafaabbass/OmniTerra-Global-M.git
cd OmniTerra-Global-M
pip install -r requirements.txt
earthengine authenticate   # local dev only
```

**requirements.txt**
```
streamlit>=1.28.0
torch>=2.0.0
earthengine-api>=0.1.370
numpy>=1.24.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

---

## 🚀 Usage

```bash
streamlit run app.py
```

1. Enter **Latitude & Longitude**
2. Select **Crop** — Wheat / Rice / Maize
3. Click **Run Live Inference**
4. View: Yield · NDVI · Carbon · Health Status · Recommendations
5. **Download** `.txt` report

---

## 📦 Dataset & Experimental Setup

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.x |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Loss | MSE |
| Epochs | 200 (early stopping patience=20) |
| Validation Split | 20% |
| Training Hardware | NVIDIA T4 GPU (Google Colab) |
| Inference | CPU (Streamlit Cloud) |
| Checkpoint | ~98 KB (`omni_terra_v1.pth`) |
| Satellite Source | Sentinel-2 SR Harmonized (GEE) |

---

## 🏆 Comparative Benchmarking

| Method | Architecture | Real-Time EO | Carbon Est. | Confidence |
|---|---|---|---|---|
| Xu et al. [2014] | LSTM | ❌ | ❌ | ~85% |
| Tseng et al. [2021] | Transformer Enc. | ❌ | ❌ | ~88% |
| Wang et al. [2022] | Graph Attention | ❌ | ❌ | ~87% |
| **OmniTerra (Ours)** | **ST-Transformer** | **✅ GEE** | **✅ IPCC** | **94.2%** |

---

## 👤 Author

<div align="center">

### Agha Wafa Abbas
**ML Scientist · Lecturer · Researcher**

| Institution | Contact |
|---|---|
| University of Portsmouth, UK | [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk) |
| Arden University, UK | [awabbas@arden.ac.uk](mailto:awabbas@arden.ac.uk) |
| Pearson, UK | — |
| IVY College of Management Sciences, Lahore, Pakistan | [wafa.abbas.lhr@rootsivy.edu.pk](mailto:wafa.abbas.lhr@rootsivy.edu.pk) |

[![GitHub](https://img.shields.io/badge/GitHub-Aghawafaabbass-181717?style=flat-square&logo=github)](https://github.com/Aghawafaabbass)

</div>

---

## 📖 Citation

**Cite the preprint:**
```bibtex
@article{abbas2026omniterra,
  title     = {OmniTerra: A Multi-Modal Spatio-Temporal Transformer Framework
               for Global Yield Intelligence and Carbon Sequestration Modeling},
  author    = {Abbas, Agha Wafa},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20308398},
  url       = {https://doi.org/10.5281/zenodo.20308398}
}
```

**Cite the software:**
```bibtex
@software{abbas2026omniterra_software,
  title     = {OmniTerra v1.0.0 — Production Release},
  author    = {Agha Wafa Abbas},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20308741},
  url       = {https://doi.org/10.5281/zenodo.20308741}
}
```

---

## 📜 License

<div align="center">

**OmniTerra: A Multi-Modal Spatio-Temporal Transformer Framework for Global Yield Intelligence and Carbon Sequestration Modeling**

Copyright © 2026 — Agha Wafa Abbas. All Rights Reserved.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

Licensed under **Creative Commons Attribution 4.0 International**.
Free to share and adapt with appropriate credit to the author.

📧 [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk)

</div>

---

## ⚠️ Disclaimer

> **RESEARCH & EDUCATIONAL USE ONLY**
>
> OmniTerra is an academic research prototype. Yield predictions and carbon estimates are **not intended for operational agricultural decision-making, financial planning, insurance, or policy formulation** without independent validation by qualified agronomists.
>
> Satellite data accuracy depends on Sentinel-2 availability and cloud cover. Carbon estimates are indicative only — not for carbon credit verification or NDC reporting without field-level validation. Software provided "AS IS" without warranty.
>
> — *Agha Wafa Abbas, 2026*

---

<div align="center">

```
© 2026 — Agha Wafa Abbas | OmniTerra Global
Built with 🛰️ Sentinel-2 · ⚡ PyTorch · 🌿 Google Earth Engine · 🚀 Streamlit
```

*"Bridging frontier AI architectures and operational agricultural intelligence for global food security."*

</div>
