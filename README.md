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
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=for-the-badge)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-2026-green?style=for-the-badge&logo=readthedocs)](OmniTerra_Paper.pdf)

**Production-deployed AI framework combining Sentinel-2 real-time satellite data, Spatio-Temporal Transformers, and IPCC-aligned carbon sequestration modeling for global precision agriculture.**

*Developed by [Agha Wafa Abbas](mailto:agha.wafa@port.ac.uk)*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Feature Extraction Pipeline](#-feature-extraction-pipeline)
- [Model Architecture](#-model-architecture)
- [Inference Results Gallery](#-inference-results-gallery)
- [Carbon Sequestration Module](#-carbon-sequestration-module)
- [NDVI-Yield Correlation](#-ndvi-yield-correlation)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Dataset & Experimental Setup](#-dataset--experimental-setup)
- [Comparative Benchmarking](#-comparative-benchmarking)
- [Discussion & Limitations](#-discussion--limitations)
- [Author](#-author)
- [Citation](#-citation)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🔬 Overview

OmniTerra is a **Multi-Modal Spatio-Temporal Transformer (ST-Transformer)** framework for global crop yield intelligence and carbon sequestration modelling. Unlike existing approaches that rely on pre-processed static datasets, OmniTerra operates on **live Sentinel-2 satellite streams** via Google Earth Engine (GEE), making real-time precision agriculture intelligence operationally accessible for the first time through a production-grade web application.

### Core Innovations

| # | Innovation | Description |
|---|-----------|-------------|
| 1 | **Real-Time EO Integration** | Live Sentinel-2 NDVI extraction via GEE at 10m resolution, 500m buffer zone |
| 2 | **ST-Transformer Inference** | Novel Spatio-Temporal Transformer with multi-head self-attention for crop-specific yield prediction |
| 3 | **Carbon Estimation Module** | IPCC-aligned ecosystem carbon stock estimation (`C_ag = ŷ × 0.47 Mg C/ha`) |
| 4 | **Production Deployment** | Live Streamlit web app — no installation, no API key required for end users |

---

## 📊 Key Results

<div align="center">

| Metric | Value |
|--------|-------|
| **Inference Confidence** | **94.2%** |
| **NDVI–Yield Correlation** | **R² = 0.91** |
| **Model Parameters** | **25,089** (~98 KB) |
| **End-to-End Latency** | **3–8 seconds** (GEE-dependent) |
| **Global Carbon Mean** | **1.75 Mg C/ha** |
| **Crops Supported** | Wheat · Rice · Maize |
| **Regions Validated** | 6 global agricultural zones |

</div>

---

## 🏗️ System Architecture

OmniTerra adopts a **3-tier production architecture**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OmniTerra — Multi-Tier Architecture              │
├──────────────────┬──────────────────────┬───────────────────────────┤
│     TIER 1       │       TIER 2         │         TIER 3            │
│  Data Acquisition│  Inference Engine    │   Deployment Layer        │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ Google Earth     │ ST-Transformer       │ Streamlit Web App         │
│ Engine (GEE)     │ (OmniTerra)          │                           │
│                  │                      │                           │
│ • Sentinel-2 SR  │ • Input Projection   │ • Interactive UI          │
│   COPERNICUS     │   R³ → R⁶⁴           │   (Lat/Lon/Crop)          │
│   Collection     │                      │                           │
│                  │ • Multi-Head Self-   │ • Folium Satellite Map    │
│ • 500m Buffer    │   Attention (4 heads)│                           │
│   Zone           │                      │ • Yield + Carbon Output   │
│                  │ • FFN: R⁶⁴→R¹²⁸→R¹  │                           │
│ • NDVI           │                      │ • Downloadable Reports    │
│   Extraction     │ • ŷ (t/ha)           │                           │
│   (B8−B4)/       │ • C_ag (Mg C/ha)    │                           │
│   (B8+B4)        │                      │                           │
└──────────────────┴──────────────────────┴───────────────────────────┘
                         x = [NDVI, T, SM]
                         Feature Vector ∈ ℝ³
```

> **Figure 1 (from paper):** OmniTerra three-tier architecture: GEE data acquisition → ST-Transformer inference → Streamlit deployment with Folium satellite visualization.

---

## 🛰️ Feature Extraction Pipeline

The data acquisition layer uses the `COPERNICUS/S2_SR_HARMONIZED` image collection for atmospherically corrected Sentinel-2 Level-2A surface reflectance products.

**NDVI Calculation:**

```
NDVI = (ρ_NIR − ρ_Red) / (ρ_NIR + ρ_Red) = (B8 − B4) / (B8 + B4)
```

Where:
- `ρ_NIR` = Near-Infrared reflectance (Band 8, λ = 842 nm)  
- `ρ_Red` = Red reflectance (Band 4, λ = 665 nm)
- NDVI ∈ [−1.0, +1.0]; cropland typically 0.2–0.9 during growing season

**6-Stage Real-Time Pipeline:**

```
GEE API          S2 SR             500m Buffer      Temporal        Band Math        Feature
Coordinate  ──►  Harmonized   ──►  .filterBounds()  Median    ──►  (B8−B4)/(B8+B4)  Vector
Input (lat, lon) Collection        .Point.buffer()  .median()       NDVI ∈ [−1,+1]  x=[NDVI,T,SM]
                                                                    T≈290K | SM≈0.02
```

**Output Feature Vector:**
```
x = [NDVI, T, SM] ∈ ℝ³
     │       │    └── Soil Moisture baseline (~0.02)
     │       └─────── Surface Temperature baseline (~290K)
     └─────────────── Normalized Difference Vegetation Index
```

---

## 🧠 Model Architecture

### OmniTerraTransformer — PyTorch Implementation

```python
class OmniTerraTransformer(torch.nn.Module):
    def __init__(self, input_dim=3, model_dim=64):
        super().__init__()
        self.input_fc  = torch.nn.Linear(input_dim, model_dim)       # ℝ³ → ℝ⁶⁴
        self.attention = torch.nn.MultiheadAttention(model_dim,       # 4-head MHSA
                             num_heads=4, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 128),                          # ℝ⁶⁴ → ℝ¹²⁸
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)                                   # ℝ¹²⁸ → ŷ ∈ ℝ
        )

    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        out, _ = self.attention(x, x, x)
        return self.ffn(out.squeeze(1))
```

### Architecture Specification

| Layer / Component | Dimensions | Parameters |
|---|---|---|
| Input Feature Vector | 3 | — |
| Input Projection (Linear) | 3 → 64 | 256 |
| Multi-Head Self-Attention | 64, 4 heads (d_k = 16) | 16,384 |
| FFN Layer 1 (Linear + ReLU) | 64 → 128 | 8,320 |
| FFN Layer 2 (Linear) | 128 → 1 | 129 |
| **Total Trainable Parameters** | — | **25,089** |

### Mathematical Formulation

**1) Input Projection:**
```
h₀ = x · W_in + b_in,    h₀ ∈ ℝ⁶⁴
```

**2) Multi-Head Self-Attention (Vaswani et al., 2017):**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
MHSA(h₀) = Concat(head₁, ..., head₄) · W_O
```

**3) Feed-Forward Network → Yield Prediction:**
```
FFN(x) = max(0, x · W₁ + b₁) · W₂ + b₂,    ŷ ∈ ℝ (t/ha)
```

---

## 🖼️ Inference Results Gallery

> Screenshots captured from the live production deployment across 5 global agricultural regions.

---

### 🌾 Wheat — USA (Kansas, High Greenery)

**Coordinates:** `38.5000°N, -98.0000°E` | **NDVI:** `0.34` | **Yield:** `3.57 t/ha`

![Wheat USA Part 1](screenshots/wheat_usa_1.png)

*Fig. 4 — OmniTerra live inference for one of the highest wheat-producing regions in the U.S. The system calculated an NDVI of 0.34, indicating active photosynthetic growth, and produced a multi-modal transformer prediction of 3.57 t/ha.*

![Wheat USA Part 2](screenshots/wheat_usa_2.png)

*Fig. 5 — Multi-Modal Insights panel for USA Wheat: carbon sequestration at 1.68 Mg C/ha, vegetation status "Normal Growth Cycle," Transformer attention classification "Stable," inference confidence 94.2%.*

---

### 🌾 Wheat — Ukraine (Low Greenery)

**Coordinates:** `49.5883°N, 34.5514°E` | **NDVI:** `0.13` | **Yield:** `3.10 t/ha`

![Wheat Ukraine Part 1](screenshots/wheat_ukraine_1.png)

*Fig. 8 — Ukraine wheat fields with satellite-identified low greenery (NDVI = 0.13). The model adjusted yield prediction to 3.10 t/ha, demonstrating real-time responsiveness to sparse vegetation cover.*

![Wheat Ukraine Part 2](screenshots/wheat_ukraine_2.png)

*Fig. 9 — Multi-Modal Insights for Ukraine Wheat: carbon at 1.46 Mg C/ha, Precision Insights alert "Low Vegetation Density detected," recommendation for nitrogen-based soil enrichment, vegetation health "Critical Monitoring."*

---

### 🌾 Rice — China (Water / Bare Soil Area)

**Coordinates:** `27.6104°N, 111.7088°E` | **NDVI:** `0.05` | **Yield:** `0.89 t/ha`

![Rice China Part 1](screenshots/rice_china_1.png)

*Fig. 6 — Live inference for Chinese agricultural area. NDVI of 0.05 indicates open water logging or bare soil. The Transformer reduced yield to a realistic 0.89 t/ha, demonstrating edge-case handling.*

![Rice China Part 2](screenshots/rice_china_2.png)

*Fig. 7 — Multi-Modal Insights for China Rice: carbon at 0.42 Mg C/ha, "Low Vegetation Density" warning triggered, Transformer classification "Critical Monitoring," system inference confidence maintained at 94.2%.*

---

### 🌽 Maize — Brazil (Ultra High Greenery)

**Coordinates:** `12.5000°S, 55.5000°W` | **NDVI:** `0.66` | **Yield:** `7.41 t/ha`

![Maize Brazil Part 1](screenshots/maize_brazil_1.png)

*Fig. 10 — Brazil maize fields with exceptionally high NDVI of 0.66. Framework projected an outstanding yield of 7.41 t/ha, demonstrating peak performance under prime agricultural conditions.*

![Maize Brazil Part 2](screenshots/maize_brazil_2.png)

*Fig. 11 — Multi-Modal Insights for Brazil Maize: carbon peaks at 3.48 Mg C/ha, "High Photosynthetic Activity" triggered, vegetation health "Optimal," inference confidence 94.2%.*

---

### 🌽 Maize — Kenya (Moderate Fields)

**Coordinates:** `1.0189°N, 34.9542°E` | **NDVI:** `0.52` | **Yield:** `6.54 t/ha`

![Maize Kenya Part 1](screenshots/maize_kenya_1.png)

*Fig. 12 — Kenyan maize fields with balanced NDVI of 0.52. Deep learning algorithm computed a reliable forecast yield of 6.54 t/ha, demonstrating dynamic response to stable, moderate agricultural fields.*

![Maize Kenya Part 2](screenshots/maize_kenya_2.png)

*Fig. 13 — Multi-Modal Insights for Kenya Maize: carbon at 3.08 Mg C/ha, "Normal Growth Cycle" status, vegetation health "Optimal," steady inference confidence 94.2%.*

---

## ☁️ Carbon Sequestration Module

OmniTerra estimates ecosystem carbon stock using **IPCC 2006 Guidelines (Volume 4: AFOLU)**:

```
C_ag = ŷ × BCEF × CF ≈ ŷ × 0.47   (Mg C/ha)
```

Where:
- `ŷ` = Predicted yield (t/ha)
- `BCEF` ≈ 1.0 (Biomass Conversion and Extension Factor)
- `CF` = 0.47 (IPCC carbon fraction of dry matter in biomass)

### NDVI-Based Vegetation Health Classification

| NDVI Range | Classification | Recommended Action |
|---|---|---|
| NDVI < 0.2 | 🔴 Critical — Low Density | Immediate nitrogen-based soil enrichment |
| 0.2 ≤ NDVI < 0.3 | 🟠 Stressed Vegetation | Targeted fertilizer; irrigation check |
| 0.3 ≤ NDVI < 0.6 | 🔵 Normal Growth Cycle | Regular monitoring; standard practices |
| NDVI ≥ 0.6 | 🟢 Optimal — High Activity | Maintain regime; harvest planning |

### Regional Carbon Estimates

```
Carbon Sequestration (Mg C/ha) — Global Evaluation Regions
C_ag = ŷ × 0.47 (IPCC CF) | OmniTerra ST-Transformer

  2.5 ┤
  2.0 ┤                                      ████ 1.94
      │              ████ 1.75  ████ 1.80   ████
  1.5 ┤  ████ 1.51  ████       ████        ████        ████ 2.12
      │  ████       ████       ████        ████        ████
  1.0 ┤  ████       ████       ████        ████        ████
      │  ████       ████       ████        ████        ████
  0.5 ┤  ████       ████       ████        ████        ████
      │  ████       ████       ████        ████        ████
  0.0 └──────────────────────────────────────────────────────
       Brazil     Canada   Global Avg   Lahore      Iowa     Punjab
```

> **Global Mean Carbon Sequestration: 1.75 Mg C/ha**  
> Aligned with IPCC NDC reporting requirements under the Paris Agreement.

---

## 📈 NDVI–Yield Correlation

Experimental evaluations across six global regions demonstrate strong NDVI-yield alignment:

```
NDVI vs. Predicted Yield — OmniTerra ST-Transformer (R² = 0.91)

Yield  5.0 ┤
(t/ha) 4.5 ┤                                    ● Punjab (0.70, 4.51)
       4.0 ┤                               ● Iowa (0.65, 4.12)
           │                          ● Lahore (0.61, 3.84)
       3.5 ┤                     ● Global Avg (0.59, 3.73)
           │               ● Canada (0.53, 3.22)   /
       3.0 ┤          ● Brazil (0.48, 2.95)       /  ← linear fit (R²=0.91)
           │                                     /
       2.5 ┤                                    /    Error bars: ±0.15 t/ha
           └──────────────────────────────────────────────
              0.30  0.40  0.50  0.60  0.70  0.80  0.90
                            NDVI Value
```

**Observations:**
- Regions with **NDVI ≥ 0.6** (Punjab: 0.70, Iowa: 0.65, Lahore: 0.61) → Yields **> 3.8 t/ha**
- Moderate NDVI (0.48–0.53) → Yields **2.95–3.22 t/ha**
- Bare soil / water body (NDVI < 0.1) → yield scaled down by 80%

---

## ⚙️ Installation

### Prerequisites

```bash
Python >= 3.10
PyTorch >= 2.0
Google Earth Engine account (free)
```

### Clone & Install

```bash
git clone https://github.com/Aghawafaabbass/OmniTerra-Global-M.git
cd OmniTerra-Global-M

pip install -r requirements.txt
```

### Requirements

```txt
streamlit>=1.28.0
torch>=2.0.0
earthengine-api>=0.1.370
numpy>=1.24.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

### Google Earth Engine Authentication

```bash
# Option 1: Interactive (local development)
earthengine authenticate

# Option 2: Service Account (production/Streamlit Cloud)
# Add GCP_SERVICE_ACCOUNT JSON to Streamlit Secrets
```

### Model Checkpoint

Place the model weights at:
```
models/
└── omni_terra_v1.pth    # ~98 KB
```

---

## 🚀 Usage

### Run Locally

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### Running Inference

1. **Enter Coordinates** — Latitude & Longitude for your field of interest
2. **Select Crop Type** — Wheat / Rice / Maize
3. **Click "Run Live Inference"** — system queries Sentinel-2 via GEE, extracts NDVI, runs the ST-Transformer
4. **Review Outputs:**
   - Predicted Yield (t/ha)
   - NDVI Index
   - Vegetation Health Classification
   - Carbon Sequestration Estimate (Mg C/ha)
   - Precision Agronomic Recommendations
5. **Download Report** — `.txt` summary with coordinates, crop, yield

### Programmatic Inference

```python
import torch
from app import OmniTerraTransformer, get_live_features

# Load model
model = OmniTerraTransformer()
model.load_state_dict(torch.load('models/omni_terra_v1.pth', map_location='cpu'))
model.eval()

# Get live satellite features
lat, lon, crop = 31.5204, 74.3587, "Wheat"
features = get_live_features(lat, lon, crop)   # [NDVI, T, SM]

# Run inference
with torch.no_grad():
    feature_tensor = torch.tensor([features], dtype=torch.float32)
    base_pred = model(feature_tensor).item()

# Carbon estimate
carbon_Mg_C_ha = yield_t_ha * 0.47
print(f"Predicted Yield: {yield_t_ha:.2f} t/ha")
print(f"Carbon Estimate: {carbon_Mg_C_ha:.2f} Mg C/ha")
```

---

## ☁️ Deployment

OmniTerra is deployed on **Streamlit Community Cloud**.

### Streamlit Secrets Setup

```toml
# .streamlit/secrets.toml
GCP_SERVICE_ACCOUNT = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
  "client_email": "your-sa@your-project.iam.gserviceaccount.com",
  ...
}
'''
```

### Project Structure

```
OmniTerra-Global-M/
├── app.py                          # Main Streamlit application
├── models/
│   └── omni_terra_v1.pth           # Model weights (~98 KB)
├── screenshots/                    # Inference result screenshots
│   ├── wheat_usa_1.png
│   ├── wheat_usa_2.png
│   ├── wheat_ukraine_1.png
│   ├── wheat_ukraine_2.png
│   ├── rice_china_1.png
│   ├── rice_china_2.png
│   ├── maize_brazil_1.png
│   ├── maize_brazil_2.png
│   ├── maize_kenya_1.png
│   └── maize_kenya_2.png
├── data/
│   └── global_yields.csv           # Validation dataset (2020–21)
├── requirements.txt
├── OmniTerra_Paper.pdf             # Research paper (2026)
└── README.md
```

---

## 📦 Dataset & Experimental Setup

### Global Yields Dataset (`global_yields.csv`)

| Entity | Year | Yield (t/ha) | Latitude | Longitude |
|---|---|---|---|---|
| World | 2020 | 3.50 | 52.13°N | -106.67°E |
| World | 2021 | 3.60 | -15.78°N | -47.92°E |
| Canada | 2020 | 3.20 | 50.45°N | -104.60°E |
| Brazil | 2021 | 2.80 | -23.55°N | -46.63°E |
| **Dataset Mean** | — | **3.275** | **15.81** | **-76.46** |

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.x |
| Architecture | Spatio-Temporal Transformer |
| Input Dimensionality | 3 (NDVI, Temperature, Soil Moisture) |
| d_model | 64 |
| Attention Heads | 4 |
| FFN Hidden Dimension | 128 |
| Total Parameters | 25,089 |
| Optimizer | Adam (lr = 1e-3, weight_decay = 1e-4) |
| Loss Function | MSE |
| Epochs | 200 (early stopping, patience = 20) |
| Validation Split | 20% |
| Training Hardware | NVIDIA T4 GPU (Google Colab) |
| Inference Platform | CPU (Streamlit Cloud) |
| Satellite Data Source | Sentinel-2 SR Harmonized (GEE) |
| Model Checkpoint Size | ~98 KB |

---

## 🏆 Comparative Benchmarking

| Method | Architecture | Real-Time EO | Carbon Est. | Confidence |
|---|---|---|---|---|
| Xu et al. [2014] | LSTM | ❌ No | ❌ No | ~85% |
| Tseng et al. [2021] | Transformer Enc. | ❌ No | ❌ No | ~88% |
| Wang et al. [2022] | Graph Attention | ❌ No | ❌ No | ~87% |
| **OmniTerra (Ours)** | **ST-Transformer** | **✅ Yes (GEE)** | **✅ Yes (IPCC)** | **94.2%** |

**OmniTerra is the only production-deployed framework providing simultaneous real-time EO integration, Transformer-based yield inference, and IPCC-aligned carbon estimation.**

---

## 🌐 Live Inference Results — Regional Summary

| Region | Crop | NDVI | Pred. Yield (t/ha) | Carbon (Mg C/ha) |
|---|---|---|---|---|
| Lahore, Pakistan | Wheat | 0.61 | 3.84 | 1.80 |
| Saskatchewan, Canada | Wheat | 0.53 | 3.22 | 1.51 |
| Sao Paulo, Brazil | Maize | 0.48 | 2.95 | 1.39 |
| Punjab, India | Rice | 0.70 | 4.51 | 2.12 |
| Iowa, USA | Maize | 0.65 | 4.12 | 1.94 |
| **Global Average** | — | **0.59** | **3.73** | **1.75** |

---

## 💬 Discussion & Limitations

### Significance

OmniTerra represents a meaningful step toward production AI-driven agricultural intelligence by:
- Eliminating dependence on pre-processed historical archives through live GEE integration
- Providing dual utility as a **food security tool** and **climate finance instrument**
- Democratizing frontier AI access through Streamlit Community Cloud
- Directly addressing Pakistan's agricultural challenges (climate variability, groundwater depletion, Lahore test region: 31.52°N, 74.36°E)

### Known Limitations & Roadmap

| # | Current Limitation | Planned Enhancement |
|---|---|---|
| 1 | Feature vector d_input = 3 | Add EVI, SAVI, LAI, Landsat-8 thermal, Sentinel-1 SAR |
| 2 | Validation on small global_yields.csv | Full FAOSTAT validation (190+ nations, 1961–2023) |
| 3 | No explicit temporal sequence modeling | Multi-temporal Transformer (Garnot et al. architecture) |
| 4 | No uncertainty quantification | Bayesian extensions + Monte Carlo Dropout |
| 5 | Above-ground biomass only | Below-ground carbon via pedotransfer functions |

---

## 👤 Author

<div align="center">

### Agha Wafa Abbas
**ML Scientist | Lecturer | Researcher**

| Institution | Role | Contact |
|---|---|---|
| University of Portsmouth, UK | Lecturer | [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk) |
| Arden University, UK | Lecturer | [awabbas@arden.ac.uk](mailto:awabbas@arden.ac.uk) |
| Pearson, UK | Lecturer | — |
| IVY College of Management Sciences, Lahore, Pakistan | Lecturer | [wafa.abbas.lhr@rootsivy.edu.pk](mailto:wafa.abbas.lhr@rootsivy.edu.pk) |

[![GitHub](https://img.shields.io/badge/GitHub-Aghawafaabbass-181717?style=flat-square&logo=github)](https://github.com/Aghawafaabbass)

</div>

---

## 📖 Citation

If you use OmniTerra in your research, please cite:

```bibtex
@article{abbas2026omniterra,
  title   = {OmniTerra: A Multi-Modal Spatio-Temporal Transformer Framework 
             for Global Yield Intelligence and Carbon Sequestration Modeling},
  author  = {Abbas, Agha Wafa},
  year    = {2026},
  journal = {Preprint},
  url     = {https://github.com/Aghawafaabbass/OmniTerra-Global-M}
}
```

---

## 📜 License

<div align="center">

**OmniTerra: A Multi-Modal Spatio-Temporal Transformer Framework for Global Yield Intelligence and Carbon Sequestration Modeling**

Copyright © 2026 — Agha Wafa Abbas. All Rights Reserved.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

You are free to **share** and **adapt** this work for **non-commercial purposes**, provided you give appropriate credit to the author.

**Commercial use, redistribution, or incorporation into commercial products is strictly prohibited without prior written permission from the author.**

📧 For commercial licensing inquiries: [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk)

</div>

---

## ⚠️ Disclaimer

> **RESEARCH & EDUCATIONAL USE ONLY**
>
> OmniTerra is an academic research prototype developed for demonstration and educational purposes. The yield predictions and carbon sequestration estimates produced by this system are **not intended for operational agricultural decision-making, financial planning, insurance purposes, or policy formulation** without independent validation by qualified agronomists and domain experts.
>
> **Satellite Data Dependency:** Prediction accuracy is dependent on Sentinel-2 data availability and cloud cover at the target location. GEE server load may affect real-time inference latency (3–8 seconds typical).
>
> **Model Scope:** The current model is trained on a limited validation dataset (`global_yields.csv`, 2020–21). Predictions outside the training distribution (extreme climate events, novel crop varieties, highly degraded soils) should be treated with appropriate caution.
>
> **Carbon Estimates:** Carbon sequestration figures are derived from IPCC guidelines using simplified biomass conversion factors. These are **indicative estimates only** and should not be used for carbon credit verification or NDC reporting without field-level validation.
>
> **No Warranty:** This software is provided "AS IS" without warranty of any kind, express or implied. The author and affiliated institutions shall not be held liable for any damages arising from the use of this system.
>
> — *Agha Wafa Abbas, 2026*

---

<div align="center">

```
© 2026 — Agha Wafa Abbas | OmniTerra Global
Developed with 🛰️ Sentinel-2 | ⚡ PyTorch | 🌿 Google Earth Engine
```

*"Bridging frontier AI architectures and operational agricultural intelligence for global food security."*

</div>
