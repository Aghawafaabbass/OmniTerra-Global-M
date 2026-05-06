import streamlit as st
import torch
import ee
import numpy as np

# 1. Initialize Earth Engine (Streamlit Cloud uses Secrets for this)
# For local/Colab testing, ensure you are authenticated
try:
    ee.Initialize()
except Exception as e:
    st.error("Earth Engine not initialized. Please authenticate.")

# 2. Model Architecture (Must match your Training)
class OmniTerraTransformer(torch.nn.Module):
    def __init__(self, input_dim=3, model_dim=64):
        super().__init__()
        self.input_fc = torch.nn.Linear(input_dim, model_dim)
        self.attention = torch.nn.MultiheadAttention(model_dim, num_heads=4, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        out, _ = self.attention(x, x, x)
        return self.ffn(out.squeeze(1))

# 3. Real-Time Feature Extractor
def get_live_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    # Fetch latest Sentinel-2 Median Image
    img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).median()
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Get values
    stats = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point.buffer(500), scale=10).getInfo()
    ndvi_val = stats.get('NDVI', 0.5) # Default if missing
    
    # Static placeholders for Weather/Precip (Can be linked to Weather API later)
    return [ndvi_val, 290.0, 0.02] 

# 4. Streamlit UI
st.set_page_config(page_title="OmniTerra AI", layout="wide")
st.title("🌍 OmniTerra: Global Yield Intelligence")
st.markdown("### Multi-Modal Spatio-Temporal Transformer Framework")

col1, col2 = st.columns(2)

with col1:
    st.header("📍 Target Location")
    lat = st.number_input("Latitude", value=52.13, format="%.4f")
    lon = st.number_input("Longitude", value=-106.67, format="%.4f")
    
    if st.button("🚀 Run Live Inference"):
        with st.spinner("Fetching Satellite Data & Running Transformer..."):
            # A. Get Data
            features = get_live_features(lat, lon)
            feature_tensor = torch.tensor([features], dtype=torch.float32)
            
            # B. Load Model
            model = OmniTerraTransformer()
            model.load_state_dict(torch.load('models/omni_terra_v1.pth', map_location='cpu'))
            model.eval()
            
            # C. Predict
            with torch.no_grad():
                prediction = model(feature_tensor).item()
            
            st.metric("Predicted Wheat Yield", f"{prediction:.2f} t/ha")
            st.success("Analysis Complete!")

with col2:
    st.header("📊 System Insights")
    st.write("Using Sentinel-2 Multi-Spectral Bands (B8, B4)")
    st.write("Architecture: Multi-Head Self-Attention")
    st.info("This model adapts dynamically to global vegetation patterns.")
