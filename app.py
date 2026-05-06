import streamlit as st
import torch
import ee
import numpy as np
import json

# 1. Initialize Earth Engine with Service Account
def initialize_ee():
    if not ee.data._is_initialized:
        try:
            # Fetch secret from Streamlit Cloud
            secret_json = st.secrets["GCP_SERVICE_ACCOUNT"]
            info = json.loads(secret_json)
            
            # Authenticate using the service account info
            credentials = ee.ServiceAccountCredentials(info['client_email'], key_data=secret_json)
            ee.Initialize(credentials)
        except Exception as e:
            st.error(f"Earth Engine failed to initialize: {e}")
            st.stop()

initialize_ee()

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

# 3. Load Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_model():
    model = OmniTerraTransformer()
    try:
        model.load_state_dict(torch.load('models/omni_terra_v1.pth', map_location='cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file not found in 'models/omni_terra_v1.pth'")
        return None

# 4. Real-Time Feature Extractor (Cached for performance)
@st.cache_data(ttl=3600)
def get_live_features(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        # Fetch latest Sentinel-2 Median Image
        img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).median()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Get values
        stats = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point.buffer(500), scale=10).getInfo()
        ndvi_val = stats.get('NDVI', 0.5) 
        
        # Features: [NDVI, Temp Placeholder, Precip Placeholder]
        return [ndvi_val, 290.0, 0.02]
    except Exception as e:
        st.error(f"Error fetching satellite data: {e}")
        return [0.5, 290.0, 0.02]

# 5. Streamlit UI
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
            model = load_model()
            
            if model:
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
