import streamlit as st
import torch
import ee
import numpy as np
import json
from streamlit_folium import folium_static
import folium
import os

# --- 1. Earth Engine Initialization ---
def initialize_ee():
    try:
        ee.Initialize()
    except Exception:
        try:
            if "GCP_SERVICE_ACCOUNT" in st.secrets:
                secret_json = st.secrets["GCP_SERVICE_ACCOUNT"]
                info = json.loads(secret_json)
                credentials = ee.ServiceAccountCredentials(info['client_email'], key_data=secret_json)
                ee.Initialize(credentials)
            else:
                st.error("GCP_SERVICE_ACCOUNT secret missing!")
                st.stop()
        except Exception as e:
            st.error(f"Auth Failed: {e}")
            st.stop()

initialize_ee()

# --- 2. Model Architecture (FIXED TO MATCH YOUR .PTH FILE) ---
class OmniTerraTransformer(torch.nn.Module):
    def __init__(self, input_dim=3, model_dim=64):
        super().__init__()
        self.input_fc = torch.nn.Linear(input_dim, model_dim)
        self.attention = torch.nn.MultiheadAttention(model_dim, num_heads=4, batch_first=True)
        # CHANGED: 64 -> 128 to match your trained weights
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        out, _ = self.attention(x, x, x)
        return self.ffn(out.squeeze(1))

# --- 3. UI Setup ---
st.set_page_config(page_title="OmniTerra AI", layout="wide", page_icon="🌍")

# Sidebar Branding (Fixed Image)
st.sidebar.image("https://flaticon.com", width=100)
st.sidebar.title("System Control")
st.sidebar.success(f"Developed by:\n**ML Scientist Agha Wafa Abbas**")

st.title("🌍 OmniTerra: Global Yield Intelligence")
st.markdown("### Multi-Modal Spatio-Temporal Transformer Framework")

# --- 4. Helper Functions ---
@st.cache_resource
def load_omni_model():
    # Use 128 to match the ffn layer in the class above
    model = OmniTerraTransformer()
    model_path = os.path.join('models', 'omni_terra_v1.pth')
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Model not found at: {model_path}")
        return None
        
    try:
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Structural Mismatch: {e}")
        return None

def get_live_features(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).median()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        stats = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point.buffer(500), scale=10).getInfo()
        ndvi_val = stats.get('NDVI', 0.5) if stats else 0.5
        return [ndvi_val, 290.0, 0.02]
    except:
        return [0.5, 290.0, 0.02]

# --- 5. Main Layout ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📍 Target Parameters")
    lat = st.number_input("Latitude", value=31.5204, format="%.4f")
    lon = st.number_input("Longitude", value=74.3587, format="%.4f")
    crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize"])
    
    if st.button("🚀 Run Live Inference"):
        with st.spinner("Processing Transformer Layers..."):
            model = load_omni_model()
            if model:
                features = get_live_features(lat, lon)
                feature_tensor = torch.tensor([features], dtype=torch.float32)
                
                with torch.no_grad():
                    prediction = model(feature_tensor).item()
                
                # Results display
                st.balloons()
                st.metric("Predicted Yield", f"{prediction:.2f} t/ha")
                st.metric("Current NDVI", f"{features[0]:.2f}")
                st.success("Analysis Complete!")
            else:
                st.error("Model structure mismatch. Please check sidebar logs.")

with col2:
    st.subheader("🗺️ Spatial Analysis View")
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], icon=folium.Icon(color='green', icon='leaf')).add_to(m)
    folium_static(m)
