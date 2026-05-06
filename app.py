import streamlit as st
import torch
import ee
import numpy as np
import json

# 1. Initialize Earth Engine using Streamlit Secrets
def initialize_ee():
    """Initializes Earth Engine using the service account key stored in secrets."""
    try:
        # Check if already initialized
        ee.Initialize()
    except Exception:
        try:
            # Ensure the secret exists in the Streamlit dashboard
            if "GCP_SERVICE_ACCOUNT" in st.secrets:
                secret_json = st.secrets["GCP_SERVICE_ACCOUNT"]
                info = json.loads(secret_json)
                
                # Create credentials from the secret JSON
                credentials = ee.ServiceAccountCredentials(
                    info['client_email'], 
                    key_data=secret_json
                )
                ee.Initialize(credentials)
            else:
                st.error("Missing Secret: Please add 'GCP_SERVICE_ACCOUNT' to your Streamlit Secrets.")
                st.stop()
        except Exception as e:
            st.error(f"Earth Engine Authentication Failed: {e}")
            st.stop()

# Run the initialization
initialize_ee()

# 2. Model Architecture (Must match your training script)
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

# 3. Cached Model Loader (Prevents reloading the model on every click)
@st.cache_resource
def load_omni_model():
    model = OmniTerraTransformer()
    try:
        # Load your weights from the models folder
        model.load_state_dict(torch.load('models/omni_terra_v1.pth', map_location='cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'models/omni_terra_v1.pth' not found. Please check your file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# 4. Real-Time Feature Extractor from Google Earth Engine
def get_live_features(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        # Fetch latest Sentinel-2 Median Image
        img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).median()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Calculate mean NDVI for the area
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=point.buffer(500), 
            scale=10
        ).getInfo()
        
        ndvi_val = stats.get('NDVI', 0.5) if stats else 0.5
        
        # Features: [NDVI, Temp (Placeholder), Precip (Placeholder)]
        return [ndvi_val, 290.0, 0.02] 
    except Exception as e:
        st.warning(f"Satellite data fetch warning: {e}")
        return [0.5, 290.0, 0.02] # Return defaults if data fetch fails

# 5. Streamlit User Interface
st.set_page_config(page_title="OmniTerra AI", layout="wide")
st.title("🌍 OmniTerra: Global Yield Intelligence")
st.markdown("### Multi-Modal Spatio-Temporal Transformer Framework")

col1, col2 = st.columns(2)

with col1:
    st.header("📍 Target Location")
    lat = st.number_input("Latitude", value=52.1300, format="%.4f")
    lon = st.number_input("Longitude", value=-106.6700, format="%.4f")
    
    if st.button("🚀 Run Live Inference"):
        with st.spinner("Fetching Satellite Data & Running Transformer..."):
            # A. Extract features from GEE
            features = get_live_features(lat, lon)
            feature_tensor = torch.tensor([features], dtype=torch.float32)
            
            # B. Load the pre-trained model
            model = load_omni_model()
            
            if model:
                # C. Run Inference
                with torch.no_grad():
                    prediction = model(feature_tensor).item()
                
                st.metric("Predicted Wheat Yield", f"{prediction:.2f} t/ha")
                st.success("Analysis Complete!")
            else:
                st.error("Model initialization failed.")

with col2:
    st.header("📊 System Insights")
    st.write("**Data Source:** Sentinel-2 Multi-Spectral Bands (B8, B4)")
    st.write("**Architecture:** Multi-Head Self-Attention Transformer")
    st.info("The system analyzes live vegetation indices (NDVI) to predict crop health and yield.")
