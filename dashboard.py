import streamlit as st
import pandas as pd
import os
import subprocess
import pickle

st.set_page_config(
    page_title="Sign Language Translator Dashboard",
    page_icon="🖖",
    layout="wide"
)

st.title("Sign Language Translator Dashboard 🖖")
st.markdown("---")

# File Paths
DATA_FILE = 'gestures.csv'
MODEL_FILE = 'model.pkl'
ENCODER_FILE = 'label_encoder.pkl'

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dataset Analytics", "⚙️ Model Controls", "🎯 Live Inference"])

# --- Helper Functions ---
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
             return pickle.load(f)
    return None

# --- Page 1: Dataset Analytics ---
if page == "📊 Dataset Analytics":
    st.header("📊 Dataset Analytics")
    df = load_data()

    if df is not None and not df.empty:
        # Metrics
        total_samples = len(df)
        unique_classes = df['label'].unique()
        num_classes = len(unique_classes)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Samples Collected", value=total_samples)
        with col2:
            st.metric(label="Unique Gestures/Classes", value=num_classes)

        st.markdown("### 📈 Data Distribution")
        # Class distribution
        class_counts = df['label'].value_counts()
        st.bar_chart(class_counts)

        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(20))
    else:
        st.warning("No data found. Please run `data_collection.py` first to collect some gestures!")
        st.info("Run in terminal: `python data_collection.py --label YOUR_LABEL`")

# --- Page 2: Model Controls ---
elif page == "⚙️ Model Controls":
    st.header("⚙️ Model Training & Info")
    model = load_model()

    if model is not None:
        st.success("Model is Trained & Loaded!")
        
        # Display model attributes
        expected_features = getattr(model, 'n_features_in_', 'Unknown')
        st.write(f"**Expected Features**: {expected_features}")
        if expected_features == 64:
             st.info("💡 Model supports Handedness detection (Left/Right feature).")
        
        if hasattr(model, 'classes_'):
             st.write("**Classes Model was trained on**:")
             # We need label encoder to map back correctly, but just unique y can help
             st.write(model.classes_)

    else:
        st.warning("No trained model found (`model.pkl` is missing).")

    st.markdown("---")
    st.markdown("### 🧠 Re-Train Model")
    st.write("Click below to retrain the classification model on the current `gestures.csv` data.")
    
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training Model... Please wait..."):
            try:
                # Run train_model.py in subprocess
                result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Model trained successfully!")
                    st.code(result.stdout)
                    st.rerun() # Refresh page to load new model
                else:
                    st.error("Training Failed.")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error executing training: {e}")

# --- Page 3: Live Inference ---
elif page == "🎯 Live Inference":
    st.header("🎯 Live Inference")
    st.write("Since Streamlit runs in a browser server loop, the best way to open the transparent real-time camera view is launching our high-performance popup locally.")
    
    st.markdown("### 🎥 Open Webcam Interface")
    st.write("Clicking the button below will launch the live overlay translator on your screen.")
    
    if st.button("🔴 Start Live Translator"):
        try:
             # run in background so it doesn't freeze streamlit
             subprocess.Popen(['python', 'inference.py'])
             st.success("Webcam Feed Opened! Watch your screen for the OpenCV window popup.")
        except Exception as e:
             st.error(f"Failed to launch: {e}")

    st.markdown("---")
    st.info("Instructions: Press **'q'** inside the webcam window popup to close it when done.")
