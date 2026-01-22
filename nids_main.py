import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #39568f;
}

.card {
    background-color: #7ba5e0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}

.badge-safe {
    background-color: #1f8b4c;
    padding: 6px 12px;
    border-radius: 8px;
    color: white;
    font-weight: bold;
}

.badge-attack {
    background-color: #b91c1c;
    padding: 6px 12px;
    border-radius: 8px;
    color: white;
    font-weight: bold;
}

.step {
    padding: 8px;
    border-radius: 6px;
    margin: 4px;
    display: inline-block;
    background: #706868;
}

.chat-ai {
    background-color: #0d3b66;
    padding: 15px;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
<h1>🛡️ AI-Based Network Intrusion Detection System</h1>
<p>Random Forest + Groq AI Cybersecurity Analyzer (Student Project)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚙️ Project Settings")

groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.markdown("## 🧠 Model Training")

# ---------------- DATA LOADER ----------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, nrows=15000)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        return None

# ---------------- MODEL TRAINING ----------------
def train_model(df):
    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s'
    ]
    target = 'Label'

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None, 0, [], None, None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    score = accuracy_score(y_test, clf.predict(X_test))

    return clf, score, features, X_test, y_test

# ---------------- LOAD DATA ----------------
df = load_data(DATA_FILE)

if df is None:
    st.error(f"❌ File '{DATA_FILE}' not found. Upload it to the project folder.")
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

# ---------------- TRAIN BUTTON ----------------
if st.sidebar.button("🚀 Train Model Now"):
    with st.spinner("Training model..."):
        clf, accuracy, feature_names, X_test, y_test = train_model(df)

        if clf:
            st.session_state['model'] = clf
            st.session_state['features'] = feature_names
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test

            st.sidebar.success(f"Training Complete! Accuracy: {accuracy:.2%}")

# ---------------- STEPS UI ----------------
st.markdown("""
<div class="card">
<span class="step">1️⃣ Load Data</span>
<span class="step">2️⃣ Train Model</span>
<span class="step">3️⃣ Capture Packet</span>
<span class="step">4️⃣ AI Explanation</span>
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
st.header("📊 Threat Analysis Dashboard")

if 'model' in st.session_state:

    col1, col2 = st.columns(2)

    # ---- SIMULATION ----
    with col1:
        st.markdown("<div class='card'><h3>🎯 Packet Simulation</h3></div>", unsafe_allow_html=True)

        if st.button("🎲 Capture Random Packet"):
            random_idx = np.random.randint(0, len(st.session_state['X_test']))
            packet_data = st.session_state['X_test'].iloc[random_idx]
            actual_label = st.session_state['y_test'].iloc[random_idx]

            st.session_state['current_packet'] = packet_data
            st.session_state['actual_label'] = actual_label

    # ---- PACKET DISPLAY ----
    if 'current_packet' in st.session_state:
        packet = st.session_state['current_packet']

        with col1:
            st.markdown("<div class='card'><h4>📦 Packet Header Info</h4></div>", unsafe_allow_html=True)
            st.dataframe(packet.to_frame(name="Value"), use_container_width=True)

        with col2:
            st.markdown("<div class='card'><h3>🤖 AI Detection Result</h3></div>", unsafe_allow_html=True)

            prediction = st.session_state['model'].predict([packet])[0]

            if prediction == "BENIGN":
                st.markdown("<span class='badge-safe'>✅ SAFE (BENIGN)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='badge-attack'>🚨 ATTACK DETECTED ({prediction})</span>", unsafe_allow_html=True)

            st.caption(f"Ground Truth Label: {st.session_state['actual_label']}")

            st.markdown("---")
            st.subheader("💬 Ask AI Analyst (Groq)")

            if st.button("Generate Explanation"):
                if not groq_api_key:
                    st.warning("Please enter your Groq API key in the sidebar.")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)

                        prompt = f"""
You are a cybersecurity analyst.

A network packet was detected as: {prediction}

Packet Technical Details:
{packet.to_string()}

Explain briefly for a student:
1. Why these values indicate {prediction}
2. Whether it is normal or malicious
"""

                        with st.spinner("Groq AI is analyzing..."):
                            completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.6,
                            )

                            st.markdown(f"""
<div class="chat-ai">
🤖 <b>AI Analyst:</b><br><br>
{completion.choices[0].message.content}
</div>
""", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"API Error: {e}")

else:
    st.info("👈 Train the model first using the sidebar.")

