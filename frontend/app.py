import streamlit as st
import pandas as pd
import requests

# -------------------------------
# Backend URL
# -------------------------------
BASE_URL = "http://127.0.0.1:8000"

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💼",
    layout="wide"
)

# -------------------------------
# CSS Styling (Colorful & Modern)
# -------------------------------
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ff7e5f, #feb47b);
    color: white;
    font-size: 16px;
}
[data-testid="stSidebar"] a {
    color: white;
}

/* Cards */
.metric-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    text-align: center;
    margin-bottom: 15px;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Helvetica', sans-serif;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 8px 24px;
}

/* Prediction Cards */
.prediction-success {
    background: #43cea2;
    color: white;
}
.prediction-fail {
    background: #ff6f61;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("💼 Customer Churn App")
section = st.sidebar.radio("Navigate", ["Home", "Train Model", "Test Model", "Predict Customer"])

# -------------------------------
# Historical Metrics
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# HOME PAGE
# -------------------------------
if section == "Home":
    st.markdown("<h1 style='text-align:center; color:#185a9d;'>💼 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
    st.write("Welcome! Explore features from the sidebar.")

    st.markdown("### 🔹 Features")
    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='metric-card'><h3>Train Model</h3><p>Upload a CSV to train ML model.</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><h3>Test Model</h3><p>Evaluate metrics like accuracy, precision, recall, F1.</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card'><h3>Predict Customer</h3><p>Enter details to predict churn with confidence scores.</p></div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("### 📊 Historical Metrics")
        st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------------------
# TRAIN MODEL PAGE
# -------------------------------
elif section == "Train Model":
    st.markdown("<h2 style='color:#ff7e5f;'>1️⃣ Train Model</h2>", unsafe_allow_html=True)
    st.write("Upload CSV to train model.")
    train_file = st.file_uploader("Upload Train CSV", type=["csv"], key="train_upload")

    if train_file:
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                progress_bar = st.progress(0)
                for percent in range(1, 101, 5):
                    progress_bar.progress(percent)
                response = requests.post(f"{BASE_URL}/train", files={"file": train_file})
                if response.status_code == 200:
                    rows_used = response.json().get("rows_used")
                    st.markdown(f"<div class='metric-card' style='background:#43cea2;color:white;'><h2>✅ Training Completed</h2><h3>Rows Used: {rows_used}</h3></div>", unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {response.text}")

# -------------------------------
# TEST MODEL PAGE
# -------------------------------
elif section == "Test Model":
    st.markdown("<h2 style='color:#ff7e5f;'>2️⃣ Test Model</h2>", unsafe_allow_html=True)
    test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test_upload")

    if test_file:
        if st.button("Test Model", key="test_button"):
            with st.spinner("Testing model..."):
                progress_bar = st.progress(0)
                for percent in range(1, 101, 10):
                    progress_bar.progress(percent)
                response = requests.post(f"{BASE_URL}/test", files={"file": test_file})
                if response.status_code == 200:
                    metrics = response.json().get("metrics")
                    st.success("✅ Testing completed!")

                    # Store history
                    st.session_state.history.append(metrics)

                    # Show metrics in colorful cards
                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2 style='color:#185a9d'>{metrics['accuracy']:.4f}</h2></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='metric-card'><h3>Precision</h3><h2 style='color:#f9a825'>{metrics['precision']:.4f}</h2></div>", unsafe_allow_html=True)
                    col3.markdown(f"<div class='metric-card'><h3>Recall</h3><h2 style='color:#ff6f61'>{metrics['recall']:.4f}</h2></div>", unsafe_allow_html=True)
                    col4.markdown(f"<div class='metric-card'><h3>F1 Score</h3><h2 style='color:#6a1b9a'>{metrics['f1_score']:.4f}</h2></div>", unsafe_allow_html=True)

# -------------------------------
# PREDICT CUSTOMER PAGE
# -------------------------------
elif section == "Predict Customer":
    st.markdown("<h2 style='color:#ff7e5f;'>3️⃣ Predict Single Customer Churn</h2>", unsafe_allow_html=True)

    # Form for input
    with st.form("predict_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5)
            balance = st.number_input("Balance", min_value=0.0, value=50000.0)
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
        with col2:
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            products_number = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
            credit_card = st.selectbox("Has Credit Card?", [1, 0])
            active_member = st.selectbox("Active Member?", [1, 0])

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": estimated_salary
        }
        with st.spinner("Predicting..."):
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                card_class = "prediction-fail" if result["prediction"] == 1 else "prediction-success"
                icon = "⚠️" if result["prediction"] == 1 else "✅"

                st.markdown(f"""
                    <div class='metric-card {card_class}' style='text-align:center;'>
                        <h2>{icon} Prediction: {result['prediction']}</h2>
                        <h3>{result['meaning']}</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="Download Prediction CSV",
                    data=pd.DataFrame([payload]).to_csv(index=False),
                    file_name="prediction.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"❌ Error: {response.text}")