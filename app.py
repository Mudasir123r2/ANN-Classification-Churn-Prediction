import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

# Load model
model = tf.keras.models.load_model('model.keras',compile=False)

# Load encoders and scaler
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl','rb') as f: 
    onehot_encoder_geo = pickle.load(f)   

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .prediction-box {
            padding: 15px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .positive {
            background-color: #ffcccc;
            color: #b30000;
        }
        .negative {
            background-color: #d4edda;
            color: #155724;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# App Title
# -----------------------------
st.markdown("<div class='title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill out the customer details below to predict churn probability</div>", unsafe_allow_html=True)

# -----------------------------
# Input Layout with Columns
# -----------------------------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        tenure = st.slider('📅 Tenure (years)', 0, 10)
        num_of_products = st.slider('🛒 Number of Products', 1, 4)

    with col2:
        credit_score = st.number_input('💳 Credit Score', min_value=0, step=1)
        balance = st.number_input('💰 Balance', min_value=0.0, step=100.0)
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, step=100.0)
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
        is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale data
input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f"### 📈 Churn Probability: **{prediction_proba:.2f}**")

    if prediction_proba > 0.5:
        st.markdown(f"<div class='prediction-box positive'>⚠️ The customer is **likely to churn**.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box negative'>✅ The customer is **not likely to churn**.</div>", unsafe_allow_html=True)
