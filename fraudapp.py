import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection System")

# =========================================
# LOAD DATA
# =========================================

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

st.success("Dataset Loaded Successfully!")

# =========================================
# TRAIN MODEL
# =========================================

@st.cache_resource
def train_model():

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()

    X["Amount"] = scaler.fit_transform(
        X["Amount"].values.reshape(-1, 1)
    )

    X["Time"] = scaler.fit_transform(
        X["Time"].values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return model, precision, recall, f1, roc_auc

# =========================================
# RUN TRAINING
# =========================================

with st.spinner("Training ML Model..."):

    model, precision, recall, f1, roc_auc = train_model()

st.success("Model Trained Successfully!")

# =========================================
# SHOW METRICS
# =========================================

st.subheader("📊 Model Performance")

st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")
st.write(f"ROC-AUC: {roc_auc:.4f}")

# =========================================
# INPUT SECTION
# =========================================

st.subheader("💳 Enter Transaction Details")

Time = st.number_input("Time", value=0.0)
Amount = st.number_input("Amount", value=0.0)

features = []

for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

# =========================================
# PREDICTION
# =========================================

if st.button("Predict Transaction"):

    input_data = pd.DataFrame([[
        Time,
        *features,
        Amount
    ]], columns=[
        'Time',
        'V1', 'V2', 'V3', 'V4', 'V5',
        'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14',
        'V15', 'V16', 'V17', 'V18',
        'V19', 'V20', 'V21', 'V22',
        'V23', 'V24', 'V25', 'V26',
        'V27', 'V28',
        'Amount'
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

# =========================================
# DATA PREVIEW
# =========================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())