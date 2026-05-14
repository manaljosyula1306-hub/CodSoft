import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Movie Rating Predictor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Rating Prediction System")
st.write("Predict IMDb movie ratings using Machine Learning")

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():
    df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
    return df

try:
    df = load_data()
    st.success("Dataset Loaded Successfully!")
except Exception as e:
    st.error(f"Error Loading CSV: {e}")
    st.stop()

# ======================================
# CLEAN COLUMN NAMES
# ======================================

df.columns = df.columns.str.strip()

# ======================================
# KEEP IMPORTANT COLUMNS
# ======================================

required_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Duration",
    "Votes",
    "Rating"
]

df = df[required_columns]

# ======================================
# REMOVE NULL VALUES
# ======================================

df.dropna(inplace=True)

# ======================================
# CLEAN DURATION
# ======================================

df["Duration"] = (
    df["Duration"]
    .astype(str)
    .str.replace(" min", "", regex=False)
)

df["Duration"] = pd.to_numeric(
    df["Duration"],
    errors="coerce"
)

# ======================================
# CLEAN VOTES
# ======================================

df["Votes"] = (
    df["Votes"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)

# ======================================
# CLEAN RATING
# ======================================

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

# Remove invalid rows again

df.dropna(inplace=True)

# ======================================
# ENCODE CATEGORICAL DATA
# ======================================

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Director"] = director_encoder.fit_transform(df["Director"])
df["Actor 1"] = actor_encoder.fit_transform(df["Actor 1"])

# ======================================
# FEATURES & TARGET
# ======================================

X = df.drop("Rating", axis=1)
y = df["Rating"]

# ======================================
# SPLIT DATA
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# TRAIN MODEL
# ======================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================
# EVALUATE MODEL
# ======================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("📊 Model Performance")

st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# ======================================
# USER INPUTS
# ======================================

st.subheader("🎥 Predict Movie Rating")

genre_input = st.selectbox(
    "Select Genre",
    genre_encoder.classes_
)

director_input = st.selectbox(
    "Select Director",
    director_encoder.classes_[:500]
)

actor_input = st.selectbox(
    "Select Main Actor",
    actor_encoder.classes_[:500]
)

duration_input = st.slider(
    "Movie Duration",
    60,
    240,
    120
)

votes_input = st.slider(
    "Votes",
    0,
    100000,
    5000
)

# ======================================
# PREDICTION
# ======================================

if st.button("Predict Rating"):

    genre_encoded = genre_encoder.transform([genre_input])[0]
    director_encoded = director_encoder.transform([director_input])[0]
    actor_encoded = actor_encoder.transform([actor_input])[0]

    input_data = pd.DataFrame([[
        genre_encoded,
        director_encoded,
        actor_encoded,
        duration_input,
        votes_input
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    st.success(f"⭐ Predicted IMDb Rating: {prediction:.1f}/10")

    if prediction >= 8:
        st.balloons()
        st.write("🔥 Potential Blockbuster")
    elif prediction >= 6:
        st.write("🎬 Likely Well Received")
    else:
        st.write("🍿 Mixed Audience Response")

# ======================================
# DATA PREVIEW
# ======================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())