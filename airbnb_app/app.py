import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and features
model = joblib.load('airbnb_price_model_rf.pkl')
features = pd.read_csv('airbnb_price_model_features.csv').squeeze().tolist()

# Sidebar: Input form
st.sidebar.header("🔍 Airbnb Listing Info")
input_data = {}

# Manual input fields (simplified)
input_data['accommodates'] = st.sidebar.slider('Accommodates', 1, 16, 2)
input_data['bedrooms'] = st.sidebar.slider('Bedrooms', 0, 10, 1)
input_data['bathrooms'] = st.sidebar.slider('Bathrooms', 0.0, 5.0, 1.0)
input_data['beds'] = st.sidebar.slider('Beds', 0, 10, 1)
input_data['latitude'] = st.sidebar.number_input('Latitude', value=34.05)
input_data['longitude'] = st.sidebar.number_input('Longitude', value=-118.25)

room_type = st.sidebar.selectbox('Room Type', ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'])
property_type = st.sidebar.selectbox('Property Type', ['Entire home', 'Entire rental unit', 'Private room in home', 'Room in hotel'])
city = st.sidebar.selectbox('City', ['Los Angeles', 'San Francisco', 'San Diego', 'Santa Clara', 'San Mateo', 'Santa Cruz'])

# Convert to model input
input_df = pd.DataFrame([input_data])

# Add one-hot columns
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# Set the right one-hot values
input_df[f'room_type_{room_type}'] = 1
input_df[f'property_type_{property_type}'] = 1
input_df[f'city_{city}'] = 1

# Reorder columns
input_df = input_df[features]

# Prediction
log_price = model.predict(input_df)[0]
predicted_price = np.expm1(log_price)

st.title("🏡 Airbnb Price Predictor")
st.write("### 💰 Estimated Price:")
st.success(f"${predicted_price:.2f} per night")

# --- Optional Visualizations ---
st.write("### 📊 Feature Importance (Top 10)")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False).head(10)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
st.pyplot(fig)

# Load listings for maps or distributions
if st.checkbox("Show Sample Listing Map & Stats"):
    df = pd.read_csv("../data/listings_combined.csv")
    df = df[df['price'].notnull()]
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    st.map(df[['latitude', 'longitude']].dropna().sample(1000))  # Sample map
    st.write("Price Distribution:")
    st.bar_chart(df['price'].clip(upper=500).value_counts().sort_index())