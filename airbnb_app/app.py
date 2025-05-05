import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Airbnb Price Estimator",
    page_icon="🏡",
    layout="wide"
)

# Load model and features
model = joblib.load('airbnb_price_model_rf.pkl')
features = pd.read_csv('airbnb_price_model_features.csv').squeeze().tolist()

# Sidebar input
st.sidebar.title("Listing Parameters")
st.sidebar.markdown("Configure details about your Airbnb listing.")

input_data = {
    'accommodates': st.sidebar.slider('Accommodates', 1, 16, 2, step=1),
    'bedrooms': st.sidebar.slider('Bedrooms', 0, 10, 1, step=1),
    'bathrooms': st.sidebar.slider('Bathrooms', 0, 5, 1, step=1),
    'beds': st.sidebar.slider('Beds', 0, 10, 1, step=1)
}

room_type = st.sidebar.selectbox('Room Type', ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'])
property_type = st.sidebar.selectbox('Property Type', ['Entire home', 'Entire rental unit', 'Private room in home', 'Room in hotel'])
city = st.sidebar.selectbox('City', ['Los Angeles', 'San Francisco', 'San Diego', 'Santa Clara', 'San Mateo', 'Santa Cruz'])

# Build model input
input_df = pd.DataFrame([input_data])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df[f'room_type_{room_type}'] = 1
input_df[f'property_type_{property_type}'] = 1
input_df[f'city_{city}'] = 1
input_df = input_df[features]

# Predict
log_price = model.predict(input_df)[0]
predicted_price = np.expm1(log_price)

# Predicted Price Display
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='
        background-color:#1e1e1e;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    '>
        <h2 style='color:#08D9D6; margin-bottom: 10px;'>Predicted Nightly Price</h2>
        <h1 style='font-size: 60px; color: #FFFFFF; margin: 0;'>${predicted_price:.2f}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Feature importance
# Mapping raw feature names to human-readable labels
name_mapping = {
    'accommodates': 'Accommodates',
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'beds': 'Beds',
    'review_scores_rating': 'Rating Score',
    'number_of_reviews': 'Number of Reviews',
    'availability_365': 'Availability (Days/Year)',
    'calculated_host_listings_count': 'Host Listings Count',
    'calculated_host_listings_count_entire_homes': 'Entire Homes Listed',
    'calculated_host_listings_count_private_rooms': 'Private Rooms Listed',
    'room_type_Entire home/apt': 'Room: Entire Home',
    'room_type_Private room': 'Room: Private Room',
    'room_type_Shared room': 'Room: Shared Room',
    'room_type_Hotel room': 'Room: Hotel Room',
    'property_type_Entire home': 'Property: Entire Home',
    'property_type_Private room in home': 'Property: Private Room in Home',
    'property_type_Entire rental unit': 'Property: Rental Unit',
    'property_type_Room in hotel': 'Property: Room in Hotel',
    'city_Los Angeles': 'City: Los Angeles',
    'city_San Francisco': 'City: San Francisco',
    'city_San Diego': 'City: San Diego',
    'city_Santa Clara': 'City: Santa Clara',
    'city_Santa Cruz': 'City: Santa Cruz',
    'city_San Mateo': 'City: San Mateo',
}

importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

importance_df = importance_df[~importance_df['Feature'].isin(['latitude', 'longitude'])]

# Sort and pick top 10
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

# Rename remaining features for display
importance_df['Feature'] = importance_df['Feature'].apply(
    lambda x: name_mapping.get(x, x.replace('_', ' ').title())
)

# Plot
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=importance_df, y='Feature', x='Importance', palette='crest', ax=ax)
ax.set_title("Top Features Impacting Price", fontsize=14)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
st.pyplot(fig)

st.divider()

# Optional stats
import pydeck as pdk

with st.expander("Show Listing Distribution and Map"):
    try:
        df = pd.read_csv("../data/listings_combined.csv")
        df = df[df['price'].notnull()]
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
        df[['latitude', 'longitude']] = df[['latitude', 'longitude']].dropna()

        # Sample for performance (adjust if needed)
        map_df = df[['latitude', 'longitude']].sample(500)

        st.write("Map of Airbnb Listings")
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_df['latitude'].mean(),
                longitude=map_df['longitude'].mean(),
                zoom=10,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position='[longitude, latitude]',
                    get_color='[255, 0, 0, 160]',
                    get_radius=100,
                    pickable=True,
                ),
            ],
            tooltip={"text": "Listing Location"},
        ))

        # Optional: Price by neighborhood bar chart
        if 'neighbourhood_cleansed' in df.columns:
            avg_price_by_area = (
                df[df['price'] < 500]
                .groupby('neighbourhood_cleansed')['price']
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )
            st.write("💰 Average Price by Neighborhood (Top 20):")
            st.bar_chart(avg_price_by_area)
        else:
            st.info("Location-level data not available for price trends.")

    except FileNotFoundError:
        st.error("The file 'listings_combined.csv' is missing.")
st.divider()



# Load clustered dataset
with st.expander("Show Clustered Listings and Neighborhood Demographics"):
    try:
        df_clustered = pd.read_csv("airbnb_cluster.csv")  # Your processed file
        df_clustered = df_clustered.dropna(subset=["latitude", "longitude", "cluster"])

        # Predict cluster of the current input
        kmeans = joblib.load("kmeans_model.pkl")  # If you saved it
        cluster_input_features = ['bathrooms', 'beds', 'room_type_Private room', 'price',
                                  'review_scores_rating', 'review_scores_cleanliness',
                                  'review_scores_location', 'review_scores_value']
        # Build cluster input from the same sidebar selections
        input_for_cluster = {
            'bathrooms': input_data['bathrooms'],
            'beds': input_data['beds'],
            'price': predicted_price,
            'review_scores_rating': 4.5,
            'review_scores_cleanliness': 4.5,
            'review_scores_location': 4.5,
            'review_scores_value': 4.5,
            'room_type_Private room': 1 if room_type == 'Private room' else 0,
        }

        input_for_cluster = pd.DataFrame([input_for_cluster])
        input_for_cluster = input_for_cluster.astype(np.float64)


        umap_model = joblib.load('umap_model.pkl')
        X_umap = umap_model.transform(input_for_cluster)

        input_cluster = kmeans.predict(X_umap)[0]
        cluster_label = df_clustered[df_clustered['cluster'] == input_cluster]['cluster_label'].iloc[0]

        st.info(f"Based on your inputs, this listing most likely falls into **Cluster {cluster_label}**.")

        # Show neighborhood stats by cluster
        cluster_stats = df_clustered.groupby('cluster_label').agg({
            'price': 'mean',
            'beds': 'mean',
            'room_type_Private room': 'mean',
            'bathrooms': 'mean'
        }).round(1)
        st.write("**Average Stats by Cluster:**")
        st.dataframe(cluster_stats)

        # Color by cluster (assign each cluster a distinct color)
        def cluster_color(cluster):
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 165, 0], [128, 0, 128]
            ]
            return colors[cluster % len(colors)] + [160]

        df_clustered['color'] = df_clustered['cluster'].apply(cluster_color)

        st.write("**Map of Listings by Cluster**")
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df_clustered['latitude'].mean(),
                longitude=df_clustered['longitude'].mean(),
                zoom=10,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_clustered,
                    get_position='[longitude, latitude]',
                    get_color='color',
                    get_radius=100,
                    pickable=True,
                )
            ],
            tooltip={"text": "Cluster: {cluster}\nPrice: ${price}"}
        ))

    except Exception as e:
        st.error(f"Error loading clustered data: {e}")
