
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================================
# Load Model and Preprocessor
# ==========================================

preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)


# ==========================================
# Main Title
# ==========================================

st.title("🍔 Food Delivery Time Prediction")

st.write(
    "Enter the delivery information to predict the delivery time."
)


# ==========================================
# Tabs
# ==========================================

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Analysis"])


# ==========================================
# Prediction Tab
# ==========================================

with tab1:

    st.header("Enter Delivery Information")

    Preparation_Time_Min = st.number_input(
        "Preparation Time (minutes)",
        min_value=0.0,
        value=15.0
    )

    Road_Distance_km = st.number_input(
        "Road Distance (km)",
        min_value=0.0,
        value=5.0
    )

    Average_Speed_kmph = st.number_input(
        "Average Speed (km/h)",
        min_value=0.0,
        value=30.0
    )

    Traffic_Level = st.selectbox(
        "Traffic Level",
        ["Low", "Moderate", "High", "Severe"]
    )

    Delivery_Distance_Category = st.selectbox(
        "Delivery Distance Category",
        ["Short", "Medium", "Long"]
    )

    Vehicle_Type = st.selectbox(
        "Vehicle Type",
        ["Bicycle", "Bike", "Electric Scooter", "Scooter"]
    )

    Weather = st.selectbox(
        "Weather",
        ["Clear", "Cloudy", "Fog", "Rain", "Storm"]
    )

    Pickup_Zone = st.selectbox(
        "Pickup Zone",
        ["CBD", "Commercial", "Industerial", "Residential", "Suburban"]
    )


    # ======================================
    # Prediction Button
    # ======================================

    if st.button("Predict Delivery Time"):

        input_data = pd.DataFrame({
            "Preparation_Time_Min": [Preparation_Time_Min],
            "Road_Distance_km": [Road_Distance_km],
            "Average_Speed_kmph": [Average_Speed_kmph],
            "Traffic_Level": [Traffic_Level],
            "Delivery_Distance_Category": [Delivery_Distance_Category],
            "Vehicle_Type": [Vehicle_Type],
            "Weather": [Weather],
            "Pickup_Zone": [Pickup_Zone]
        })


        # Apply the same preprocessing
        input_encoded = preprocessor.transform(input_data)


        # Make prediction
        prediction = model.predict(input_encoded)


        # Display result
        st.success(
            f"Estimated Delivery Time: {prediction[0]:.2f} minutes"
        )


# ==========================================
# Analysis Tab
# ==========================================

with tab2:

    st.header("Food Delivery Data Analysis")

    st.write(
        "Average delivery time by traffic level."
    )


    # Load dataset
    df = pd.read_csv("Food_Delivery_Time_Prediction.csv")


    # Calculate average delivery time
    traffic_avg = df.groupby(
        "Traffic_Level"
    )["Time_taken_min"].mean()


    # Arrange traffic levels
    order = [
        "Low",
        "Moderate",
        "High",
        "Severe"
    ]

    traffic_avg = traffic_avg.reindex(order)


    # Create chart
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        traffic_avg.index,
        traffic_avg.values
    )

    ax.set_title(
        "Average Delivery Time by Traffic Level"
    )

    ax.set_xlabel(
        "Average Delivery Time (minutes)"
    )

    ax.set_ylabel(
        "Traffic Level"
    )

    st.pyplot(fig)


    # Display table
    st.subheader("Average Delivery Time")

    analysis_df = traffic_avg.reset_index()

    analysis_df.columns = [
        "Traffic Level",
        "Average Delivery Time"
    ]

    st.dataframe(
        analysis_df,
        use_container_width=True
    )
