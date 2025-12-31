import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
API_URL = "https://smart-electricity-load-forecasting-7ry2.onrender.com/predict"
WINDOW_SIZE = 14
ANOMALY_THRESHOLD = 1.5 
st.set_page_config(page_title="Electricity Load Forecasting", layout="centered")
st.title("Smart Electricity Load Forecasting")
st.markdown("Predict next-day electricity demand using LSTM")
st.subheader("Enter last 14 daily load values")
values = []
for i in range(WINDOW_SIZE):
    val = st.number_input(
        f"Day {i+1}",
        min_value=0.0,
        step=0.1,
        key=f"val_{i}"
    )
    values.append(val)
if st.button("Predict Next Day Load"):
    if len(values) != WINDOW_SIZE:
        st.error("Please enter exactly 14 values.")
    else:
        payload = {"past_values": values}
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            prediction = result["predicted_next_day_load"]
            st.success(f" Predicted Next-Day Load: **{prediction:.2f} kW**")
            mean_load = np.mean(values)
            std_load = np.std(values)
            if abs(prediction - mean_load) > ANOMALY_THRESHOLD * std_load:
                st.error(" Anomaly Detected!")
            else:
                st.info("Load is within normal range")
            fig, ax = plt.subplots()
            ax.plot(range(1, 15), values, marker="o", label="Past Load")
            ax.scatter(15, prediction, color="red", label="Predicted Load")
            ax.set_xlabel("Day")
            ax.set_ylabel("Electricity Load")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

        except Exception as e:
            st.error("API error. Is the server live?")
            st.write(e)

