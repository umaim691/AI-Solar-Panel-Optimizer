import streamlit as st
import pandas as pd
import numpy as np
import joblib
from groq import Groq

st.set_page_config(
    page_title="AI Solar Panel Optimizer",
    page_icon="☀️"
)

# Load the trained model
model = joblib.load("solar_model.pkl")

st.title("☀️ AI Solar Panel Optimizer")
st.write(
    "Physics-informed machine learning for solar panel tilt optimization."
)

irradiance = st.slider(
    "Solar Irradiance (W/m²)",
    400, 1000, 800
)

temperature = st.slider(
    "Temperature (°C)",
    10, 50, 30
)

latitude = st.slider(
    "Latitude (°)",
    0, 60, 30
)

if st.button("☀️ Optimize Panel"):

    angles = np.arange(0, 61, 1)

    test_data = pd.DataFrame({
        "tilt": angles,
        "irradiance": irradiance,
        "temperature": temperature,
        "latitude": latitude
    })

    predictions = model.predict(test_data)

    best_index = np.argmax(predictions)

    best_angle = int(angles[best_index])
    best_energy = float(predictions[best_index])

    st.success(
        f"Recommended Tilt: {best_angle}°"
    )

    st.metric(
        "Predicted Energy",
        f"{best_energy:.2f}"
    )

    chart_df = pd.DataFrame({
        "Tilt Angle": angles,
        "Predicted Energy": predictions
    })

    st.line_chart(
        chart_df.set_index("Tilt Angle")
    )

    # Groq AI explanation
# Groq AI explanation
st.subheader("🤖 AI Explanation")

try:
    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(api_key=api_key)

    prompt = f"""
You are a solar energy expert.

The physics-informed ML optimizer produced:

Solar irradiance: {irradiance} W/m²
Temperature: {temperature} °C
Latitude: {latitude}°
Recommended tilt angle: {best_angle}°
Predicted energy output: {best_energy:.2f}

Explain this result in simple scientific language.

Explain:
1. Why this tilt angle was selected.
2. How irradiance affects energy.
3. How temperature affects solar panel performance.

Keep the explanation under 150 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    explanation = response.choices[0].message.content

    st.write(explanation)

except KeyError:
    st.error(
        "GROQ_API_KEY is not available in Streamlit Secrets."
    )

except Exception as e:
    st.error(f"Groq error: {e}")
