import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler,PowerTransformer
scaler=StandardScaler()
with open("models.pkl",'rb') as f:
    models=pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pt.pkl", "rb") as f:
    pt=pickle.load(f)
model = models["XGBRegressor"] 
st.title("gdb prediction App using ML")
col1, col2, col3 = st.columns(3)

with col1:
    population = st.number_input("Population", min_value=1.0, value=1000.0)
    density = st.number_input("Pop. Density (per sq. mi.)", min_value=0.0, value=50.0)
    net_migration = st.number_input("Net migration", value=0.0)
    literacy = st.number_input("Literacy (%)", min_value=0.0, max_value=100.0, value=80.0)
    arable = st.number_input("Arable (%)", min_value=0.0, max_value=100.0, value=20.0)
    birthrate = st.number_input("Birthrate", min_value=0.0, value=15.0)

with col2:
    area = st.number_input("Area (sq. mi.)", min_value=1.0, value=500.0)
    coastline = st.number_input("Coastline (coast/area ratio)", min_value=0.0, value=0.1)
    infant_mortality = st.number_input("Infant mortality (per 1000 births)", min_value=0.0, value=10.0)
    phones = st.number_input("Phones (per 1000)", min_value=0.0, value=200.0)
    crops = st.number_input("Crops (%)", min_value=0.0, max_value=100.0, value=5.0)
    deathrate = st.number_input("Deathrate", min_value=0.0, value=5.0)

with col3:
    other = st.number_input("Other (%)", min_value=0.0, max_value=100.0, value=75.0)
    climate = st.number_input("Climate", min_value=0.0, max_value=5.0, value=1.0)
    agriculture = st.number_input("Agriculture", min_value=0.0, max_value=1.0, value=0.2)
    industry = st.number_input("Industry", min_value=0.0, max_value=1.0, value=0.3)
    service = st.number_input("Service", min_value=0.0, max_value=1.0, value=0.5)
if st.button("🔍 Predict GDP"):
    input_data = pd.DataFrame([[
    population,      # Population
    area,            # Area (sq. mi.)
    density,         # Pop. Density (per sq. mi.)
    coastline,       # Coastline (coast/area ratio)
    net_migration,   # Net migration
    infant_mortality,# Infant mortality (per 1000 births)
    literacy,        # Literacy (%)
    phones,          # Phones (per 1000)
    arable,          # Arable (%)
    crops,           # Crops (%)
    other,           # Other (%)
    climate,         # Climate
    birthrate,       # Birthrate
    deathrate,       # Deathrate
    agriculture,     # Agriculture
    industry,        # Industry
    service          # Service
]], columns=[
    "Population", "Area (sq. mi.)", "Pop. Density (per sq. mi.)", "Coastline (coast/area ratio)",
    "Net migration", "Infant mortality (per 1000 births)", "Literacy (%)", "Phones (per 1000)",
    "Arable (%)", "Crops (%)", "Other (%)", "Climate", "Birthrate", "Deathrate",
    "Agriculture", "Industry", "Service"
])

    transformed_data=pt.transform(input_data)
    scaled_input = scaler.transform(transformed_data)

    gdp = model.predict(scaled_input)[0]

    st.success(f"### 💰 Predicted GDP ($ per capita): **{gdp:.2f}**")


st.write("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")