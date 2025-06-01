# pages/Map.py

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🗺️ Map of Cleaned Addresses")

@st.cache_data
def load_geodata():
    # Same CSV as main page; must include 'latitude' & 'longitude'
    return pd.read_csv("Data/cleaned_addresses_with_coords.csv", dtype=str)

geo_df = load_geodata()

if "latitude" in geo_df.columns and "longitude" in geo_df.columns:
    # Convert strings to float, drop missing
    geo_df = geo_df.dropna(subset=["latitude", "longitude"]).copy()
    geo_df["latitude"]  = geo_df["latitude"].astype(float)
    geo_df["longitude"] = geo_df["longitude"].astype(float)

    # If too many points, sample down (optional)
    if len(geo_df) > 50000:
        geo_df = geo_df.sample(30000, random_state=1)
        st.info("Showing a random subset of 30k points on the map.")

    # Rename for st.map
    geo_df = geo_df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Center the map on the continental USA
    st.map(geo_df[["lat", "lon"]], zoom=3)  
    #   - zoom=3 ≈ show entire USA by default

    st.markdown(f"**Total plotted points:** {len(geo_df):,}")
else:
    st.warning("⚠️ Cannot find `latitude`/`longitude` columns. Run the geocoding step first.")

# Optionally, show a small sample of the geo data
st.subheader("Sample of Geocoded Addresses")
st.dataframe(geo_df.head(10)[["clean_full_address", "lat", "lon"]])
