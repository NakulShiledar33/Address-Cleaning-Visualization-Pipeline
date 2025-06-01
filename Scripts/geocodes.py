import pandas as pd
import pgeocode

# 1) Load your cleaned addresses (with a column “clean_zip”)
df = pd.read_csv("Data/cleaned_addresses.csv", dtype=str)

# 2) Initialize the US‐ZIP lookup object
nomi = pgeocode.Nominatim("us")

# 3) Bulk‐lookup latitude & longitude for each 5‑digit ZIP
#    pgeocode returns a DataFrame when you pass a list of zips.
zip_list = df["clean_zip"].fillna("").str[:5].tolist()
#   (take first 5 chars to ensure “02115‑0000” → “02115”)
zip_df = nomi.query_postal_code(zip_list)

# zip_df has columns: postal_code, country_code, place_name, state_name, latitude, longitude, etc.
# The order matches zip_list, so zip_df.latitude[i] corresponds to zip_list[i].

df["latitude"]  = zip_df["latitude"]
df["longitude"] = zip_df["longitude"]

# 4) Save to a new CSV
df.to_csv("Data/cleaned_addresses_with_coords.csv", index=False)

print("Done! Saved cleaned_addresses_with_coords.csv with ZIP‑centroid lat/lon.")