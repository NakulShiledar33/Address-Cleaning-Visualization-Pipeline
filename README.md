This repository provides a complete, scalable pipeline for normalizing and cleaning address data, filling in missing ZIPs/cities/states, clustering similar street names, and (optionally) splitting any raw address ID that maps to multiple distinct cleaned addresses. After cleaning, you can geocode each 5‑digit ZIP centroid and launch Streamlit dashboards to inspect both change summaries and a map of all cleaned addresses.
```
├── Data/
│   ├── fa.csv                       # Raw address master table (AID → address fields)
│   ├── fe.csv                       # Entity master (EID → entity info)
│   ├── r_fe_fa.csv                  # Relation table (EID, AID)
│   ├── cleaned_addresses.csv        # Output after running clean_addresses.py
│   └── cleaned_addresses_with_coords.csv  # After geocode_zips.py
│
├── main.py                           # Main cleaning script
├── scripts/
│   └── geocode_zips.py              # Optional: look up ZIP centroids via pgeocode
│
├── streamlit_app.py                 # Streamlit page: bar chart of changes + sample rows
└── pages/
    └── Map.py                       # Streamlit page: map of cleaned addresses
```
## Dependencies
- Python 3.8+
- pandas
- numpy
- rapidfuzz
- pgeocode ( Optional )
- streamlit ( Optional )

```bash
pip install pandas numpy rapidfuzz pgeocode streamlit
```

## Usage

### 1. Clean & Normalize Addresses

**Run the main pipeline:**
```bash
python main.py
```

**Inputs:**

- `Data/r_fe_fa.csv` – Relation table linking entities to address IDs  
- `Data/fa.csv` – Address master  
- `Data/fe.csv` – Entity master  

**Outputs:**

- `Data/cleaned_addresses.csv`, containing:  
  - `clean_zip`, `clean_street`, `clean_city`, `clean_state`  
  - `clean_full_address`  
  - Possibly updated `address_id` (with `_a`, `_b` suffixes for splits)  

**Cleaning rules implemented:**

- **ZIP‑Fill**  
  For each (`entity_id`, `street_number`, `street_norm`), if exactly one non‑blank ZIP exists, fill missing `clean_zip` with that ZIP.

- **Street‑Clust**  
  Within each (`entity_id`, `street_number`), cluster `clean_street` variants whose normalized Levenshtein distance < 0.10, then replace all with the most common spelling.

- **City‑Fill**  
  For each (`entity_id`, `street_number`, `clean_street`, `clean_zip`), if exactly one non‑blank `clean_city` exists, fill missing `clean_city` rows.

- **State‑Fill**  
  For each (`entity_id`, `street_number`, `clean_street`, `clean_zip`, `clean_city`), if exactly one non‑blank `clean_state` exists, fill missing `clean_state` rows.

- **Split Address IDs**  
  If a raw `address_id` maps to multiple distinct `clean_full_address` values, generate new sub‑IDs (`oldID_a`, `oldID_b`, …) so that each unique cleaned address has its own ID.

**Reporting:**

- Prints counts of how many rows changed under each rule.  
- Prints up to 5 random before→after examples for ZIP, Street, City, and State changes, plus up to 5 split‑ID samples.

## 2. (Optional) Geocode ZIPs

If you want to visualize on a map, run:
```bash
python scripts/geocode_zips.py
```
**Input:**
- `Data/cleaned_addresses.csv`

**Output:**
- `Data/cleaned_addresses_with_coords.csv`
  
Generates latitude and longitude for each 5‑digit cleaned ZIP
Uses `pgeocode` to look up each 5‑digit `clean_zip` and append latitude and longitude.

## 3. (Optional) Launch Streamlit Dashboards

### 3A. Bar Chart & Sample Display

```bash
streamlit run summary.py
```
**Shows:**
- A bar chart (Altair) of “Rows Changed” for each cleaning rule (ZIP‑Fill, Street‑Clust, City‑Fill, State‑Fill, Address‑Split).
- A table of those counts.
- Under “Sample Changes,” up to 5 random before→after examples for each rule.

### 3B. Map of Cleaned Addresses

**Requirement:**
- `Data/cleaned_addresses_with_coords.csv` must exist (run the geocode step first).

**Shows:**
- A map centered on the continental USA, plotting all cleaned address centroids (down‑samples to 30,000 if > 50,000 points).  
- A sample table of `clean_full_address`, `lat`, and `lon`.









