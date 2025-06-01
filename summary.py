import streamlit as st
import pandas as pd
import random
import altair as alt
from collections import defaultdict, Counter
from rapidfuzz.distance import Levenshtein

st.set_page_config(
    layout="wide",
    page_title="Address‑Cleaning Dashboard",
    initial_sidebar_state="expanded",
)

st.title("🏠 Address Cleaning Dashboard")

@st.cache_data
def load_data():
    # We assume 'cleaned_addresses_with_coords.csv' contains at least:
    #   raw columns:      zip_c, streetName_c, city_c, state_c, address_id (which may now include suffixes "_a", "_b")
    #   clean columns:    clean_zip, clean_street, clean_city, clean_state, clean_full_address
    #   geocode columns:  latitude, longitude
    return pd.read_csv("Data/cleaned_addresses_with_coords.csv", dtype=str)

df = load_data()

# -------------------------------------------------------------------------
# PART A: Change Summary by Rule + Address‑Split 
# -------------------------------------------------------------------------

st.header("1️⃣ Change Summary by Rule")

# 1) ZIP‑Fill: count rows where normalized raw zip_c != clean_zip
zip_raw_norm = df["zip_c"].fillna("").str.strip().str.lstrip("_")
zip_clean    = df["clean_zip"].fillna("")
zip_changes  = int((zip_raw_norm != zip_clean).sum())

# 2) Street‑Clust: recompute exactly as in cleaning script
LEV_THRESHOLD = 0.10

def normalize_str(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def normalized_lev(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))

def is_similar(a: str, b: str, thresh: float = LEV_THRESHOLD) -> bool:
    return normalized_lev(a, b) < thresh

def cluster_names(name_list: list[str], thresh: float = LEV_THRESHOLD) -> dict[str, str]:
    freq = Counter(name_list)
    unique = list(freq.keys())
    n = len(unique)
    if n <= 1:
        return {unique[0]: unique[0]} if unique else {}
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if is_similar(unique[i], unique[j], thresh):
                adj[i].append(j)
                adj[j].append(i)
    rep_map = {}
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)
        comp_names = [unique[k] for k in comp]
        rep = max(comp_names, key=lambda nm: freq[nm])
        for nm in comp_names:
            rep_map[nm] = rep
    return rep_map

# Rebuild "street_parts" exactly as in the cleaning script
df["street_parts"] = (
    df["streetDirectionalPre_c"].fillna("") + " " +
    df["streetName_c"].fillna("") + " " +
    df["streetSuffix_c"].fillna("") + " " +
    df["streetDirectionalPost_c"].fillna("")
).str.strip().map(normalize_str)

# Group by (entity_id, street_number)
group2 = defaultdict(list)
for idx, row in df.iterrows():
    num = row["num1_c"].strip() if isinstance(row["num1_c"], str) else ""
    key = (row["entity_id"], num)
    group2[key].append(idx)

street_changed = set()
for key, idxs in group2.items():
    norms = [df.at[i, "street_parts"] for i in idxs]
    mapping = cluster_names(norms)
    for i in idxs:
        orig = df.at[i, "street_parts"]
        rep  = mapping.get(orig, orig)
        if rep != orig:
            street_changed.add(i)

street_changes = len(street_changed)

# 3) City‑Fill: count rows where normalized raw city_c != clean_city
city_raw_norm = df["city_c"].fillna("").str.strip().str.title()
city_clean    = df["clean_city"].fillna("")
city_changes  = int((city_raw_norm != city_clean).sum())

# 4) State‑Fill: count rows where normalized raw state_c != clean_state
state_raw_norm = df["state_c"].fillna("").str.strip().str.upper()
state_clean    = df["clean_state"].fillna("")
state_changes  = int((state_raw_norm != state_clean).sum())

# 5) Address‑Split: count rows whose address_id contains an underscore (“_”)
#    (our main script appends “_a”, “_b”, etc. whenever it splits)
split_mask    = df["address_id"].fillna("").str.contains("_")
split_changes = int(split_mask.sum())

summary_df = pd.DataFrame({
    "Rule": [
        "ZIP‑Fill", 
        "Street‑Clust", 
        "City‑Fill", 
        "State‑Fill", 
        "Address‑Split"
    ],
    "Rows Changed": [
        zip_changes, 
        street_changes, 
        city_changes, 
        state_changes, 
        split_changes
    ]
})

# Preserve order of rules
summary_df["Rule"] = pd.Categorical(
    summary_df["Rule"],
    categories=summary_df["Rule"],
    ordered=True
)

# Build a “tight” Altair bar chart
bar = (
    alt.Chart(summary_df)
       .mark_bar(size=40)  # narrower bars so they cluster closely
       .encode(
           x=alt.X(
               "Rule:N",
               axis=alt.Axis(labelAngle=0, title="Cleaning Rule", labelPadding=5),
               sort=summary_df["Rule"].tolist()
           ),
           y=alt.Y("Rows Changed:Q", axis=alt.Axis(title="Rows Changed")),
           color=alt.Color(
               "Rule:N",
               scale=alt.Scale(domain=[
                   "ZIP‑Fill",
                   "Street‑Clust",
                   "City‑Fill",
                   "State‑Fill",
                   "Address‑Split"
               ],
               range=[
                   "#1f77b4",  # blue
                   "#ff7f0e",  # orange
                   "#2ca02c",  # green
                   "#d62728",  # red
                   "#9467bd"   # purple for splits
               ]),
               legend=alt.Legend(title="Rule")
           )
       )
       .properties(width=600, height=400)  
)

st.altair_chart(bar, use_container_width=False)
st.dataframe(summary_df, use_container_width=True)

# -------------------------------------------------------------------------
# PART B: Show up to SAMPLE_SIZE “Before → After” examples
# -------------------------------------------------------------------------

st.header("2️⃣ Sample Changes")
SAMPLE_SIZE = 5

def sample_changes(raw_col, clean_col, rule_name, normalize_raw=None):
    """
    Show up to SAMPLE_SIZE random rows where normalized raw_col != clean_col.
    """
    if normalize_raw:
        raw_vals = normalize_raw(df[raw_col].fillna(""))
    else:
        raw_vals = df[raw_col].fillna("")
    clean_vals = df[clean_col].fillna("")
    mask = raw_vals != clean_vals
    changed_idxs = df.index[mask].tolist()

    if not changed_idxs:
        st.markdown(f"**{rule_name}** – no changes detected.")
        return

    st.subheader(f"{rule_name} — Showing up to {SAMPLE_SIZE} Samples")
    chosen = random.sample(changed_idxs, min(len(changed_idxs), SAMPLE_SIZE))
    sample_df = pd.DataFrame({
        "Index": chosen,
        f"Raw ({raw_col})": raw_vals.loc[chosen].tolist(),
        f"Clean ({clean_col})": clean_vals.loc[chosen].tolist()
    })
    st.write(sample_df)

# ZIP‑Fill sample
sample_changes(
    raw_col="zip_c",
    clean_col="clean_zip",
    rule_name="ZIP‑Fill",
    normalize_raw=lambda s: s.str.strip().str.lstrip("_")
)

# Street‑Clust sample
sample_changes(
    raw_col="street_parts",
    clean_col="clean_street",
    rule_name="Street‑Clust"
)

# City‑Fill sample
sample_changes(
    raw_col="city_c",
    clean_col="clean_city",
    rule_name="City‑Fill",
    normalize_raw=lambda s: s.str.strip().str.title()
)

# State‑Fill sample
sample_changes(
    raw_col="state_c",
    clean_col="clean_state",
    rule_name="State‑Fill",
    normalize_raw=lambda s: s.str.strip().str.upper()
)

# -------------------------------------------------------------------------
# Address‑Split sample: show up to SAMPLE_SIZE sets of splits for each base ID
# -------------------------------------------------------------------------
st.subheader(f"Address‑Split — Showing up to {SAMPLE_SIZE} Split Pairs")

# 1) collect all indices whose address_id contains “_”
split_idxs = df.index[split_mask].tolist()

# 2) group by the base address ID (i.e. drop the suffix after the underscore)
split_groups: dict[str, list[int]] = defaultdict(list)
for idx in split_idxs:
    full_aid = df.at[idx, "address_id"]
    base_aid = full_aid.split("_", 1)[0]
    split_groups[base_aid].append(idx)

# 3) keep only those base IDs that actually split into ≥2 rows
multi_split = {base: idxs for base, idxs in split_groups.items() if len(idxs) > 1}

if multi_split:
    # 4) randomly pick up to SAMPLE_SIZE different base IDs
    chosen_bases = random.sample(
        list(multi_split.keys()),
        min(len(multi_split), SAMPLE_SIZE)
    )

    # 5) build a DataFrame showing all splits for each chosen base ID
    rows = []
    for base in chosen_bases:
        for idx in multi_split[base]:
            rows.append({
                "Index": idx,
                "New AID": df.at[idx, "address_id"],
                "Clean Addr": df.at[idx, "clean_full_address"]
            })

    sample_split = pd.DataFrame(rows)
    st.write(sample_split)

else:
    st.markdown("**Address‑Split** – no multi‐split address IDs found.")

# End of dashboard
