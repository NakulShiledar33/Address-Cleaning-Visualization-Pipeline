#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from collections import defaultdict, Counter
import random

# ─────────────── CONFIGURATION ───────────────
FA_CSV        = "Data/fa.csv"
FE_CSV        = "Data/fe.csv"
REL_CSV       = "Data/r_fe_fa.csv"
OUTPUT_CSV    = "Data/cleaned_addresses.csv"
LEV_THRESHOLD = 0.10
SAMPLE_SIZE   = 5  # how many changed rows to randomly show

def normalize_str(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

def normalized_lev(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))

def is_similar(a: str, b: str, thresh: float = LEV_THRESHOLD) -> bool:
    return normalized_lev(a, b) < thresh

def cluster_names(name_list: list[str], thresh: float = LEV_THRESHOLD) -> dict[str, str]:
    """
    Given a list of normalized street strings, returns a map {orig_name → representative_name}
    by grouping any pair whose normalized_lev < thresh into the same connected component,
    then choosing the most frequent member in each component as its representative.
    """
    freq = Counter(name_list)
    unique = list(freq.keys())
    if len(unique) <= 1:
        return {unique[0]: unique[0]} if unique else {}

    n = len(unique)
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
        comp_indices = []
        while stack:
            node = stack.pop()
            comp_indices.append(node)
            for neigh in adj[node]:
                if neigh not in visited:
                    visited.add(neigh)
                    stack.append(neigh)

        comp_names = [unique[k] for k in comp_indices]
        representative = max(comp_names, key=lambda nm: freq[nm])
        for nm in comp_names:
            rep_map[nm] = representative

    return rep_map

def build_full_address(r: pd.Series) -> str:
    """
    Construct “clean_full_address” by:
      1) trimming “.0” off of street_number if present
      2) Title‐capitalizing the street, then appending city, state, zip
    """
    raw_num = r["street_number"]
    if isinstance(raw_num, str) and raw_num.endswith(".0"):
        num = raw_num[:-2]
    else:
        try:
            f = float(raw_num)
            num = str(int(f)) if f.is_integer() else raw_num
        except (ValueError, TypeError):
            num = raw_num

    street_title = " ".join(w.title() for w in r["clean_street"].split())
    city_part    = r["clean_city"].strip().title()
    state_part   = r["clean_state"].strip().upper()
    zip_part     = r["clean_zip"].strip()

    addr_core = []
    if num:
        addr_core.append(num)
    if street_title:
        addr_core.append(street_title)
    addr = " ".join(addr_core)

    if city_part:
        addr = f"{addr}, {city_part}" if addr else city_part
    if state_part:
        addr = f"{addr}, {state_part}" if addr else state_part
    if zip_part:
        addr = f"{addr} {zip_part}" if addr else zip_part

    return addr.strip()

def print_rule_samples(
    merged: pd.DataFrame,
    changed_indices: set[int],
    rule_name: str,
    before_col: str,
    after_col: str
) -> None:
    """
    Print up to SAMPLE_SIZE random rows where after_col differs from before_col.
    """
    if not changed_indices:
        return
    count = min(len(changed_indices), SAMPLE_SIZE)
    print(f"\nSample {count} under {rule_name}:")
    for i in random.sample(changed_indices, count):
        before_val = merged.at[i, before_col]
        after_val  = merged.at[i, after_col]
        print(f" Row {i}: BEFORE→ {before_col}='{before_val}', AFTER→ {after_col}='{after_val}'")

def main():
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Load CSVs
    # ────────────────────────────────────────────────────────────────────────────
    fa  = pd.read_csv(FA_CSV, dtype=str)
    fe  = pd.read_csv(FE_CSV, dtype=str)
    rel = pd.read_csv(REL_CSV, dtype=str)

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Merge entity ↔ address
    # ────────────────────────────────────────────────────────────────────────────
    merged = rel.merge(
        fa,
        left_on="AID_2",
        right_on="AID",
        how="left",
        validate="many_to_one"
    )
    merged.rename(columns={"EID_1": "entity_id", "AID_2": "address_id"}, inplace=True)

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Build "street_parts" and normalize raw columns
    # ────────────────────────────────────────────────────────────────────────────
    merged["street_parts"] = (
        merged["streetDirectionalPre_c"].fillna("") + " " +
        merged["streetName_c"].fillna("") + " " +
        merged["streetSuffix_c"].fillna("") + " " +
        merged["streetDirectionalPost_c"].fillna("")
    ).str.strip()

    merged["street_norm"]   = merged["street_parts"].map(normalize_str)
    merged["city_norm"]     = merged["city_c"].map(lambda s: s.strip().title() if isinstance(s, str) else "")
    merged["state_norm"]    = merged["state_c"].map(lambda s: s.strip().upper() if isinstance(s, str) else "")
    merged["zip_norm"]      = merged["zip_c"].map(lambda s: s.strip().lstrip("_") if isinstance(s, str) else "")
    merged["street_number"] = merged["num1_c"].map(lambda s: s.strip() if isinstance(s, str) else "")

    # Initialize “clean_” columns from normalized originals
    merged["clean_zip"]    = merged["zip_norm"]
    merged["clean_street"] = merged["street_norm"]
    merged["clean_city"]   = merged["city_norm"]
    merged["clean_state"]  = merged["state_norm"]

    # Track changed rows for each rule
    changed_zip    = set()
    changed_street = set()
    changed_city   = set()
    changed_state  = set()

    # ────────────────────────────────────────────────────────────────────────────
    # RULE 1: ZIP‑Fill via fast pandas groupby + merge
    # ────────────────────────────────────────────────────────────────────────────
    zip_key_cols = ["entity_id", "street_number", "street_norm"]

    # 1a) Build group‐level nunique count of non‐blank zip_norm
    nonblank_zips = merged.loc[merged["zip_norm"] != "", :]
    zip_nunique = (
        nonblank_zips
        .groupby(zip_key_cols)["zip_norm"]
        .nunique()
        .rename("zip_unique_count")
    )

    # 1b) Build group‐level “the single zip value” for groups with exactly one unique zip
    zip_single_val = (
        nonblank_zips
        .groupby(zip_key_cols)["zip_norm"]
        .first()
        .rename("zip_single_val")
    )

    # 1c) Merge back these two group‐level series into merged
    merged = (
        merged
        .merge(zip_nunique.reset_index(), on=zip_key_cols, how="left")
        .merge(zip_single_val.reset_index(), on=zip_key_cols, how="left")
    )

    # 1d) Fill only those rows where clean_zip is blank AND zip_unique_count == 1
    mask_zip_fill = merged["clean_zip"].eq("") & (merged["zip_unique_count"] == 1)
    merged.loc[mask_zip_fill, "clean_zip"] = merged.loc[mask_zip_fill, "zip_single_val"]
    changed_zip.update(merged.index[mask_zip_fill])

    # drop helper columns
    merged.drop(columns=["zip_unique_count", "zip_single_val"], inplace=True)

    # ────────────────────────────────────────────────────────────────────────────
    # RULE 2: Street‑Clust (string‐distance clustering per (entity_id, street_number))
    # ────────────────────────────────────────────────────────────────────────────
    street_groups = defaultdict(list)
    for idx, row in merged.iterrows():
        key = (row["entity_id"], row["street_number"])
        street_groups[key].append(idx)

    for key, idxs in street_groups.items():
        norms = [merged.at[i, "clean_street"] for i in idxs]
        mapping = cluster_names(norms)
        for i in idxs:
            orig = merged.at[i, "clean_street"]
            rep = mapping.get(orig, orig)
            if rep != orig:
                merged.at[i, "clean_street"] = rep
                changed_street.add(i)

    # ────────────────────────────────────────────────────────────────────────────
    # RULE 3: City‑Fill via pandas groupby + merge
    # ────────────────────────────────────────────────────────────────────────────
    city_key_cols = ["entity_id", "street_number", "clean_street", "clean_zip"]

    nonblank_cities = merged.loc[merged["clean_city"] != "", :]
    city_nunique = (
        nonblank_cities
        .groupby(city_key_cols)["clean_city"]
        .nunique()
        .rename("city_unique_count")
    )
    city_single_val = (
        nonblank_cities
        .groupby(city_key_cols)["clean_city"]
        .first()
        .rename("city_single_val")
    )

    merged = (
        merged
        .merge(city_nunique.reset_index(), on=city_key_cols, how="left")
        .merge(city_single_val.reset_index(), on=city_key_cols, how="left")
    )

    mask_city_fill = merged["clean_city"].eq("") & (merged["city_unique_count"] == 1)
    merged.loc[mask_city_fill, "clean_city"] = merged.loc[mask_city_fill, "city_single_val"]
    changed_city.update(merged.index[mask_city_fill])

    merged.drop(columns=["city_unique_count", "city_single_val"], inplace=True)

    # ────────────────────────────────────────────────────────────────────────────
    # RULE 4: State‑Fill via pandas groupby + merge
    # ────────────────────────────────────────────────────────────────────────────
    state_key_cols = ["entity_id", "street_number", "clean_street", "clean_zip", "clean_city"]

    nonblank_states = merged.loc[merged["clean_state"] != "", :]
    state_nunique = (
        nonblank_states
        .groupby(state_key_cols)["clean_state"]
        .nunique()
        .rename("state_unique_count")
    )
    state_single_val = (
        nonblank_states
        .groupby(state_key_cols)["clean_state"]
        .first()
        .rename("state_single_val")
    )

    merged = (
        merged
        .merge(state_nunique.reset_index(), on=state_key_cols, how="left")
        .merge(state_single_val.reset_index(), on=state_key_cols, how="left")
    )

    mask_state_fill = merged["clean_state"].eq("") & (merged["state_unique_count"] == 1)
    merged.loc[mask_state_fill, "clean_state"] = merged.loc[mask_state_fill, "state_single_val"]
    changed_state.update(merged.index[mask_state_fill])

    merged.drop(columns=["state_unique_count", "state_single_val"], inplace=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Build “clean_full_address”
    # ────────────────────────────────────────────────────────────────────────────
    merged["clean_full_address"] = merged.apply(build_full_address, axis=1)

    # ────────────────────────────────────────────────────────────────────────────
    # OPTIONAL: Split address_id into sub‐IDs if one raw AID → multiple clean_full_address
    # ────────────────────────────────────────────────────────────────────────────
    addr_groups = defaultdict(set)
    for idx, row in merged.iterrows():
        addr_groups[row["address_id"]].add(row["clean_full_address"])

    new_id_map = {}
    for old_aid, clean_set in addr_groups.items():
        if len(clean_set) == 1:
            only_addr = next(iter(clean_set))
            new_id_map[(old_aid, only_addr)] = old_aid
        else:
            suffix = ord("a")
            for clean_addr in sorted(clean_set):
                new_subid = f"{old_aid}_{chr(suffix)}"
                new_id_map[(old_aid, clean_addr)] = new_subid
                suffix += 1

    merged["new_address_id"] = merged.apply(
        lambda r: new_id_map[(r["address_id"], r["clean_full_address"])],
        axis=1
    )
    merged["old_address_id"] = merged["address_id"]

    split_changed = merged.index[merged["old_address_id"] != merged["new_address_id"]].tolist()

    # ────────────────────────────────────────────────────────────────────────────
    # PRINT “Address‑Split” SAMPLE (up to SAMPLE_SIZE distinct base IDs)
    # ────────────────────────────────────────────────────────────────────────────
    split_groups = defaultdict(list)
    for idx in split_changed:
        base = merged.at[idx, "old_address_id"]
        split_groups[base].append(idx)

    print(f"\n=== ADDRESS‑SPLIT SAMPLES (up to {SAMPLE_SIZE} base IDs) ===")
    if split_groups:
        chosen_bases = random.sample(
            list(split_groups.keys()),
            min(len(split_groups), SAMPLE_SIZE)
        )
        for base in chosen_bases:
            print(f"\nBase AID: {base}")
            for i in split_groups[base]:
                new_aid = merged.at[i, "new_address_id"]
                addr    = merged.at[i, "clean_full_address"]
                print(f" Row {i}: OLD AID→{base}, NEW AID→{new_aid}, Addr→{addr}")
    else:
        print(" No address_id needed splitting.")

    # Replace old address_id with new split IDs
    merged.drop(columns=["address_id"], inplace=True)
    merged.rename(columns={"new_address_id": "address_id"}, inplace=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Save to cleaned_addresses.csv (now with split IDs)
    # ────────────────────────────────────────────────────────────────────────────
    output_cols = [
        "entity_id", "address_id",
        "num1_c", "streetName_c", "streetSuffix_c",
        "streetDirectionalPre_c", "streetDirectionalPost_c",
        "unit_c", "city_c", "state_c", "zip_c",
        "clean_street", "clean_city", "clean_state",
        "clean_zip", "clean_full_address"
    ]
    merged.to_csv(OUTPUT_CSV, columns=output_cols, index=False)
    print(f"\nDone with cleaning + splitting → saved '{OUTPUT_CSV}'")

    # ────────────────────────────────────────────────────────────────────────────
    # Print summary & sample “before→after” for each rule
    # ────────────────────────────────────────────────────────────────────────────
    print("\n=== CHANGES SUMMARY ===")
    print(f"Rule 1 (ZIP‑Fill):     {len(changed_zip)}")
    print(f"Rule 2 (Street‑Clust): {len(changed_street)}")
    print(f"Rule 3 (City‑Fill):    {len(changed_city)}")
    print(f"Rule 4 (State‑Fill):   {len(changed_state)}")
    print(f"Address‑Split:         {len(split_changed)}")

    print_rule_samples(merged, changed_zip,    "Rule 1 (ZIP‑Fill)",      "zip_c",          "clean_zip")
    print_rule_samples(merged, changed_street, "Rule 2 (Street‑Clust)", "streetName_c",   "clean_street")
    print_rule_samples(merged, changed_city,   "Rule 3 (City‑Fill)",    "city_c",         "clean_city")
    print_rule_samples(merged, changed_state,  "Rule 4 (State‑Fill)",   "state_c",        "clean_state")

    print("\nAll done!")

if __name__ == "__main__":
    main()
