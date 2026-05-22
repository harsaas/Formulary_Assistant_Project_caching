import json
import os
from pathlib import Path
import pandas as pd

def parse_local_fda_file():
    """Parses the bulk downloaded openFDA NDC JSON file from the data folder."""
    expected_path = Path("data") / "drug-ndc-0001-of-0001.json"
    print(f"Parsing local openFDA file: {expected_path}")

    if not expected_path.exists():
        raise FileNotFoundError(
            "Missing downloaded data. Expected either a file at 'data/drug-ndc-0001-of-0001.json' "
            "or a folder 'data/drug-ndc-0001-of-0001.json/' containing that JSON file."
        )

    # Some downloads/extractions produce a directory named *.json that contains the real file.
    if expected_path.is_dir():
        file_path = expected_path / expected_path.name
    else:
        file_path = expected_path

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Found '{expected_path}', but could not find a JSON file at '{file_path}'."
        )

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # FDA data in to 'results' array
    results = data.get('results', [])
    cache_records = []
    print(f"🔄 Processing {len(results)} raw FDA entries...")
    for drug in results:
        product_ndc = drug.get("product_ndc", "Unknown")
        brand_name = drug.get("brand_name", "Unknown")
        generic_name = drug.get("generic_name", "Unknown")
        dosage_form = drug.get("dosage_form", "Unknown")

        # extract nested OpenFDA array elements for RXCUI
        openfda_data = drug.get("openfda", {})
        rxcui_list = openfda_data.get("rxcui", ["None"])
        rxcui = rxcui_list[0] if rxcui_list else "None"

        pharm_classes = drug.get("pharm_class", [])
        primary_class = pharm_classes[0] if pharm_classes else "Unknown Class"


        query_context = (
            f"What is the alternative for {brand_name} {dosage_form}? "
            f"Generic profile: {generic_name}. Class: {primary_class}."
        )

        # Store this row payload in your cache dataset
        cache_records.append({
            "plan_id": "CHOICE_DUMMY_PLAN",  # Placeholder for plan context
            "ndc": product_ndc,
            "rxcui": rxcui,
            "user_query": query_context, 
            "drug_key": brand_name,  # key for the rapidfuzz layer
            "approved_response": f"Switch to Generic {generic_name} (Tier 1 Preferred)"
        })
    # Drop duplicates to keep our cache memory clean
    cache_df = pd.DataFrame(cache_records).drop_duplicates(subset=['user_query'])
    
    cache_output = os.path.join("data", "real_fda_cache.csv")
    cache_df.to_csv(cache_output, index=False)
    print(f"✅ Extracted {len(cache_df)} unique real-world drug formulas into '{cache_output}'")
    return cache_df

def create_real_world_validation_set(cache_df):
    """Generates evaluation traffic matching against the pulled FDA data to test pipeline thresholds."""
    validation_data = []
    
    # Select the first few items from the real data to generate realistic hits/misses
    for idx, row in cache_df.head(50).iterrows():
        drug = row['drug_key']
        
        # 1. Semantic Hit (Varying sentence style, correct drug name)
        validation_data.append({
            "query": f"Can you give me a substitute option for {drug} because it's non-covered?",
            "ground_truth": "Hit"
        })
        
        # 2. Fuzzy Hit (Injecting a typo into the real FDA brand name to test rapidfuzz)
        if len(drug) > 4:
            drug_list = list(drug)
            drug_list[2], drug_list[3] = drug_list[3], drug_list[2]  # Swap characters
            typo_drug = "".join(drug_list)
        else:
            typo_drug = drug + drug[-1]  # Double tail letter if very short
            
        validation_data.append({
            "query": f"Alternative for {typo_drug}?",
            "ground_truth": "Hit"
        })
        
        # 3. Dangerous Dosage / Formulation Changes (Must trip a Cache MISS via Cross-Encoder)
        # Using generic placeholders that will structurally look close but mean entirely different things
    validation_data.extend([
            {"query": "What is the alternative for Lipitor 80mg?", "ground_truth": "Miss"},
            {"query": "Alternative drug for Adderall IR 10mg", "ground_truth": "Miss"}
    ])
    
    validation_output = os.path.join("data", "real_fda_validation.csv")
    pd.DataFrame(validation_data).to_csv(validation_output, index=False)
    print(f"🎯 Evaluation dataset generated successfully at '{validation_output}'")

if __name__ == "__main__":
    # Test file processing locally
    df = parse_local_fda_file()
    create_real_world_validation_set(df)