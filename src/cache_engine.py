import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from rapidfuzz import process, fuzz

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

class AdvancedCache:
    def __init__(self):
        #define the encoders and thresholds for the cache layers
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.bi_threshold = float(os.getenv("BI_THRESHOLD", 0.75))
        self.cross_threshold = float(os.getenv("CROSS_THRESHOLD", 0.0))
        self.fuzzy_cutoff = float(os.getenv("FUZZY_SCORE_CUTOFF", 85))

        self.fuzzy_dictionary = {}
        # define the redis cache schema

        schema_dict = {
            "index": {"name": "Advanced_pbm_cache", "prefix": "pbm_prod"},
            "fields": [
                {"name": "plan_id", "type": "tag"},
                {"name": "ndc", "type": "tag"},
                {"name": "rxcui", "type": "tag"},
                {"name": "user_query", "type": "text"},
                {"name": "drug_key", "type": "tag"},
                {"name": "approved_response", "type": "text"},
                {"name": "query_vector", "type": "vector", "attrs": {
                    "dims": 384, "algorithm": "flat", "distance_metric": "cosine"
                }}
            ]
        }

        redis_url = f"redis://:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
        self.cache_index = SearchIndex(schema=IndexSchema(**schema_dict), redis_url=redis_url)

        if not self.cache_index.exists():
            self.cache_index.create(overwrite=True)
    # Seed cache layer from day 0 to show some real-world hits/misses and to test the end-to-end pipeline with actual FDA data. This will also help us tune our thresholds before we start generating synthetic data.
    def seed_semantic_cache(self, df):
        """Populates both the Redis Vector Index and the local Layer 1 Fuzzy Dictionary."""
        print("Delete old cache contents...")
        self.cache_index.clear()
        self.fuzzy_dictionary.clear()
        
        records = []
        print(f" Generating embeddings for {len(df)} real openFDA entries...")
        
        for idx, row in df.iterrows():
            # Seed Layer 1 Fuzzy Mapping
            self.fuzzy_dictionary[str(row['drug_key'])] = str(row['approved_response'])
            
            # Seed Layer 2 Vector encoding for Redis
            vector = self.bi_encoder.encode(str(row['user_query'])).tolist()
            vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
            records.append({
                "plan_id": str(row['plan_id']),
                "ndc": str(row['ndc']),
                "rxcui": str(row['rxcui']),
                "user_query": str(row['user_query']),
                "drug_key": str(row['drug_key']),
                "approved_response": str(row['approved_response']),
                "query_vector": vector_bytes
            })
            
        self.cache_index.load(records)
        print("Redis Caching Layers built successfully.")

    def check_fuzzy_match(self, raw_query):
        """Fuzzy matching layer to catch the typo errors and common misspellings in drug names."""
        words = raw_query.split()
        for word in words:
            match = process.extractOne(
                word, 
                self.fuzzy_dictionary.keys(), 
                scorer=fuzz.token_sort_ratio, 
                score_cutoff=self.fuzzy_cutoff
            )
            if match:
                matched_drug = match[0]
                return True, self.fuzzy_dictionary[matched_drug]
        return False, None
    

    def evaluate_request(self, target_plan, text_query, use_fuzzy: bool = True):
        """Evaluation routing pipeline through three process"""
        # --- LAYER 1: FUZZY LAYER  ---
        if use_fuzzy:
            fuzzy_hit, fuzzy_payload = self.check_fuzzy_match(text_query)
            if fuzzy_hit:
                return "Fuzzy Cache Hit", fuzzy_payload

        # --- LAYER 2: BI-ENCODER + REDIS COSINE VECTOR SEARCH ---
        query_vec = self.bi_encoder.encode(text_query).tolist()
        plan_filter = Tag("plan_id") == target_plan
        #Building the vector query for Redis search with the appropriate filter to ensure we are only comparing against relevant plan context in our cache. This is crucial to avoid cross-plan contamination in our cache hits. The Redis Vector Search will return the most semantically similar cached query along with its approved response for further verification in the next layer.
        v_query = VectorQuery(
            vector=query_vec, #incoming vector embedded from text query
            vector_field_name="query_vector", #Redis stored vector field to compare against
            return_fields=["user_query", "approved_response"], #After match found return these fields for cross-encoder verification
            num_results=1, #Single best match for verification in layer 3
            filter_expression=plan_filter #Filter to ensure we are only matching against the relevant plan context in the cache
        )
        #Pass the vector query build above to search in cache_index and get the most similar cached query and its approved response for the next layer of verification. If no results are returned, it means we had a cache miss at the vector search layer, and we can immediately route to LLM fallback without needing to do cross-encoder verification.
        results = self.cache_index.query(v_query)
        if not results:
            return "Cache Miss (Unindexed Drug Route)", "LLM_FALLBACK"
            
        matched_doc = results[0]
        #Convert the vector distnace returned by Redis into a similarity score. Since Redis returns distance, we subtract from 1 to get similarity (assuming cosine distance where 0 means identical and 1 means completely different). This similarity score will be used to determine if we even want to proceed to the cross-encoder verification step or if we should immediately consider it a cache miss due to low similarity.
        bi_similarity = 1 - float(matched_doc['vector_distance'])
        
        # threshold check to determine if we should even trust this match enough to send it to the cross-encoder for verification. If the bi-encoder similarity is below our defined threshold, we can consider this a cache miss and route to LLM fallback without needing to do the more expensive cross-encoder step. This helps us maintain high precision in our cache hits by ensuring that only sufficiently similar matches are considered for final approval.
        if bi_similarity < self.bi_threshold:
            return f"Cache Miss (Low Similarity: {bi_similarity:.2f})", "LLM_FALLBACK"
            
        # --- LAYER 3: CROSS-ENCODER  ---
        cross_score = self.cross_encoder.predict([text_query, matched_doc['user_query']])
        
        if cross_score >= self.cross_threshold:
            return f"Semantic Cache Hit (Verified Score: {cross_score:.2f})", matched_doc['approved_response']
        else:
            return f"Cache Miss (Cross-Encoder Rejected Dosage/Nuance: {cross_score:.2f})", "LLM_FALLBACK"

