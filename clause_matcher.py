import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_clause_id

class LegalClauseMatcher:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}

    def get_embeddings(self, clauses: List[Dict], doc_name: str) -> np.ndarray:
        embeddings = []
        for clause in clauses:
            cid = generate_clause_id(clause['text'], doc_name)
            if cid in self.cache:
                embeddings.append(self.cache[cid])
            else:
                emb = self.embedder.encode(clause['text'])
                self.cache[cid] = emb
                embeddings.append(emb)
        return np.array(embeddings)

    def match_documents(self, doc1_clauses, doc2_clauses,
                        doc1_name="Document 1", doc2_name="Document 2",
                        match_threshold=0.75,
                        top_k=3):
        """
        Match clauses using embeddings only.
        - mutual best matching first
        - then greedy (top_k candidates per doc1 clause)
        All matches require similarity >= match_threshold.
        """
        start = time.time()
        print("Generating embeddings...")
        emb1 = self.get_embeddings(doc1_clauses, doc1_name)
        emb2 = self.get_embeddings(doc2_clauses, doc2_name)

        n1 = len(doc1_clauses)
        n2 = len(doc2_clauses)
        sim_matrix = cosine_similarity(emb1, emb2)

        matched_doc1 = [False] * n1
        matched_doc2 = [False] * n2
        match_pairs = []   # (i, j, sim, reason)

        # ----- 1. Mutual best matching -----
        print("Step 1: Mutual best matches...")
        best1 = np.argmax(sim_matrix, axis=1)
        best_sim1 = sim_matrix[np.arange(n1), best1]
        best2 = np.argmax(sim_matrix, axis=0)

        mutual_count = 0
        for i in range(n1):
            j = best1[i]
            sim = best_sim1[i]
            if best2[j] == i and sim >= match_threshold:
                if not matched_doc1[i] and not matched_doc2[j]:
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, 'Mutual best match'))
                    mutual_count += 1

        print(f"  → {mutual_count} mutual best matches")

        # ----- 2. Greedy matching (top-k) -----
        print(f"Step 2: Greedy matching (top {top_k} candidates)...")
        greedy_count = 0

        for i in range(n1):
            if matched_doc1[i]:
                continue

            # top-k candidates
            sims = sim_matrix[i, :]
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_sims = sims[top_indices]

            for j, sim in zip(top_indices, top_sims):
                if matched_doc2[j]:
                    continue
                if sim >= match_threshold:
                    matched_doc1[i] = True
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, f'Greedy match (≥{match_threshold})'))
                    greedy_count += 1
                    break

            # progress
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1}/{n1}")

        print(f"  → {greedy_count} greedy matches")

        # ----- 3. Build results -----
        matching_details = []
        only_in_doc1 = []
        match_map = {i: (j, sim, reason) for i, j, sim, reason in match_pairs}

        for i, clause1 in enumerate(doc1_clauses):
            if i in match_map:
                j, sim, reason = match_map[i]
                clause2 = doc2_clauses[j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': True,
                    'best_match': {
                        'clause_idx': j,
                        'similarity': sim,
                        'confidence': 1.0,
                        'reason': reason,
                        'key_differences': [],
                        'used_llm': False
                    },
                    'top_similarity': sim,
                    'top_match_idx': j
                })
            else:
                top_j = np.argmax(sim_matrix[i, :])
                top_sim = sim_matrix[i, top_j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': False,
                    'best_match': None,
                    'top_similarity': top_sim,
                    'top_match_idx': top_j
                })
                only_in_doc1.append({
                    'text': clause1['text'],
                    'number': clause1.get('number', str(i+1)),
                    'closest_match': doc2_clauses[top_j]['text'] if top_j < n2 else "",
                    'similarity': top_sim,
                    'metadata': clause1.get('metadata', {})
                })

        # Unmatched doc2
        only_in_doc2 = []
        for j in range(n2):
            if not matched_doc2[j]:
                sims = cosine_similarity([emb2[j]], emb1)[0]
                best_idx = np.argmax(sims)
                only_in_doc2.append({
                    'text': doc2_clauses[j]['text'],
                    'number': doc2_clauses[j].get('number', str(j+1)),
                    'closest_match': doc1_clauses[best_idx]['text'] if best_idx < n1 else "",
                    'similarity': sims[best_idx],
                    'metadata': doc2_clauses[j].get('metadata', {})
                })

        elapsed = time.time() - start
        matching_count = len(match_pairs)
        print(f"✅ Final: {matching_count} matches, total time: {elapsed:.2f}s")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': n1,
            'total_doc2': n2,
            'matching_count': matching_count,
            'processing_time': elapsed,
            'match_threshold': match_threshold,
            'llm_matches': 0,
            'high_sim_matches': matching_count,   # all matches are embedding-based
            'doc2_best_similarities': [0.0] * n2
        }