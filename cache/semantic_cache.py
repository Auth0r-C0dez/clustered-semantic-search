import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, similarity_threshold=0.85):
        
        self.cache = []
        self.threshold = similarity_threshold
        
        self.hit_count = 0
        self.miss_count = 0

    def search(self, query_embedding, cluster_id):

        if len(self.cache) == 0:
            return None

        best_match = None
        best_score = 0

        for entry in self.cache:

            if entry["cluster"] != cluster_id:
                continue

            sim = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > best_score:
                best_score = sim
                best_match = entry

        if best_score >= self.threshold:
            self.hit_count += 1
            return best_match, best_score

        self.miss_count += 1
        return None

    def add(self, query, embedding, result, cluster_id):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster_id
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0
