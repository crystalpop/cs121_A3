import json
import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import time

import numpy as np

class SearchEngine:
    def __init__(self, index_file: str, metadata_file: str, term_offsets_file: str):
        """Initialize search engine with an inverted index and metadata."""
        start_time = time.time()
        self.index_file = index_file
        self.metadata = self.load_metadata(metadata_file)
        self.stemmer = PorterStemmer()

        if not os.path.exists(term_offsets_file):
            print("Building term offsets")
            self.term_offsets = self.build_term_offsets() # Build and save term_offset
        else:
            with open(term_offsets_file, "r") as f:
                self.term_offsets = json.load(f)

        end_time = time.time()
        print(f"Search Engine initialized in {(end_time - start_time) * 1000:.2f} ms.")

    def load_metadata(self, metadata_file: str):
        """Load document metadata (doc_id → URL mapping)."""
        with open(metadata_file, "r") as f:
            return json.load(f)

    def build_term_offsets(self):
        """Builds offsets for fast term lookup in the index file."""
        term_offsets = {}
        with open(self.index_file, "r") as f:
            while True:
                offset = f.tell() # Get file position
                line = f.readline().strip()
                if not line:
                    break
                try:
                    term_data = json.loads("{" + line.rstrip(",") + "}")
                    term = next(iter(term_data.keys()))
                    term_offsets[term] = offset
                except:
                    continue # Ignore errors

        with open("term_offsets.json", "w") as f:
            json.dump(term_offsets, f)
        return term_offsets

    def binary_search_term(self, term):
        """Retrieve postings for a term using binary search."""
        if term not in self.term_offsets:
            return {} # Term not in index

        with open(self.index_file, "r") as f:
            f.seek(self.term_offsets[term]) # Move to the term position
            line = f.readline().strip()
            try:
                data = json.loads("{" + line.rstrip(",") + "}")
                return next(iter(data.values()))
            except:
                return {}

    def preprocess_query(self, query: str):
        """Tokenizes and stems query terms """
        tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
        tokens = tokenizer.tokenize(query.lower())
        return [self.stemmer.stem(word) for word in tokens]

    def compute_cosine_similarity(self, query_terms):
        query_tfidf = {}
        doc_vectors = {}

        for term in query_terms:
            postings = self.binary_search_term(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((len(self.metadata)) / (df + 1))  
            tf_query = 1 + math.log(1 + query_terms.count(term))  # log scaled TF
            query_tfidf[term] = tf_query * idf

            for posting in postings:
                doc_id = next(iter(posting))
                tfidf_doc = posting[doc_id][1]  

                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = {}

                doc_vectors[doc_id][term] = tfidf_doc

        query_vec = np.array(list(query_tfidf.values()))
        scores = {}

        for doc_id, term_weights in doc_vectors.items():
            doc_vec = np.array([term_weights.get(term, 0) for term in query_tfidf])
            doc_norm = np.linalg.norm(doc_vec)
            query_norm = np.linalg.norm(query_vec)

            cosine_sim = (np.dot(query_vec, doc_vec) / (doc_norm * query_norm)) if doc_norm > 0 and query_norm > 0 else 0
            scores[doc_id] = cosine_sim

        return scores


    def ranked_search(self, query: str):
        """
        Perform ranked search using cosine similarity.
        """
        query_terms = self.preprocess_query(query)
        scores = self.compute_cosine_similarity(query_terms)
        # Sort by descending cos sim
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # get top 5 
        return [(self.metadata[doc_id], score) for doc_id, score in ranked_results[:5]]

if __name__ == "__main__":
    # File paths
    print("Booting the search engine...\n")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files2/final_inverted_index.json")
    METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
    TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")

    # Initialize search engine
    search_engine = SearchEngine(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

    print('Type "exit" to quit\n')
    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            print("\nGoodbye!\n")
            break

        results = search_engine.ranked_search(query)
        print("\nResults:\n")
        if results:
            for i, (url, score) in enumerate(results):
                print(f"{i+1}. {url} (Score: {score:.4f})")
        else:
            print("No results found.")
        print("\n-----------------------------------------\n\n")
