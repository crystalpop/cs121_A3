import json
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import time
import math
import numpy as np

class SearchEngine:
    def __init__(self, index_file: str, metadata_file:str, term_offsets_file: str):
        """
        Initialize Boolean search with a single large index file sorted alphabetically.
        """
        start_time = time.time()
        self.index_file = index_file
        self.metadata = self.load_metadata(metadata_file)
        self.stemmer = PorterStemmer()

        if not os.path.exists(term_offsets_file):
            print("Building term offsets")
            self.term_offsets = self.build_term_offsets()  # Build and save term_offset
        else:
            with open(term_offsets_file, "r") as f:
                self.term_offsets = json.load(f)
        
        # Open index and keep it open
        self.index_file_obj = open(self.index_file, "r")

        end_time = time.time()
        init_time = (end_time - start_time) * 1000
        print(f"Search Engine initialized in {init_time:.2f} ms.")

    def __del__(self):
        # Ensure the file is closed when the SearchEngine object is deleted.
        if hasattr(self, 'index_file_obj') and not self.index_file_obj.closed:
            self.index_file_obj.close()

    def load_metadata(self, metadata_file: str):
        """ 
        Load document metadata (doc_id → URL mapping).
        Time complexity: O(n) where n is size of the file.
        """
        with open(metadata_file, "r") as f:
            return json.load(f)

    def build_term_offsets(self):
        """
        Read the index file and store offsets of each term for fast lookup.
        Time complexity: O(n) where n is the number of terms in the index.
        """
        term_offsets = {}
        with open(self.index_file, "r") as f:
            while True:
                offset = f.tell()  # Get file position
                line = f.readline().strip()
                if not line:
                    break
                try:
                    if line[-1] == ",":
                        term_data = json.loads("{" + line[:-1] + "}")
                        term = next(iter(term_data.keys()))
                        term_offsets[term] = offset
                    else:
                        term_data = json.loads("{" + line + "}")
                        term = next(iter(term_data.keys()))
                        term_offsets[term] = offset

                except Exception as e:
                    continue  # Ignore errors

        with open("term_offsets.json", "w") as f:
            json.dump(term_offsets, f)

        return term_offsets


    def binary_search_term(self, term):
        """
        Perform binary search in the sorted index file to retrieve term postings.
        Time complexity: O(1).
        """
        if term not in self.term_offsets:
            return {}  # Term not in index


        self.index_file_obj.seek(self.term_offsets[term])  # Move to the term position
        line = self.index_file_obj.readline()
        try:
            data = json.loads("{" + line.strip()[:-1] + "}")
            posting_list = next(iter(data.values()))
            return posting_list
        except:
            return {}


    def preprocess_query(self, query: str):
        """
        Tokenize, lowercase, and stem query terms.
        Time complexity: O(n) where n is the number of terms in the query.
        """
        tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
        tokens = tokenizer.tokenize(query.lower())
        return [self.stemmer.stem(word) for word in tokens]

    # def boolean_and_search(self, query_terms: str):
    #     """
    #     Perform a Boolean AND search using binary search in the index.
    #     Time complexity: O(t * p) where t is the number of terms in the query and p is the size of the posting list for that term.
    #     """

    #     posting_lists = []

    #     for term in query_terms:
    #         postings_list = self.binary_search_term(term)  # Now returns a list of dicts
    #         if postings_list:
    #             term_doc_ids = set()
    #             for posting in postings_list:
    #                 term_doc_ids.update(posting.keys())
    #             posting_lists.append(term_doc_ids)

    #     # If no valid term exists, return empty result
    #     if not posting_lists:
    #         return []

    #     # Perform Boolean AND intersection
    #     common_docs = set.intersection(*posting_lists) if posting_lists else set()

    #     # Retrieve URLs from metadata
    #     # results = [self.metadata[doc_id] for doc_id in common_docs if doc_id in self.metadata]
    #     results = [doc_id for doc_id in common_docs]
    #     return results


    def compute_cosine_similarity(self, query_terms):
        """
        Computes the tf-idf vectors for the query and the documents, then
        computes the cosine similarity scores for each document.
        Tiime complexity: O(m * N) where m is the number of query terms,
        and N is the total number of documents (in case every document is relevant).
        """
        query_terms_set = set(query_terms)
        doc_vectors = {}  # doc_id -> tfidf vector
        query_tfidf = {}  # Query term -> weighted tf idf in query

        for term in query_terms_set:
            postings = self.binary_search_term(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((len(self.metadata)) / (df + 1))  
            tf_query = (1 + math.log(1 + (query_terms.count(term)))) / (1+ math.log(len(query_terms)))  # log scaled TF
            query_tfidf[term] = tf_query * idf

            for posting in postings:
                doc_id = next(iter(posting))
                tfidf_doc = posting[doc_id][2]

                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = {}

                doc_vectors[doc_id][term] = tfidf_doc

        scores = {}

        # Document must contain all the query terms
        threshold = len(query_tfidf)

        query_norm = math.sqrt(sum(weight**2 for weight in query_tfidf.values()))

        for doc_id, term_weights in doc_vectors.items():
            # Count how many query terms are present in the document
            common_terms_count = sum(1 for term in query_tfidf if term in term_weights)
            if common_terms_count < threshold:
                scores[doc_id] = 0
                continue

            doc_norm = math.sqrt(sum(weight**2 for weight in term_weights.values()))
            dot_product = sum(query_tfidf.get(term, 0) * term_weights.get(term, 0) for term in query_tfidf)
            if doc_norm > 0 and query_norm > 0:
                cosine_sim = dot_product / (query_norm * doc_norm)
            else:
                cosine_sim = 0

            scores[doc_id] = cosine_sim

        return scores
    

    def ranked_search(self, query: str, k: int):
        """Perform ranked search using cosine similarity"""
        start_time = time.time()
        query_terms = self.preprocess_query(query)
        scores = self.compute_cosine_similarity(query_terms)
        end_time = time.time()
        print(f"\nSearch completed in {(end_time - start_time) * 1000} ms.\n")
        # Sort by descending cos sim
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top k scores
        return [(self.metadata[doc_id], score) for doc_id, score in ranked_results[:k]]




if __name__ == "__main__":
    # File paths
    print("Booting the search engine...\n")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files/final_inverted_index.json")
    METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
    TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")
    # Initialize search engine
    search_engine = SearchEngine(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

    
    print('\nType "exit" to quit\n' )
    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            print("\nGoodbye!\n")
            break
        
        results = search_engine.ranked_search(query, 5)

        print(f"\nResults: \n")
        if results:
            for i, url in enumerate(results):
                print(f"{i+1}. {url}")
        else:
            print("No results found.")
        print("\n-----------------------------------------\n\n")
