import json
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import time
import numpy as np

class BooleanSearch:
    def __init__(self, index_file: str, metadata_file:str, term_offsets_file: str):
        """
        Initialize Boolean search with a single large index file sorted alphabetically.
        """
        start_time = time.time()
        self.index_file = index_file
        self.metadata = self.load_metadata(metadata_file)
        self.stemmer = PorterStemmer()
        # self.term_offsets = self.build_term_offsets()  # Binary search mapping

        if not os.path.exists(term_offsets_file):
            print("Building term offsets")
            self.term_offsets = self.build_term_offsets()  # Build and save term_offset
        else:
            with open(term_offsets_file, "r") as f:
                self.term_offsets = json.load(f)
        
        end_time = time.time()
        init_time = (end_time - start_time) * 1000
        print(f"Search Engine initialized in {init_time:.2f} ms.")

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
        index_of_index = {}
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

        with open(self.index_file, "r") as f:
            f.seek(self.term_offsets[term])  # Move to the term position
            line = f.readline()
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
    
    # def search(self, query: str):
    #     query_terms = self.preprocess_query(query)
    #     doc_vectors = {}  
    #     query_vector = {}  

    #     for term in query_terms:
    #         postings_list = self.binary_search_term(term)
    #         if postings_list:
    #             query_vector[term] = 1 

    #             for posting in postings_list:
    #                 doc_id, values = list(posting.items())[0]
    #                 tf_idf_score = values[1]

    #                 if doc_id not in doc_vectors:
    #                     doc_vectors[doc_id] = {}

    #                 doc_vectors[doc_id][term] = tf_idf_score

    #     doc_scores = {}
    #     query_norm = np.sqrt(sum(weight**2 for weight in query_vector.values()))  # ||Q||

    #     for doc_id, vector in doc_vectors.items():
    #         doc_norm = np.sqrt(sum(weight**2 for weight in vector.values()))  # ||D||
    #         dot_product = sum(query_vector.get(term, 0) * vector.get(term, 0) for term in query_vector)

    #         if doc_norm > 0:  
    #             doc_scores[doc_id] = dot_product / (query_norm * doc_norm)

    #     ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    #     return [self.metadata[doc_id] for doc_id, _ in ranked_results[:5] if doc_id in self.metadata]



    def search(self, query: str):
        query_terms = self.preprocess_query(query)
        doc_vectors = {}  # doc_id -> tfidf vector
        query_vector = {}  # Query term -> weighted tf idf in query

        for term in query_terms:
            postings_list = self.binary_search_term(term)
            if postings_list:
                query_tf = 1 + np.log(1 + query_terms.count(term))  
                query_vector[term] = query_tf  


                for posting in postings_list:
                    doc_id, values = list(posting.items())[0]
                    tf_idf_score = values[1]

                    if doc_id not in doc_vectors:
                        doc_vectors[doc_id] = {}

                    doc_vectors[doc_id][term] = tf_idf_score  

        filtered_doc_vectors = {doc_id: vector for doc_id, vector in doc_vectors.items() if any(term in vector for term in query_terms)}
        doc_scores = {}
        query_norm = np.sqrt(sum(weight**2 for weight in query_vector.values()))  # ||Q||

        for doc_id, vector in filtered_doc_vectors.items():
            doc_norm = np.sqrt(sum(weight**2 for weight in vector.values()))  # ||D||
            dot_product = sum(query_vector.get(term, 0) * vector.get(term, 0) for term in query_vector)

            if doc_norm > 0:  
                cosine_score = dot_product / (query_norm * doc_norm)

                url = self.metadata.get(doc_id, "")
                if any(char.isdigit() for char in url):
                    cosine_score *= 0.9  

                doc_scores[doc_id] = cosine_score

        min_similarity_threshold = 0.2  
        filtered_results = {doc_id: score for doc_id, score in doc_scores.items() if score >= min_similarity_threshold}

        ranked_results = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)

        return [self.metadata[doc_id] for doc_id, _ in ranked_results[:5] if doc_id in self.metadata]

    # def boolean_and_search(self, query: str):
    #     """
    #     Perform a Boolean AND search using binary search in the index.
    #     Time complexity: O(t * p) where t is the number of terms in the query and p is the size of the posting list for that term.
    #     """
    #     query_terms = self.preprocess_query(query)

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
    #     results = [self.metadata[doc_id] for doc_id in common_docs if doc_id in self.metadata]

    #     return results[:5]  # Return top 5 results


if __name__ == "__main__":
    # File paths
    print("Booting the search engine...\n")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files2/final_inverted_index.json")
    METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
    TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")
    # Initialize search engine
    search_engine = BooleanSearch(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

    
    print('Type "exit" to quit\n' )
    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            print("\nGoodbye!\n")
            break
        results = search_engine.search(query)
        print(f"\nResults: \n")
        if results:
            for i, url in enumerate(results):
                print(f"{i+1}. {url}")
        else:
            print("No results found.")
        print("\n-----------------------------------------\n\n")
