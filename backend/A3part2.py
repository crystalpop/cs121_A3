import json
import os
import time
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

class BooleanSearch:
    def __init__(self, index_file: str, metadata_file, term_offsets_file: str):
        """
        Initialize Boolean search with a single large index file sorted alphabetically.
        :param index_file: Path to large sorted JSON index.
        :param metadata_file: Path to document metadata JSON file.
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
                self.term_offset = json.load(f)

        end_time = time.time()
        init_time = (end_time - start_time) * 1000
        print(f"Search Engine initialized in {init_time:.2f} ms.")

    def load_metadata(self, metadata_file: str):
        """Load document metadata (doc_id → URL mapping)."""
        with open(metadata_file, "r") as f:
            return json.load(f)

    def preprocess_query(self, query: str):
        """Tokenize, lowercase, and stem query terms."""
        tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
        tokens = tokenizer.tokenize(query.lower())
        return [self.stemmer.stem(word) for word in tokens]  # Apply stemming

    def binary_search_term(self, term):
        """Perform binary search in the sorted index file to retrieve term postings."""
        if term in self.term_offset:
            with open(self.index_file, "r") as f:
                f.seek(self.term_offset[term])  # Jump to the exact position
                line = f.readline().strip()

                try:
                    term_data = json.loads("{" + line.rstrip(",") + "}")
                    return term_data.get(term, {})
                except:
                    print(f"Error parsing term: {term}")
                    return {}

        return {}

    def boolean_and_search(self, query: str):
        """Perform a Boolean AND search using binary search in the index."""
        start_time = time.time()
        query_terms = self.preprocess_query(query)

        posting_lists = []
        for term in query_terms:
            postings_list = self.binary_search_term(term)
            if postings_list:
                term_doc_ids = set()
                for posting in postings_list:
                    term_doc_ids.update(posting.keys())
                posting_lists.append(term_doc_ids)

        if not posting_lists:
            return []

        # Perform Boolean AND intersection
        common_docs = set.intersection(*posting_lists) if posting_lists else set()

        # Retrieve URLs from metadata
        results = [self.metadata[doc_id] for doc_id in common_docs if doc_id in self.metadata]

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"Query executed in {execution_time:.2f} ms.")
        return results[:5]  # Return top 5 results

if __name__ == "__main__":
    # File paths
    print("Booting the search engine...\n")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files/final_inverted_index.json")
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
        results = search_engine.boolean_and_search(query)
        print(f"\nResults: \n")
        if results:
            for i, url in enumerate(results):
                print(f"{i+1}. {url}")
        else:
            print("No results found.")
        print("\n-----------------------------------------\n\n")