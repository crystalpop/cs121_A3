import json
import os
import bisect
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
# import sys

class BooleanSearch:
    def __init__(self, index_file: str, metadata_file):
        """
        Initialize Boolean search with a single large index file sorted alphabetically.
        :param index_file: Path to large sorted JSON index.
        :param metadata_file: Path to document metadata JSON file.
        """
        self.index_file = index_file
        self.metadata = self.load_metadata(metadata_file)
        self.stemmer = PorterStemmer()
        self.term_offsets = self.build_term_offsets()  # Binary search mapping

    def load_metadata(self, metadata_file: str):
        """Load document metadata (doc_id → URL mapping)."""
        with open(metadata_file, "r") as f:
            return json.load(f)

    def build_term_offsets(self):
        """
        Read the index file and store offsets of each term for fast lookup.
        :return: Dictionary mapping terms to their file offset position.
        """
        term_offsets = {}
        with open(self.index_file, "r") as f:
            while True:
                offset = f.tell()  # Get file position
                line = f.readline().strip()
                if not line:
                    # print("NO LINE")
                    break
                # print(f"LINE: {line}")
                try:
                    # print(f"getting term data")
                    if line[-1] == ",":
                        term_data = json.loads("{" + line[:-1] + "}")
                        # print(f"term data: {term_data}")
                        term = next(iter(term_data.keys()))
                        term_offsets[term] = offset
                    else:
                        term_data = json.loads("{" + line + "}")
                        # print(f"term data: {term_data}")
                        term = next(iter(term_data.keys()))
                        term_offsets[term] = offset
                    
                    # print(f"term: {term}")
                except Exception as e:
                    # print(f"BUILD OFFSETS ERROR: {e}")
                    continue  # Ignore errors
        # print(f"term offsets: {term_offsets}")
        return term_offsets

    def binary_search_term(self, term):
        """
        Perform binary search in the sorted index file to retrieve term postings.
        :param term: The query term.
        :return: Posting list (dict of doc_id → positions) or empty dict if not found.
        """
        if term not in self.term_offsets:
            # print(f"{term} NOT IN OFFSETS")
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
        """Tokenize, lowercase, and stem query terms."""
        tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
        tokens = tokenizer.tokenize(query.lower())
        return [self.stemmer.stem(word) for word in tokens]  # Apply stemming

    def boolean_and_search(self, query: str):
        """
        Perform a Boolean AND search using binary search in the index.
        :param query: User query string.
        :return: List of top 5 URLs matching the query.
        """
        query_terms = self.preprocess_query(query)

        posting_lists = []
        # for term in query_terms:
        #     postings = self.binary_search_term(term)  # Load only needed term
        #     if postings:
        #         posting_lists.append(set(postings.keys()))  # Convert doc_ids to set

        for term in query_terms:
            postings_list = self.binary_search_term(term)  # Now returns a list of dicts
            if postings_list:
                term_doc_ids = set()
                for posting in postings_list:
                    term_doc_ids.update(posting.keys())
                posting_lists.append(term_doc_ids)

        # If no valid term exists, return empty result
        if not posting_lists:
            return []

        # Perform Boolean AND intersection
        common_docs = set.intersection(*posting_lists) if posting_lists else set()

        # Retrieve URLs from metadata
        results = [self.metadata[doc_id] for doc_id in common_docs if doc_id in self.metadata]

        return results[:5]  # Return top 5 results

if __name__ == "__main__":
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files/final_inverted_index.json")  # Large sorted index
    METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
    # Initialize search engine
    search_engine = BooleanSearch(INDEX_FILE, METADATA_FILE)

    # Example Queries
    queries = ["Iftekhar Ahmed", "machine learning", "ACM", "master of software engineering"]
    
    # for query in queries:
    #     print(f"\n🔍 Query: {query}")
    #     results = search_engine.boolean_and_search(query)
    #     if results:
    #         for i, url in enumerate(results):
    #             print(f"{i+1}. {url}")
    #     else:
    #         print("No results found.")

    print("Booting the search engine...\n")
    print('Type "exit" to quit\n' )
    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        results = search_engine.boolean_and_search(query)
        print(f"Results: \n")
        if results:
            for i, url in enumerate(results):
                print(f"{i+1}. {url}")
        else:
            print("No results found.")
        print("\n-----------------------------------------\n\n")


