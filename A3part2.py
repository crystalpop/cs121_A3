import json
import os
import bisect
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import time
# import sys

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

    def build_term_offsets(self):
        """
        Read the index file and store offsets of each term for fast lookup.
        :return: Dictionary mapping terms to their file offset position.
        """
        term_offsets = {}
        index_of_index = {}
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
                    
                    # Store first-letter-based offset for faster access
                    first_letter = term[0].lower()
                    if first_letter not in index_of_index:
                        index_of_index[first_letter] = offset
                    
                    # print(f"term: {term}")
                except Exception as e:
                    # print(f"BUILD OFFSETS ERROR: {e}")
                    continue  # Ignore errors
        # print(f"term offsets: {term_offsets}")

        # Save the "Index of Index" for fast lookups
        with open("index_of_index.json", "w") as f:
            json.dump(index_of_index, f)

        with open("term_offsets.json", "w") as f:
            json.dump(term_offsets, f)
        
        return term_offsets

    def binary_search_term(self, term):
        """
        Perform binary search in the sorted index file to retrieve term postings.
        :param term: The query term.
        :return: Posting list (dict of doc_id → positions) or empty dict if not found.
        """

        first_letter = term[0].lower()
        # Load Index of Index
        with open("index_of_index.json", "r") as f:
            index_of_index = json.load(f)

        if first_letter in index_of_index:
            start_offset = index_of_index[first_letter]  # Get section start
        else:
            print(f"{term} NOT IN INDEX")
            return {}

        with open(self.index_file, "r") as f:
            f.seek(start_offset)  # Jump to section
            while True:
                offset = f.tell()
                line = f.readline().strip()
                if not line:
                    break

                try:
                    # Parse term from JSON
                    term_data = json.loads("{" + line.rstrip(",") + "}")
                    current_term = next(iter(term_data.keys()))

                    if current_term == term:  # Found the term!
                        return term_data[current_term]

                    if current_term > term:  # Early exit if past the term alphabetically
                        break
                except:
                    continue  # Ignore errors

        print(f"{term} NOT IN OFFSETS")
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

        start_time = time.time()
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

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  
        print(f"Query executed in {execution_time:.2f} ms.")
        return results[:5]  # Return top 5 results

if __name__ == "__main__":
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "index_files/final_inverted_index.json")  # Large sorted index
    METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
    TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")
    # Initialize search engine
    search_engine = BooleanSearch(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

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

