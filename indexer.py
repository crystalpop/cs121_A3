import os
import json
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

class Indexer:
    def __init__(self, data_folder: str, output_dir: str):
        """
        Initialize the indexer.
        :param data_folder: Path to dataset folder.
        :param output_dir: Directory to store partial indexes.
        :param batch_size: Number of documents before writing partial index.
        """
        self.data_folder = data_folder 
        self.output_dir = output_dir
        self.batch_size = 5000  # batch size, saved to global index every 5000 documents
        self.unique_tokens = 0  ##### ADDED A NEW PARAM TO MAKE TOKEN COUNTING EASIER 

        # initialize other parameters
        self.doc_count = 0  # keep track of number of documents
        self.inverted_index = defaultdict(dict) # temporary partial index, will get emptied every 5000 docs
    def load_json_files(self):
        """
        Step 1: Load JSON files from dataset folder.
        """
        for root, _ , files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    # Step 2: extract raw HTML from json file
                    raw_html = self.extract_text_from_json(file_path)
                    print(f"Processing file: {file_path}")
                    # Step 3: clean HTML, extract text and tokenize
                    tokens = self.clean_text(raw_html)
                    # Step 4: add tokens to the global inverted index
                    self.build_inverted_index(tokens, doc_id=file)

                    self.doc_count += 1
                    
                    # save partial index every 5000 docs
                    if self.doc_count % self.batch_size == 0:
                        self.save_partial_index()
                        self.inverted_index.clear() # partial index cleared
              

    def extract_text_from_json(self, file_path: str) -> str:
        """
        Step 2: Extract raw HTML content from content field.
        :param file_path: Path to JSON file.
        :return: Extracted raw HTML.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_html = data.get("content", "")  # Extract HTML
                return raw_html
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def clean_text(self, raw_html: str) -> list:
        """
        Step 3: Clean raw HTML, extract text, tokenize.
        :param raw_html: HTML content.
        :return: List of cleaned tokens.
        """
        soup = BeautifulSoup(raw_html, "lxml")

        # Extract text from HTML
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()

        # Tokenization
        words = word_tokenize(text.lower())

        # Remove non-alphanumeric characters & stopwords, then apply stemming
        tokens = [ps.stem(w) for w in words if w.isalnum() and w not in stop_words]

        return tokens

    def build_inverted_index(self, tokens: list, doc_id: str):
        """
        Step 4: Build inverted index.
        :param tokens: List of processed words.
        :param doc_id: Unique document identifier.
        """
          # Loop through tokens and update inverted index

        for position, word in enumerate(tokens):
            if word not in self.inverted_index:
                self.inverted_index[word] = {}
            if doc_id not in self.inverted_index[word]:
                self.inverted_index[word][doc_id] = []
            self.inverted_index[word][doc_id].append(position)

    """ 
    doc_1 doc id with tokens = ["hello", "world", "hello", "python"] 
    {
    "hello": {"doc_1": [0, 2]},
    "world": {"doc_1": [1]},
    "python": {"doc_1": [3]}
    }
    """


    def save_partial_index(self):
        """
        Step 5: Save partial indexes every batch_size documents.
        """
        # Save a partial index file and reset memory

        index_file = os.path.join(self.output_dir, f"index_part_{self.doc_count // self.batch_size}.json")
        with open(index_file, "w") as f:
            json.dump(self.inverted_index, f, indent=4)
        print(f"Saved partial index: {index_file}")

    
    def merge_partial_indexes_json(self):
        """
        Step 6: Merge all partial indexes into a final index.
        """
        # Read and combine multiple index files
        merged_index = defaultdict(dict)
        index_files = [f for f in os.listdir(self.output_dir) if f.startswith("index_part_")]

        for index_file in index_files:
            index_path = os.path.join(self.output_dir, index_file)

            with open(index_path, "r") as f:
                partial_index = json.load(f)

                for word, doc_data in partial_index.items():
                    for doc_id, positions in doc_data.items():
                        if doc_id not in merged_index[word]:
                            merged_index[word][doc_id] = positions
                        else:
                            merged_index[word][doc_id].extend(positions)

        # Save final merged index
        final_index_path = os.path.join(self.output_dir, "final_inverted_index.json")
        with open(final_index_path, "w") as f:
            json.dump(merged_index, f, indent=4)

        self.unique_tokens = len(merged_index)  # Update token count
        print(f"Merged final index saved at {final_index_path}")

    def compute_statistics(self):
        """
        Step 7: Compute report statistics (total documents, unique words, index size).
        """
        # Count total docs, unique words, and index size

        total_documents = self.doc_count  # Number of docs processed
        unique_terms = self.unique_tokens  # Retrieved directly
        index_size = sum(os.path.getsize(os.path.join(self.output_dir, f)) 
                     for f in os.listdir(self.output_dir) if f.endswith(".json")) // 1024

        print(f"Total Documents: {total_documents}")
        print(f"Unique Terms: {unique_terms}")
        print(f"Total Index Size (KB): {index_size}") 

        with open(os.path.join(self.output_dir, "index_report.txt"), "w") as f:
            f.write(f"Total Documents: {total_documents}\n")
            f.write(f"Unique Terms: {unique_terms}\n")
            f.write(f"Total Index Size (KB): {index_size}")

        print("Index report saved successfully.")

    # def generate_report(self): --> Doing this in previous func itself 
        """
        Step 8: Generate a report with index statistics.
        """
         # Save report to a text file


if __name__ == "__main__":
    # Step 1: Initialize Indexer
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASET_PATH = os.path.join(BASE_DIR, "DEV")
    OUTPUT_DIR = os.path.join(BASE_DIR, "index_files") 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indexer = Indexer(data_folder=DATASET_PATH, output_dir=OUTPUT_DIR)
    # Step 2: Load and process JSON files
    indexer.load_json_files()

    # Step 6: Merge partial indexes
    indexer.merge_partial_indexes_json()

    # Step 7: Compute statistics and writes to a txt file 
    indexer.compute_statistics()

    # Step 8: Generate final report
    # indexer.generate_report()