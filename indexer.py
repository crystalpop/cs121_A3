import os
import json
from collections import defaultdict

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
        pass  

    def clean_text(self, raw_html: str) -> list:
        """
        Step 3: Clean raw HTML, extract text, tokenize.
        :param raw_html: HTML content.
        :return: List of cleaned tokens.
        """
        pass # stemmer, stopwords

    def build_inverted_index(self, tokens: list, doc_id: str):
        """
        Step 4: Build inverted index.
        :param tokens: List of processed words.
        :param doc_id: Unique document identifier.
        """
        pass  # Loop through tokens and update inverted index

    def save_partial_index(self):
        """
        Step 5: Save partial indexes every batch_size documents.
        """
        pass  # Save a partial index file and reset memory

    def merge_partial_indexes(self):
        """
        Step 6: Merge all partial indexes into a final index.
        """
        pass  # Read and combine multiple index files

    def compute_statistics(self):
        """
        Step 7: Compute report statistics (total documents, unique words, index size).
        """
        pass  # Count total docs, unique words, and index size

    def generate_report(self):
        """
        Step 8: Generate a report with index statistics.
        """
        pass  # Save report to a text file


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
    indexer.merge_partial_indexes()

    # Step 7: Compute statistics
    indexer.compute_statistics()

    # Step 8: Generate final report
    indexer.generate_report()
