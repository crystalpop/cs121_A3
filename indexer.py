import os
import json
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk import stem
import ijson
import heapq

DOC_ID_DICT = {}

#TODO: comment about time complexity and memory efficiency

"""
Our index structure:
Index = {
            'word1': [{docID: frequency}, {docID2: frequency}],
            'word2': [{docID2: frequency}, {docID3: frequency}]
}
"""

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
        self.partial_count = 1
        self.batch_size = 5000  # batch size, saved to global index every 5000 documents
        self.unique_tokens = 0  ##### ADDED A NEW PARAM TO MAKE TOKEN COUNTING EASIER 

        # initialize other parameters
        self.doc_count = 0  # keep track of number of documents
        self.inverted_index = defaultdict(dict) # temporary partial index, will get emptied every 5000 docs
    def load_json_files(self):
        """
        Load JSON files from dataset folder.
        """
        for root, _ , files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    # Step 2: extract raw HTML from json file
                    raw_html, url = self.extract_text_from_json(file_path)
                    # Step 3: clean HTML, extract text and tokenize
                    tokens = self.clean_text(raw_html)
                    # Step 4: add tokens to the global inverted index
                    self.build_inverted_index(tokens, doc_id=self.doc_count)

                    DOC_ID_DICT[self.doc_count] = url
                    self.doc_count += 1
                    
                    # save partial index every 5000 docs
                    if self.doc_count % self.batch_size == 0:
                        self.save_partial_index()
                        self.inverted_index.clear() # partial index cleared
        
        if self.inverted_index:
            self.save_partial_index()
            self.inverted_index.clear()
              

    def extract_text_from_json(self, file_path: str) -> str:
        """
        Extract raw HTML content from content field.
        :param file_path: Path to JSON file.
        :return: Extracted raw HTML.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                html = data['content']
                url = data['url']
                return (html, url)
        except Exception as e:
            print(e)
          

    def clean_text(self, raw_html: str) -> list:
        """
        Clean raw HTML, extract text, tokenize.
        :param raw_html: HTML content.
        :return: List of cleaned tokens.
        """
        # Beautiful soup should automatically handle broken html
        soup = BeautifulSoup(raw_html, 'lxml')
        # Get only the human-readable text
        text = soup.get_text()
        # Alphnum characters only for tokenizer
        tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
        tokens = tokenizer.tokenize(text)
        stems = []
        stemmer = stem.PorterStemmer()
        for token in tokens:
            stems.append(stemmer.stem(token))
        return stems




    def build_inverted_index(self, tokens: list, doc_id: int):
        """
        Build inverted index.
        :param tokens: List of processed words.
        :param doc_id: Unique document identifier.
        """
        # Loop through tokens and update inverted index
        for word in tokens:
            if word not in self.inverted_index:
                self.inverted_index[word] = [] # list of postings
                self.inverted_index[word].append({doc_id: 1}) # first posting
            else:
                # Check if the last posting corresponds to the current document.
                # Assuming that tokens come from one document at a time.
                last_posting = self.inverted_index[word][-1]
                if next(iter(last_posting)) == doc_id:
                    # Update existing posting: increment frequency and add new position.
                    last_posting[doc_id] += 1
                    # last_posting['positions'].append(position)
                else:
                    # Add a new posting for this document.
                    self.inverted_index[word].append({doc_id: 1})



    def save_partial_index(self):
        """
        Save partial indexes every batch_size documents.
        """
        # Save a partial index file and reset memory

        index_file = os.path.join(self.output_dir, f"index_part_{self.partial_count}.json")
        with open(index_file, "w") as f:
            # Sort keys before writing to file
            json.dump(self.inverted_index, f, sort_keys=True, indent=4)
        self.partial_count += 1
        print(f"Saved partial index: {index_file}")

    
    def merge_partial_indexes_json(self):
        """
        Merge all partial indexes into a final index.
        """
        # Read and combine multiple index files

        # merged_index = {}
        heap = []
        partial_index_files = [
        os.path.join(self.output_dir, fname)
        for fname in os.listdir(self.output_dir)
        if fname.endswith(".json")
        ]
        final_index_file = os.path.join(self.output_dir, "final_inverted_index.json")

            # Open each file and create an iterator over its key-value pairs.
        for file_idx, file_path in enumerate(partial_index_files):
            f = open(file_path, 'r')
            # Using ijson.kvitems to iterate over key-value pairs in the top-level JSON object.
            iterator = ijson.kvitems(f, '')
            try:
                token, postings = next(iterator)
                heapq.heappush(heap, (token, file_idx, postings, iterator, f))

            except StopIteration:
                # If the file is empty, close it.
                f.close()

        # Open the output file for writing the merged index.
        # We will write the JSON object manually in a streaming manner.
        with open(final_index_file, 'w') as out_f:
            out_f.write("{\n")
            first_entry = True

            while heap:
                # Pop the smallest token from the heap.
                token, file_idx, postings, iterator, f = heapq.heappop(heap)
                merged_postings = list(postings)  # start with this file's postings

                # Check if other partial indexes have the same token.
                while heap and heap[0][0] == token:
                    t, f_idx2, postings2, iterator2, f2 = heapq.heappop(heap)
                    merged_postings.extend(postings2)
                    try:
                        # Advance the iterator for this file and push its next token.
                        next_token, next_postings = next(iterator2)
                        heapq.heappush(heap, (next_token, f_idx2, next_postings, iterator2, f2))
                    except StopIteration:
                        f2.close()

                # After merging all postings for the current token,
                # push the next token from the current file's iterator.
                try:
                    next_token, next_postings = next(iterator)
                    heapq.heappush(heap, (next_token, file_idx, next_postings, iterator, f))
                except StopIteration:
                    f.close()

                self.unique_tokens += 1

                # Write the merged result for the token to the output file.
                # We need to handle commas between JSON key–value pairs.
                if not first_entry:
                    out_f.write(",\n")
                else:
                    first_entry = False
                out_f.write(json.dumps(token))            # Write the key as JSON string.
                out_f.write(": ")
                out_f.write(json.dumps(merged_postings))    # Write the merged postings.

            out_f.write("\n}\n")

        
        print(f"Merged final index saved at {final_index_file}")



    def compute_statistics(self):
        """
        Compute report statistics (total documents, unique words, index size).
        """
        # Count total docs, unique words, and index size

        total_documents = self.doc_count  # Number of docs processed
        unique_terms = self.unique_tokens  # Retrieved directly
        index_file_path = os.path.join("index_files", "final_inverted_index.json")  
        index_size = os.path.getsize(index_file_path) // 1024

        print(f"Total Documents: {total_documents}")
        print(f"Unique Terms: {unique_terms}")
        print(f"Total Index Size (KB): {index_size}") 

        with open(os.path.join(self.output_dir, "index_report.txt"), "w") as f:
            f.write(f"Total Documents: {total_documents}\n")
            f.write(f"Unique Terms: {unique_terms}\n")
            f.write(f"Total Index Size (KB): {index_size}")

        print("Index report saved successfully.")



if __name__ == "__main__":
    # Initialize Indexer
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASET_PATH = os.path.join(BASE_DIR, "DEV")
    OUTPUT_DIR = os.path.join(BASE_DIR, "index_files") 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indexer = Indexer(data_folder=DATASET_PATH, output_dir=OUTPUT_DIR)
    # Load and process JSON files
    indexer.load_json_files()

    # Merge partial indexes
    indexer.merge_partial_indexes_json()

    # Compute statistics and writes to a txt file 
    indexer.compute_statistics()
