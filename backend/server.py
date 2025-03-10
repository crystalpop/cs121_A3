from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from backend.A3part2 import BooleanSearch
import os
from A3part2 import SearchEngine  # Use SearchEngine instead of BooleanSearch

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "http://localhost:3000"}})

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index_files2/final_inverted_index.json")
METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")

# Ensure files exist before loading the search engine
if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE) or not os.path.exists(TERM_OFFSETS_FILE):
    raise FileNotFoundError("One or more required index files are missing in the backend deployment.")

# Initialize the Boolean Search Engine
search_engine = BooleanSearch(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])  # Return empty array if query is missing

    results = search_engine.boolean_and_search(query)  # Perform search
    print("Search Results:", results)

    return jsonify(results)  # Return only results

if __name__ == "__main__":
    app.run(debug=True, port=5001)
