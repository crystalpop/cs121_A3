from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from A3part2 import BooleanSearch
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "http://localhost:3000"}})

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index_files/final_inverted_index.json")
METADATA_FILE = os.path.join(BASE_DIR, "docID_dict.json")
TERM_OFFSETS_FILE = os.path.join(BASE_DIR, "term_offsets.json")

# Initialize the Boolean Search Engine
search_engine = BooleanSearch(INDEX_FILE, METADATA_FILE, TERM_OFFSETS_FILE)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    retrieval_start_time = time.time()  # Start retrieval timing
    results = search_engine.boolean_and_search(query)
    retrieval_end_time = time.time()  # Stop retrieval timing

    retrieval_time = (retrieval_end_time - retrieval_start_time) * 1000  # Convert to milliseconds
    print(f"Retrieval time: {retrieval_time:.2f} ms.")  # Print retrieval time in terminal

    return jsonify({
        "query": query,
        "results": results,
        "retrieval_time": retrieval_time  # Include retrieval time in API response
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
