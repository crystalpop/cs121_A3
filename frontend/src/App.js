import React, { useState } from "react";
import "./App.css";

function App() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [retrievalTime, setRetrievalTime] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setResults([]);

        try {
            const response = await fetch(`http://127.0.0.1:5001/search?q=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error("Failed to fetch results");

            const data = await response.json();
            setResults(data); // Directly storing results
        } catch (err) {
            setError("Error fetching search results. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    // Function to fetch the title of a webpage
    const fetchTitle = async (url) => {
        try {
            const response = await fetch(`https://corsproxy.io/?${encodeURIComponent(url)}`);
            const text = await response.text();
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, "text/html");
            return doc.title || url; // Return title if found, else return the URL
        } catch (error) {
            return url; // If fetching fails, fallback to showing the URL
        }
    };

    return (
        <div className="App">
            <h1>Search Engine</h1>
            <form onSubmit={handleSearch}>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your search query..."
                    required
                />
                <button type="submit">Search</button>
            </form>

            {retrievalTime && <p>Retrieval time: <strong>{retrievalTime} ms</strong></p>}
            {loading && <p>Loading...</p>}
            {error && <p style={{ color: "red" }}>{error}</p>}

            <div className="results">
                {results.length > 0 ? (
                    <ul>
                        {results.map((url, index) => (
                            <li key={index} className="result-item">
                                <a href={url} target="_blank" rel="noopener noreferrer">
                                    <strong>{url}</strong>
                                </a>
                                <p className="url">{result.url}</p>
                            </li>
                        ))}
                    </ul>
                ) : (
                    !loading && <p>No results found.</p>
                )}
            </div>
        </div>
    );
}

export default App;
