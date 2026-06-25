import os
import requests
from datetime import datetime


DOCUMENTS_DIR = "documents"


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return summarized results."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(
            "https://api.duckduckgo.com/",
            params=params,
            headers=headers,
            timeout=10
        )
        data = response.json()

        results = []

        # Abstract (main summary)
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if not results:
            return f"No results found for: {query}"

        return "\n".join(results)

    except Exception as e:
        return f"Search error: {str(e)}"


def file_write(filename: str, content: str) -> str:
    """Write content to a .txt file in the documents/ directory."""
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
        if not safe_name.endswith(".txt"):
            safe_name += ".txt"

        filepath = os.path.join(DOCUMENTS_DIR, safe_name)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            f.write(content)

        return f"Saved to {filepath} ({len(content)} chars)"

    except Exception as e:
        return f"File write error: {str(e)}"


# Tool registry — maps tool names to callables and their schemas
TOOLS = {
    "web_search": {
        "fn": web_search,
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "query": "The search query string"
        }
    },
    "file_write": {
        "fn": file_write,
        "description": "Save text content to a .txt file in the documents/ folder. This makes it available to the RAG pipeline.",
        "parameters": {
            "filename": "Name of the file (without path)",
            "content": "The full text content to save"
        }
    }
}