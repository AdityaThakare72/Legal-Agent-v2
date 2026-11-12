# Legal Agent v2

A conversational RAG (Retrieval-Augmented Generation) assistant for Indian law — ask questions in natural language and get grounded, cited answers from a knowledge base of Indian legal acts and explainers.

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

## MCP server

`mcp_server.py` exposes the same hybrid retriever (BM25 + vector search, fused with RRF) over the Model Context Protocol, so an MCP client such as Claude Desktop can search the knowledge base directly.

| Name | Kind | What it does |
| --- | --- | --- |
| `search_indian_law` | tool | Hybrid search. Optional `category` and `act_name` filters, `top_k` from 1 to 10. Returns chunks with act name, section number, title, category, source and text. |
| `list_acts` | tool | Lists acts with their chunk counts. Use it to get the exact `act_name` for a filtered search. |
| `legal://categories` | resource | The categories every chunk is tagged with. |

No LLM runs inside the server. The app's classifier and reranker are left out on purpose: the client's model chooses the filter and writes the answer. The only API call is the Gemini embedding of the query.

Setup (the database is stored with Git LFS, so run `git lfs pull` after cloning):

```bash
pip install -r requirements-mcp.txt
echo "GOOGLE_API_KEY=your-key" > .env
python mcp_server.py   # stdio transport
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "indian-law": {
      "command": "python",
      "args": ["/absolute/path/to/Legal-Agent-v2/mcp_server.py"]
    }
  }
}
```

The first tool call is slow, because it loads every chunk and builds the BM25 index. Later calls reuse it.

Tests use a small throwaway ChromaDB with fake embeddings, so they need no API key:

```bash
pip install pytest
pytest tests/test_mcp_server.py
```
