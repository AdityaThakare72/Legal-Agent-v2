# Legal Agent v2

A conversational RAG (Retrieval-Augmented Generation) assistant for Indian law — ask questions in natural language and get grounded, cited answers from a knowledge base of Indian legal acts and explainers.

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

## MCP server

`mcp_server.py` exposes the same hybrid retriever over MCP, so any MCP client can search the knowledge base.

- `search_indian_law` (tool): hybrid search, optional `category` / `act_name` filter, `top_k` 1 to 10. Returns `{"results": [...], "warning": ...}`; `warning` is null unless vector search failed and the results are keyword-only
- `list_acts` (tool): acts with chunk counts, gives the exact `act_name` to filter on
- `legal://categories` (resource): the category list

No LLM runs in the server. The classifier and reranker are skipped, the client model picks the filter and writes the answer. The only API call is the query embedding.

```bash
git lfs pull                          # chroma_db is stored with Git LFS
pip install -r requirements-mcp.txt
python mcp_server.py                  # stdio
```

Needs `GOOGLE_API_KEY` in `.env`. The first call is slow because it loads all chunks and builds the BM25 index.

Client config:

```json
{
  "mcpServers": {
    "indian-law": {
      "command": "python",
      "args": ["/path/to/Legal-Agent-v2/mcp_server.py"]
    }
  }
}
```

Tests need no API key, they use a tiny throwaway db:

```bash
pytest tests/test_mcp_server.py
```
