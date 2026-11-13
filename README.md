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

## Evaluation

`eval/` holds 78 retrieval questions with gold sections, and a runner that scores the app's retriever against the real `chroma_db` (top 10, no classifier filter, no reranker).

| run | hit@1 | hit@5 | MRR@10 | wrong-act rate | p50 latency |
|---|---|---|---|---|---|
| before_hybrid | 0.24 | 0.67 | 0.417 | 7.4% | 254 ms |
| after_hybrid | 0.27 | 0.69 | 0.450 | 0.0% | 386 ms |
| after_bm25 | 0.03 | 0.08 | 0.056 | 0.0% | 238 ms |
| after_vector | 0.40 | 0.77 | 0.561 | 0.0% | 129 ms |
| after_bm25_tok | 0.09 | 0.23 | 0.142 | 0.0% | 154 ms |
| after_hybrid_tok | 0.28 | 0.74 | 0.467 | 0.0% | 286 ms |

Leaving out the mislabelled "Motor Vehicle Act, 1988" chunks removed every wrong-act result (7.4% of returned chunks before) and lifted hybrid MRR@10 from 0.417 to 0.450. The ablation then showed BM25 failing even on exact section lookups (hit@10 of 0.07 on questions like "What does section 303 of BNS say?"). The cause was the BM25 index splitting text on whitespace only, so "303" never matched "303." and "theft" never matched "Theft.—(1)". With a tokenizer that lowercases, keeps only letters and digits and drops a short stopword list (the `_tok` rows), BM25 alone went from 0.056 to 0.142 MRR@10 and hybrid from 0.450 to 0.467 (hit@5 0.69 to 0.74); the same tokenizer without stopword removal gave hybrid 0.447. Hybrid still trails vector search alone (MRR@10 0.467 against 0.561, hit@1 0.28 against 0.40).

Latency: on the same 78 queries with cached embeddings, the `$nin` filter that keeps the excluded act out takes a vector query from 1.2 ms to about 132 ms (p50), which is why the after hybrid runs are slower than before_hybrid.

Caveats: 78 questions, drafted from corpus chunks and hand-verified; one run each; latency is local retrieval only (query embeddings are cached, so the Gemini call is not counted); gold labels are sections, so an explainer on the right topic counts as a miss. Each results file also has scores without the 8 questions marked `known_issue`.

```bash
python eval/run_eval.py --mode hybrid --include-excluded --label before_hybrid
python eval/run_eval.py --mode hybrid --label after_hybrid_tok    # also --mode bm25 / vector
python eval/compare.py before_hybrid after_hybrid after_bm25 after_vector after_bm25_tok after_hybrid_tok
pytest tests/test_eval_metrics.py
```

`eval/REVIEW.md` lists every question with the text of its gold chunks (`python eval/make_review.py` rebuilds it).

## Known data issues

- **Mislabelled chunks.** 2,718 chunks are labelled "Motor Vehicle Act, 1988" but come from a Kaggle file that mixed many acts (IPC, CrPC, Evidence Act and others), and mostly carry no section number. Example: "Whoever commits theft shall be punished with imprisonment … three years" is filed under that name. They are excluded through `EXCLUDED_ACTS` in `config.py`.
- **BNSS heading over the wrong text.** Example: the chunk for BNSS s.58 has the heading "Person arrested not to be detained more than twenty-four hours" but the text of s.59 ("Police to report apprehensions").
- **Outdated text.** Hindu Succession Act s.6 is the pre-2005 version (the share passes by survivorship), and IT Act s.66 is the pre-2008 "Hacking with Computer System".
- **RTI Act numbering.** Section numbers in the metadata do not follow the act. Example: the chunk labelled s.7 holds the list of exempt information.
- **Heading-only chunks.** Example: Motor Vehicles Act s.183 holds only "Driving at excessive speed, etc.".
- **Chunks cut mid-URL.** Explainer chunks often start partway through a link. Example: a "Personal Photos/Videos of You" chunk starts with "/cybercrime.gov.in/.Share all collected evidence".
