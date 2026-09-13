# Insurance RAG Assistant — integrations

## Version 1:
 Blocking CLI. BM25 → RAG. Full answer, no stream

```bash
python main_integration.py
```

Sources: `clause` / `medical` / `case` (Enter = no filter). Type `exit` to quit.

## Version 1 path

```text
question
  → Redis exact cache → return
  → BM25 vs MySQL (threshold 0.85)
        hit  → cache, return canned answer
        miss → RAGSystem.generate_answer
                 BERT general_inquiry → LLM, no retrieve
                 BERT claims_adjudication → retrieve + generate
```

Switch is `IntegratedQASystem.query`. LLM is OpenAI `gpt-5-mini` via `call_dashscope`.

| Init fails | Still runs? |
| --- | --- |
| MySQL / Redis / BM25 | Yes — all queries go to RAG |
| VectorStore | Yes — generate with empty context |
| OpenAI | No — CLI cannot start |

v1 does **not** stream tokens and does **not** pick adjuster/case in the CLI.

Retrieval internals: [`RAG_README.md`](./RAG_README.md)
