# RAG Notes (Education-Assistant)
Code:

- `integrated_qa_system/rag_qa/core/document_processor.py`
- `integrated_qa_system/rag_qa/edu_text_spliter/`
- `integrated_qa_system/rag_qa/core/vector_store.py`

---

## 1. Why dense-only RAG falls over on this corpus

Course material is not web paragraphs. It is a mix of PDF, PPT, scans, and Markdown, mostly Chinese, heavy on jargon.

What actually breaks:

1. Scanned PDFs have no selectable text. Skip OCR and there is nothing to index.
2. An English recursive splitter cuts on `. ! ?`. Chinese sentences either never split or split in the wrong place.
3. Dense embeddings catch paraphrases. Queries that must hit the original term (`违约责任`, `Transformer`) drift.
4. Returning a 300-char hit is too thin. The match is local; later generation has no section around it.

So this is not `embedding + top-k`. Retrieval unit and answer unit are split on purpose: **search on small chunks, return the parent as context**.

---

## 2. Stack (pipeline order, not a name dump)

| Stage | Choice | Role here |
| --- | --- | --- |
| Document object | LangChain `Document` | `page_content` + `metadata` through split, index, and recall |
| Loaders | Custom OCR loaders + LangChain loaders | `.pdf/.docx/.ppt/.pptx/.jpg/.png` → RapidOCR; `.txt/.md` → text loaders |
| OCR | RapidOCR (paddle first, onnxruntime fallback) | Text from images and scans; CPU / GPU paths |
| Rule split | `EduRecursiveTextSplitter` | Recurse on sentence punctuation: `。！？` / `. ! ?` → `；` / `;` → `，` / `,` |
| Structure split | `MarkdownTextSplitter` | `.md` by headings / paragraphs, not punctuation |
| Semantic split (optional) | `AliTextSplitter` + ModelScope `nlp_bert_document-segmentation_chinese-base` | DAMO document segmentation, [2107.09278](https://arxiv.org/abs/2107.09278); not on the main path |
| Embedding | **BGE-M3** | One forward pass → dense (1024-d) and sparse |
| Vector DB | **Milvus 2.4 standalone** (pymilvus) | Both vector types in one collection; hybrid search |
| Dense index | IVF_FLAT + IP, `nlist=128`, search `nprobe=10` | Semantic neighbors |
| Sparse index | SPARSE_INVERTED_INDEX + IP, `drop_ratio_build=0.2` | Keyword inverted index |
| Fusion | `WeightedRanker(1.0, 0.7)` | Dense leads; sparse boosts exact terms |
| Rerank | **bge-reranker-large** (CrossEncoder) | Score deduped parents; keep Top-M |
| Primary key | MD5 of child text | `upsert`; re-ingest does not duplicate |
| Filter | `source` scalar | Cut search space by subject (`ai` / `java` / …) |
| Device | MPS fp16 / CPU | Cast embeddings to `float32` before write or Milvus rejects them |

Defaults from `config.ini [retrieval]`:

- parent_chunk_size = **1200**
- child_chunk_size = **300**
- chunk_overlap = **50**
- retrieval_k = **5** (hybrid search returns 5 children)
- candidate_m = **2** (at most 2 parents after rerank)

---

## 3. Ingest: files → Document

`document_processor.load_documents_from_directory` walks the tree. Folder name `ai_data` becomes metadata `source = ai`. That field is the later filter.

Every successful load also sets:

- `file_path`
- `timestamp` (ISO)

A failed file logs an error and the rest of the directory still runs. One bad screenshot should not kill a whole subject dump.

OCR is a fallback, not multimodal understanding. Blurry slides produce dirty text; dirty text poisons recall.
<p><img src="./images/db_1.png" alt="docker container" width="700" /></p>
---

## 4. Chunk: parent / child, not one window

`process_documents` splits parents first, then children. Each child keeps the full parent:

```text
parent_id        = doc_{i}_parent_{j}
parent_content   = parent text
id               = {parent_id}_child_{k}
```

What gets indexed is the child. After a hit, `parent_content` rebuilds the parent; parents are deduped by that text.

Why:

- Child 300: sharper vectors. Terms, formulas, and headings are easier for hybrid search to hit.
- Parent 1200: rerank (and later generation) see a full subsection, not half a sentence.
- Overlap 50: sentences on the window edge still line up across adjacent children.

`.md` uses `MarkdownTextSplitter`. Everything else uses `EduRecursiveTextSplitter`. The recursive splitter walks paragraph → sentence → clause → comma until something actually cuts, then collapses extra blank lines. Separators stay in the chunk (`keep_separator=True`), so sentence punctuation is not stripped.

`AliTextSplitter` is in the package and not wired into the main path. Semantic splits follow paragraph boundaries better, but they pull in a BERT and are slow on CPU. Course files currently use rule-based splits; semantic split is a spare option.
<p><img src="./images/db_2.png" alt="docker container" width="700" /></p>
---

## 5. Index: BGE-M3 + two vector fields in Milvus

Collection: `edurag.edurag_final`. The database is created if missing.

Fields that actually affect retrieval:

| Field | Type | Use |
| --- | --- | --- |
| id | VARCHAR PK | child MD5 |
| text | VARCHAR | child body |
| dense_vector | FLOAT_VECTOR(1024) | BGE-M3 dense |
| sparse_vector | SPARSE_FLOAT_VECTOR | BGE-M3 term weights as `{index: weight}` |
| parent_id / parent_content | VARCHAR | rebuild parent after a hit |
| source | VARCHAR | subject filter |
| timestamp | VARCHAR | provenance |

One BGE-M3 encode gives dense and sparse, so there is no separate BM25 corpus to keep in sync. Sparse index build drops the bottom 20% of weights; the inverted list is cleaner on function words in slides.

Writes go through `upsert`. Re-running `vector_store.py` on the same files overwrites identical children instead of appending.

On Mac MPS, embeddings default to fp16. Milvus FLOAT_VECTOR wants `float32`. `add_documents` does `np.asarray(..., dtype=np.float32)` and casts sparse index/value to Python `int` / `float`. That is a real type error, not a style choice.
<p><img src="./images/db_3.png" alt="docker container" width="700" /></p>
---

## 6. Retrieve: hybrid → parent dedupe → CrossEncoder

`hybrid_search_with_rerank(query, k=5, source_filter=...)`:

1. Encode the query with BGE-M3 → dense + sparse.
2. Two `AnnSearchRequest`s share the same filter, e.g. `source == 'ai'`.
3. Milvus `hybrid_search` + `WeightedRanker(1.0, 0.7)`: dense owns semantics, sparse pulls the original term back.
4. Map children to parents; dedupe on `parent_content`. Several child hits in one section collapse to one parent.
5. Fewer than 2 parents: skip rerank, slice to `candidate_m`.
6. Otherwise `bge-reranker-large` scores `[query, parent_content]`, sort descending, keep Top-2.

Rerank runs on parents, not children. A CrossEncoder is more accurate than a bi-encoder and more expensive, so it only sees the short fused list, not the whole collection.

`__main__` currently stops at ingest. The search function is ready; generation can call it directly.
<p><img src="./images/db_4.png" alt="docker container" width="700" /></p>

---

## Frame it as “where recall error comes from, and which layer patches which kind”:

1. **Parse.** No text on the slide → OCR first, or the rest of the stack is idle.
2. **Split.** Sentence punctuation recursion + Markdown structure. Parent/child separates retrieval grain from answer grain.
3. **Represent.** BGE-M3 emits dense and sparse in one pass: paraphrase and exact term.
4. **Index.** IVF_FLAT for semantics, inverted index for jargon, scalar `source` to drop other subjects first.
5. **Rank.** Weighted fusion only mixes the two recall lists. Final order is the CrossEncoder.

If asked why not ES + embeddings: a course corpus fits one Milvus collection with two vector fields. No extra tokenizer or BM25 corpus to keep in sync. If asked why not ColBERT multi-vector: BGE-M3 can emit `colbert_vecs`; they are not in the schema yet. Complexity stays down on purpose.

Parameters: 1200 / 300 / 50 are working values. Too little overlap drops sentences; too large a parent adds noise for the reranker. `WeightedRanker` 1.0 / 0.7 can move with how term-heavy the queries are.

---

## 8. This file tracks the code

What is written here is ingest + hybrid retrieve. Do not pre-fill empty sections. When the next pieces land, update this file in place:

- query classification / routing (subject, chit-chat, retrieve)
- prompts and LLM generation
- handoff to MySQL / Redis QA

If chunk sizes, schema, or fusion weights change, edit the tables above. Do not start a second RAG doc.
