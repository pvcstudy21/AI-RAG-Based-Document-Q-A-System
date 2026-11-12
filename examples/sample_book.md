# A Practical Guide to Document Question Answering

This short sample book is included to make the repository testable immediately. It explains the stages of a document retrieval-augmented generation (RAG) system. The application should answer from these sections only when this file is selected, and should cite the section that supports its answer.

# 1. Loading and Parsing

Document ingestion begins with loading and parsing. A parser extracts text and useful structure from a source file. For PDFs, preserve page boundaries so an answer can point readers to a page. For Markdown and DOCX files, preserve heading or section context when possible. Store metadata such as the source filename, page number, section title, and source identifier alongside the extracted text. Image-only scanned PDFs need OCR before ordinary text extraction can work.

# 2. Chunking Strategy

Chunking divides long documents into passages small enough to embed and retrieve, while retaining enough context to make each passage meaningful. A Recursive Character Text Splitter is a good general-purpose starting point for prose: it prefers paragraph boundaries, then line and sentence boundaries, and only then splits smaller units. A practical starting configuration for books is a chunk size around 900 characters with 150 characters of overlap. These are tuning defaults, not universal constants. Keep pages or sections as parent boundaries so chunks do not silently cross unrelated sections.

# 3. Embeddings and Cosine Similarity

An embedding model converts each chunk into a dense numeric vector. The same model must be used for document chunks and user queries so their vectors share a space. Semantic search compares the query vector with stored chunk vectors. Cosine similarity measures the angle between two vectors; a higher cosine similarity usually indicates that the texts are semantically closer. Normalize embeddings when using cosine-based retrieval. Store the vector together with the chunk text and metadata so a retrieved vector can be traced back to its document.

# 4. BM25 and Hybrid Search

Dense semantic retrieval is useful when a question uses different wording from the source. BM25 is a lexical ranking method that rewards useful term overlap and can be especially effective for exact names, identifiers, technical phrases, and rare keywords. A hybrid retriever runs dense vector search and BM25 in parallel. Reciprocal Rank Fusion combines the two ranked lists using the reciprocal of each result rank plus a small constant. Rank fusion is useful because cosine scores and BM25 scores are on different scales and should not be naively added together.

# 5. Cross-Encoder Reranking

A bi-encoder embedding model independently encodes a question and a passage, making it efficient to search a large collection. A cross-encoder instead reads the question and candidate passage together and produces a relevance score. It is more expensive per candidate, so use it after first-stage retrieval: retrieve perhaps 20 to 40 candidates, rerank that smaller set, and keep the best few chunks for the answer context. The reranker improves ordering; it does not replace source-grounding or factuality checks.

# 6. Grounded Answers and Source Citations

A grounded answer should use only the retrieved excerpts, not general model knowledge. Each excerpt passed to the generator should have a stable source marker such as [S1], plus human-readable source metadata. The prompt should tell the model to cite factual claims and to abstain when the excerpts do not contain the answer. Display the source filename and page number or section title next to the answer. Validate that each generated marker refers to a source actually supplied to the model. Citations make answers auditable, but they do not by themselves prove that every claim is entailed by its cited text; human review is still important.

# 7. Retrieval and Generation Evaluation

Retrieval should be evaluated with a manually reviewed set of questions and relevant chunks. Precision at k is the fraction of the top k retrieved chunks that are relevant. Recall at k is the fraction of all known relevant chunks that appear in the top k. Mean reciprocal rank rewards a relevant chunk appearing near the top. Normalized discounted cumulative gain gives more credit to relevant results at higher ranks. Generation checks can measure whether citations are present and valid, whether expected answer terms appear, and whether the system abstains on questions whose answers are absent. These automatic checks are useful signals, not a substitute for expert evaluation of correctness, completeness, and faithfulness.

# 8. A Reliable End-to-End Workflow

A useful RAG workflow has separate ingestion, retrieval, and generation stages. Ingestion parses files, chunks text, generates embeddings, and writes vectors and metadata to a vector database. Retrieval applies dense search and lexical search, fuses their rankings, optionally reranks candidates, and selects a small evidence set. Generation constructs a bounded context from those chunks and asks a language model to answer with citations or abstain. Keep the stages independently testable so failures can be diagnosed rather than treating RAG as one opaque prompt.
