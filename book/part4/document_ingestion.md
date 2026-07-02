---
title: "Document Ingestion"
<!-- image: figures/ch3/journey_part3_ch3.svg -->
---


> **The canonical question for this chapter:**
> *You upload a PDF, a Word document, a webpage, or a database export. Before
> any of it can be retrieved or reasoned over, it must be transformed into
> something a retrieval system can work with. What actually happens during that
> transformation?*

---

::: {.callout-note appearance="minimal"}
**Where are we?**

![The journey through the Model Mind.](figures/ch3/journey_part4_ch3.svg){#fig-progress width="80%"}

Before a document can be retrieved, it must be ingested, parsed, cleaned, 
structured, and prepared for the pipeline that follows. This is where most RAG 
systems fail, and it happens long before any query is ever issued.
:::

---

## The ingestion problem

A language model's knowledge is frozen at training time. It knows what was in
its training data and nothing more. For most production applications, this is
insufficient, users ask about documents the model has never seen, databases
that change daily, proprietary internal knowledge that was never part of any
public corpus.

Retrieval-Augmented Generation solves this by giving the model access to
external knowledge at inference time. But before any document can be retrieved,
it must be ingested: parsed, cleaned, structured, and stored in a form that
retrieval systems can query efficiently.

Ingestion is the foundation of every RAG system and where most RAG failures 
actually originate. It is not in the retrieval algorithm or the language model,
but in poor parsing, inadequate cleaning, or naive chunking that destroys the
coherence of the source material. Getting ingestion right is a prerequisite
for optimizing anything else.

---

## Document formats and their challenges

The first step in ingestion is parsing: extracting usable text from whatever
format the document arrives in. Different formats present different challenges.

### PDF

The most common and most problematic format. PDF is a presentation format,
not a semantic one. It describes where each character is placed on a page,
not what structure the document has.

A PDF does not have "paragraphs" or "sections" in any structural sense. It has
positioned text elements, each with a font, size, and (x, y) coordinate.
Reconstructing reading order, identifying headings, distinguishing body text
from captions, and separating columns requires heuristics that work well on
simple documents and poorly on complex ones.

Specific PDF challenges:

**Multi-column layouts.** Text from different columns is interleaved in the
raw extraction order. Naive extractors produce text that mixes columns:
"The company revenue | left column second line | right column second line |
grew significantly." Correct column separation requires spatial analysis.

**Tables.** Tabular data in PDFs is often just positioned text with no
structural information. Extractors must infer table structure from spatial
relationships between text elements. This fails on complex merged cells,
rotated headers, and tables that span pages.

**Scanned PDFs.** PDFs that contain scanned images rather than text contain
no extractable text at all, what appears as “text” is actually just pixels. 
Extraction requires OCR, which adds its own error rate. A 98% OCR accuracy 
rate sounds high until you realize it means roughly one error every fifty 
words in a dense technical document.

**Headers and footers.** Page numbers, document titles, and section headers
in running headers and footers are often extracted inline with body text.
A 100-page document might have "CONFIDENTIAL Page N of 100" injected into
the text every few paragraphs.

**Mathematical notation.** Equations in PDFs are often stored as positioned
symbols rather than structured mathematical expressions. Extraction typically
produces garbled sequences of symbols.

PDF libraries for Python:

- **PyMuPDF (fitz)**: fast, good text extraction for standard PDFs, reasonable
  layout analysis
- **pdfplumber**: better table extraction than PyMuPDF, slower
- **pypdf**: pure Python, simpler, adequate for basic extraction
- **Adobe PDF Extract API**: commercial, highest quality for complex layouts


For high-stakes RAG applications (legal documents, financial filings, technical
documentation) investing in high-quality PDF parsing pays off significantly.
A poorly parsed PDF produces retrieval failures that are invisible at the
extraction stage but manifest as hallucinations and incorrect answers at
inference time.

### Word documents (DOCX)

Structurally richer than PDF. DOCX files have an explicit document model:
paragraphs, headings (with levels), tables (with cell structure), lists, and
styles. Extraction can respect this structure rather than reconstructing it
from layout.


Challenges: embedded objects (images, embedded PDFs, charts) are opaque blobs
whose content is inaccessible without separate processing. Tracked changes and
revision marks can clutter extracted text. 

### HTML and web pages

HTML is semantically rich when used correctly. Heading tags (`<h1>` through
`<h6>`), paragraph tags, list tags, table tags, and semantic elements like
`<article>`, `<section>`, and `<nav>` provide structural information that text
extractors can use.

In practice, web HTML is rarely so clean. Navigation menus, cookie banners,
advertisements, and footers must be removed. JavaScript-rendered content
requires a headless browser to extract. 


### Plain text and Markdown

The easiest cases. Text files require no parsing; they are read directly.
Markdown adds lightweight semantic structure (headings via `#`, code blocks
via backticks, lists) that parsers can respect.

Markdown is increasingly common as a RAG source format because it is both
human-readable and machine-parseable. Documents intentionally written for RAG
ingestion are sometimes authored in Markdown for this reason. 

### Spreadsheets (XLSX, CSV)

Tabular data presents a different challenge: how do you convert a table into
text that can be retrieved and understood by a language model?

Options:

- **Row-by-row serialization**: "Row 3: Name=Alice, Age=32, Department=Engineering,
  Salary=95000"
- **Markdown table format**: convert to a Markdown table that the LLM can parse
- **Schema + sample rows**: describe the schema and include representative
  examples; retrieve the schema for schema-level questions and specific rows
  for value-level questions

For large spreadsheets (thousands of rows), row-by-row serialization produces
too much text to embed meaningfully. SQL or structured databases with
text-to-SQL generation are often a better architecture than RAG for tabular
data. Retrieval finds relevant rows; the model generates SQL; the database
executes it exactly.

### Images and multimodal documents

Documents with meaningful visual content (diagrams, charts, photographs with
captions, infographics) require multimodal processing.

OCR extracts text from images. Tesseract is the standard open-source OCR
library; AWS Textract, Google Cloud Vision, and Azure Computer Vision are
commercial alternatives with higher accuracy on complex layouts.

Vision-language models can describe image content, answer questions about figures, 
and extract structured information from charts. For documents where figures carry 
significant information, routing image content through a VLM before ingestion 
produces much better retrieval coverage than ignoring figures or relying on captions 
alone. A technical diagram with no caption is invisible to text-based retrieval 
without VLM description.

---

## Text cleaning

After parsing, raw extracted text often contains artifacts that reduce retrieval quality 
and cleaning is used to remove or normalize them.

### Common cleaning steps

**Whitespace normalization.** Collapse multiple spaces to one, remove
leading/trailing whitespace from paragraphs, normalize line endings. Necessary
because PDF extraction produces variable whitespace depending on character
positioning.

**Header and footer removal.** Identify and remove page numbers, document
titles repeated on each page, copyright notices, and other non-content text.
Pattern matching (regex for "Page N of M") handles simple cases and spatial
analysis during PDF parsing handles complex ones.

**Hyphenation repair.** PDF extraction often breaks hyphenated words at line
breaks: "trans-\naction" appears as two tokens rather than "transaction".
Detecting and joining these is straightforward for end-of-line hyphens but
ambiguous for legitimate compound hyphenation.

**Encoding normalization.** Documents from multiple sources may have different
encodings (UTF-8, Latin-1, Windows-1252). Normalize to UTF-8. Handle common
encoding artifacts: curly quotes that appear as `â€œ`, em dashes that appear
as `â€"`.

**Boilerplate removal.** Repeated passages that appear across many documents
(terms of service, disclaimer language, standard legal notices) add noise
to embeddings and can dominate retrieval for questions that accidentally match
boilerplate patterns. Deduplication at the passage level or explicit
blocklisting removes this.

**Table of contents removal.** Tables of contents in PDFs and Word documents
are often extracted as body text, producing a list of section titles that
duplicates content the model will encounter in the actual sections. Identifying
and removing ToC passages prevents this duplication from dominating retrieval.

**Reference list handling.** Bibliographies and reference lists at the end
of academic papers are often extracted as body text. Whether to include or
exclude them depends on whether the references themselves are useful for
retrieval. For scientific RAG, including references can help answer "what
papers support this claim" questions.

### Cleaning for specific domains

**Legal documents.** Case citations, statutory references, and defined terms
have specific patterns. Preserving these patterns rather than normalizing them
is important for legal search. Removing citation formatting destroys the
semantic signal that distinguishes a reference to a case from a discussion
of it.

**Medical records.** PHI (Protected Health Information) removal is legally
required before storing patient records in retrieval systems. Names, dates,
locations, and identifiers must be de-identified before ingestion. This is
not optional and not a post-processing step, it must happen before any
content is stored.

**Code.** Programming language syntax should not be normalized like prose.
Tab/space preservation matters for Python. Comment blocks, docstrings, and
variable names are semantically important and should not be striped. Code is 
often better chunked by function or class boundary than by token count.

**Financial filings.** Numerical tables in SEC filings have specific conventions
(negative numbers in parentheses, thousands vs. millions) which should be
preserved or explicitly normalized consistently across the corpus.

---

## Document metadata extraction

Every ingested document should be accompanied by metadata that enables
filtering, ranking, and attribution during retrieval.

### Core metadata fields

```python
{
    "document_id": "uuid-abc123",
    "source_url": "https://example.com/docs/api-reference.pdf",
    "source_type": "pdf",
    "title": "API Reference Guide v3.2",
    "author": "Engineering Team",
    "created_at": "2024-01-15T10:30:00Z",
    "modified_at": "2024-03-22T14:15:00Z",
    "language": "en",
    "page_count": 48,
    "word_count": 23500,
    "section": "Technical Documentation",
    "tags": ["api", "reference", "v3"],
    "access_level": "internal",
}
```

### Why metadata matters

**Filtering.** Retrieval can be limited to documents matching metadata criteria.
"What does our API documentation say about authentication?" should retrieve from
technical documentation, not from marketing materials or HR policies. Without
metadata filtering, vector similarity is the only selection mechanism, and it
will surface whatever is most semantically similar regardless of source.

**Recency ranking.** For questions about current state, recently modified
documents should rank higher. Metadata enables this without modifying embeddings.

**Access control.** Metadata carries authorization information. Users should
not retrieve documents their access level does not permit, even if those
documents are semantically relevant to their query. Access control must be
enforced at retrieval time, not just at ingestion time.

**Attribution.** When the model uses retrieved content to generate a response,
metadata provides the citation: "According to the API Reference Guide v3.2,
section 4..." Users who need to verify information can follow the citation to
the source.

### Automatic metadata extraction

Some metadata fields must be inferred rather than extracted directly:

**Language detection.** langdetect and fastText language identification models
classify language from text samples. Important for multilingual corpora where
language-specific retrieval is needed, you do not want a French query
retrieving German documents.

**Topic classification.** Embedding the document title and first paragraph and
comparing to topic embedding clusters can assign broad topical categories.
Useful for filtering and for routing queries to the right sub-corpus.

**Entity extraction.** Named entity recognition (NER) identifies people,
organizations, locations, and other entities mentioned in the document. These
can be stored as metadata fields enabling entity-based filtering: "find
documents that mention Acme Corp."

---

## The ingestion pipeline assembled

Putting the components together, a production document ingestion pipeline:

```
Raw document (PDF, DOCX, HTML, etc.)
      │
      ▼
Format detection
      │
      ▼
Parser selection and execution
  ├── PDF: PyMuPDF / pdfplumber / LlamaParse
  ├── DOCX: python-docx
  ├── HTML: trafilatura + BeautifulSoup
  ├── Image: OCR + VLM captioning
  └── Plain text: direct read
      │
      ▼
Raw text + structural information
  (paragraphs, headings, tables, page numbers)
      │
      ▼
Text cleaning
  ├── Whitespace normalization
  ├── Header/footer removal
  ├── Encoding normalization
  ├── Boilerplate removal
  └── Domain-specific cleaning
      │
      ▼
Metadata extraction
  ├── Core fields (title, source, timestamps)
  ├── Language detection
  └── Entity extraction (optional)
      │
      ▼
Chunking
      │
      ▼
Embedding
      │
      ▼
Vector store + metadata store
```

### Idempotency and incremental updates

A production ingestion pipeline runs continuously as new documents arrive and
existing documents are updated. Two properties are essential:

**Idempotency.** Ingesting the same document twice should produce the same
result as ingesting it once. Without idempotency, re-running the pipeline on
a partially-updated corpus duplicates documents and corrupts the retrieval
index. Implementation: checksums enable detection of already-ingested documents.
If a document's checksum matches an existing entry, skip re-ingestion. If the
checksum differs (document has been updated), delete the old chunks and embeddings
and re-ingest.

**Atomic updates.** When a document is updated, the retrieval index should not
be in a mixed state (some chunks from the old version, some from the new).
Atomic updates require: ingesting the new version to a staging area, deleting
all old chunks and embeddings for this document ID, and promoting the new
chunks and embeddings from staging to production. These steps are typically
transactional in a properly designed metadata store.

### Error handling and monitoring

Document ingestion fails silently in ways that are hard to detect. A document
that fails to parse produces no chunks and no embeddings, it simply does not
exist in the retrieval index. Users asking questions about that document get
answers based on whatever other documents are retrieved instead. Without
logging and monitoring, these failures are invisible.

Instrument the pipeline to track: documents processed, failed, and skipped
per run; parse failures by format and error type; quality gate rejection rates;
embedding failures; and processing latency by document type and size.

Alert on anomalies: if the PDF parse failure rate suddenly spikes, the parser
library may have a bug or the input format may have changed. If the quality
gate rejects an unusually large fraction of documents, the source may be
producing degraded content.

---

## Ingestion at scale

For large corpora (millions of documents) the ingestion pipeline must be
designed for scale from the start.

### Parallelization

Document parsing is CPU-bound and embarrassingly parallel, each document can
be processed independently. Use a task queue (Celery, Ray, AWS SQS + Lambda)
to distribute documents across multiple workers. Each worker handles parsing,
cleaning, chunking, and embedding for its assigned documents.

The embedding step often calls an external API (OpenAI, claude). API
rate limits impose a ceiling on parallelism. Design the pipeline to batch
embedding requests at the maximum allowed batch size and implement rate limit
handling with exponential backoff.

### Incremental ingestion

For corpora that change frequently, full re-ingestion on every update is too
slow. Incremental ingestion processes only new and modified documents:

1. Maintain a manifest of all ingested documents and their checksums
2. On each run, compare new document checksums to the manifest
3. Process only documents with new or changed checksums
4. Update the manifest after successful ingestion

For web crawls, compare last-modified headers or ETags to detect changes
without re-downloading content.

### Storage tiering

Ingested documents at various stages should be stored durably:

- **Raw documents**: store the original files in object storage (S3, GCS).
  Enables re-ingestion if the pipeline changes without retrieving the original
  from the source.
- **Parsed text**: store the cleaned extracted text alongside metadata. Enables
  re-chunking and re-embedding without re-parsing.
- **Chunks**: store chunk text and metadata. Enables re-embedding without
  re-chunking.
- **Embeddings**: stored in the vector database. The final retrieval index.

Each tier enables rerunning from that point forward without rerunning earlier
stages. When you improve your embedding model, you only need to re-embed, not
re-parse. When you improve your chunking strategy, you only need to re-chunk
and re-embed, not re-parse. This separation pays off every time a component
of the pipeline changes.

### Versioning the pipeline

The ingestion pipeline is code that runs over data to produce embeddings. 
Changes to any component (parser version, cleaning rules, chunking strategy, 
embedding model) invalidate some or all of the stored artifacts.

Track the pipeline version with each stored artifact. When the pipeline version
changes, identify which artifacts need to recompute. This is analogous to
dependency tracking in build systems and is essential for maintaining a coherent
retrieval index. Without it, a corpus may silently contain chunks embedded with
an old model alongside chunks embedded with a new one, making similarity
comparisons meaningless.

---

## Key takeaways

- Document ingestion is the foundation of every RAG system; most RAG failures
  originate in poor parsing, inadequate cleaning, or naive structure destruction
  rather than in retrieval algorithms or language models
- PDF is the most common and most problematic format: a presentation format
  with no semantic structure, requiring heuristic reconstruction of reading
  order, headings, columns, and tables; for high-stakes applications, invest
  in high-quality PDF parsing
- Different formats require different parsers and present different challenges;
  text files and Markdown are easy; scanned PDFs and multi-column layouts are
  hard
- Text cleaning removes artifacts introduced by parsing — whitespace, headers
  and footers, encoding issues, boilerplate — that degrade retrieval quality
  without conveying content; domain-specific cleaning (PHI removal, citation
  preservation, code formatting) requires domain knowledge
- Metadata is not optional: it enables filtering by source, date, and access
  level; provides attribution for citations; and allows quality gating and
  deduplication before embedding
- A production ingestion pipeline must be idempotent (re-running is safe) and
  support atomic updates (document updates replace all old chunks, not mix
  old and new)
- Silent failures — documents that fail to parse or are rejected by quality
  gates — are invisible without explicit logging and monitoring; instrument
  the pipeline before you need to debug it
- At scale, ingestion must be parallelized, incremental, and tiered: store
  artifacts at each stage so that pipeline improvements can be applied from
  that stage forward without rerunning from scratch
- Track pipeline versions with stored artifacts; a corpus that mixes chunks
  embedded with different models has corrupted similarity comparisons

---

## Further reading

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks.* — The foundational RAG paper; establishes the ingestion →
  retrieval → generation framework.
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization.* — GraphRAG; extends ingestion to extract
  entities and relationships, not just text chunks.
- Unstructured.io (2023). *Unstructured: Document parsing for LLM applications.*
  — The leading open-source document parsing library for RAG; handles many
  formats with layout analysis.
- Shi et al. (2023). *REPLUG: Retrieval-Augmented Black-Box Language Models.*
  — Analysis of how retrieval quality affects end-to-end RAG performance;
  ingestion quality is the upstream dependency.

---