# XXGSC KM RAG Agent - Database Design Documentation

## 1. Purpose of this document

This document describes the database design for the **XXGSC KM RAG Agent** ingestion and retrieval platform.
It is written as a product-style design document so that product managers, architects, developers, DBAs, and support teams can understand:

- what each table is used for
- what every field means
- how the tables are related
- how ingestion works end to end
- how replacement/versioning should work
- how the design supports future RAG query and governance use cases

This documentation is based on the schema currently defined in `databaseSchema.dbml`.

---

## 2. Business context

The system allows users to upload knowledge documents from an APEX application.
Those documents are stored in **OCI Object Storage**, while document metadata is stored in **Oracle Database**.
The backend ingestion service then:

1. identifies the uploaded documents
2. downloads them from Object Storage
3. extracts document text
4. optionally anonymizes the content
5. chunks the content
6. generates embeddings
7. stores the embeddings for vector search

When an end user later asks a question, the system retrieves the most relevant chunks using vector similarity search and then joins those chunks back to the originating document and document metadata.

This design ensures:

- document traceability
- metadata-aware retrieval
- support for single-file and bulk-file ingestion
- support for retries and operational logging
- support for document replacement/versioning

---

## 3. Design principles

The schema follows these key design principles:

### 3.1 Separate document metadata from vector data
Business metadata such as document type, pillar, geography, and business unit belongs to the **document**, not directly to the vector row.

### 3.2 Preserve traceability
Every chunk and every embedding must be traceable back to the source document.

### 3.3 Support both single and bulk upload
All uploads are treated as a batch. A batch may contain one document or many documents.

### 3.4 Support versioning and replacement
If a user uploads an updated or corrected document, the system should support version tracking instead of uncontrolled overwrite.

### 3.5 Support operations and auditability
The ingestion service needs execution tracking and step-level logs for monitoring, troubleshooting, and audit readiness.

---

## 4. High-level architecture

The logical flow is:

**APEX Upload -> OCI Object Storage -> Database registration -> Ingestion run -> Chunking -> Embedding -> Retrieval**

### Main entity flow

```text
XXGSC_KM_INGESTION_BATCH
    -> XXGSC_KM_DOCUMENT
        -> XXGSC_KM_DOCUMENT_METADATA
        -> XXGSC_KM_DOCUMENT_VERSION
        -> XXGSC_KM_DOCUMENT_CHUNK
            -> XXGSC_KM_CHUNK_VECTOR

XXGSC_KM_INGESTION_BATCH
    -> XXGSC_KM_INGESTION_RUN
        -> XXGSC_KM_INGESTION_STEP_LOG
```

### Core mapping rule

The key retrieval mapping is:

**Vector -> Chunk -> Document -> Metadata**

That means:

- vector search identifies the matching chunk
- the chunk identifies the source document
- the document links to document-level business metadata

This is how the system returns both the relevant content and the associated document attributes.

---

## 5. Status enums used in the design

### 5.1 `batch_status`
Used by `XXGSC_KM_INGESTION_BATCH.STATUS`

- `PENDING` - batch created but not yet processed
- `IN_PROGRESS` - ingestion processing is currently running
- `PARTIAL_SUCCESS` - some documents succeeded and some failed
- `COMPLETED` - all documents in the batch completed successfully
- `FAILED` - the batch failed completely

### 5.2 `document_status`
Used by `XXGSC_KM_DOCUMENT.DOC_STATUS`

- `PENDING` - registered but not yet processed
- `DOWNLOADED` - downloaded from object storage
- `EXTRACTED` - text extracted successfully
- `ANONYMIZED` - anonymization completed
- `CHUNKED` - chunk generation completed
- `EMBEDDED` - embeddings generated
- `COMPLETED` - full ingestion completed successfully
- `FAILED` - document processing failed

### 5.3 `run_type`
Used by `XXGSC_KM_INGESTION_RUN.RUN_TYPE`

- `AUTO` - automatically triggered run
- `MANUAL_RETRY` - manually retried run
- `SCHEDULED` - scheduler-triggered run

### 5.4 `run_status`
Used by `XXGSC_KM_INGESTION_RUN.STATUS`

- `PENDING`
- `IN_PROGRESS`
- `COMPLETED`
- `FAILED`
- `PARTIAL_SUCCESS`

### 5.5 `step_status`
Used by `XXGSC_KM_INGESTION_STEP_LOG.STEP_STATUS`

- `STARTED`
- `COMPLETED`
- `FAILED`

### 5.6 `chunk_status`
Used by `XXGSC_KM_DOCUMENT_CHUNK.CHUNK_STATUS`

- `ACTIVE` - chunk is current and searchable
- `SUPERSEDED` - chunk belongs to an older replaced version
- `FAILED` - chunk creation or post-processing failed

### 5.7 `vector_status`
Defined in the DBML but not yet present as a table column in the current vector table.
Recommended values are:

- `ACTIVE`
- `SUPERSEDED`
- `FAILED`

### 5.8 `yes_no_flag`
Used for binary business flags:

- `Y`
- `N`

---

## 6. Table-by-table design

---

## 6.1 Table: `XXGSC_KM_INGESTION_BATCH`

### Purpose
This table stores the top-level ingestion request.
It represents one upload event coming from APEX or another external source.
It supports both:

- a single document upload
- a bulk upload containing multiple documents

### Business meaning
This is the parent operational record for a group of documents that should be processed together.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `BATCH_ID` | number | Yes | Primary key. Unique identifier for the ingestion batch. |
| `SOURCE_SYSTEM` | varchar2(50) | Yes | Indicates where the batch came from, such as APEX or another integration source. |
| `REQUESTED_BY` | varchar2(255) | Yes | User or system account that initiated the upload. |
| `REQUESTED_AT` | timestamp | Yes | Time when the batch was created. |
| `STATUS` | batch_status | Yes | Current lifecycle status of the batch. |
| `TOTAL_DOCUMENTS` | number | Yes | Total number of documents registered in this batch. |
| `SUCCESSFUL_DOCUMENTS` | number | Yes | Number of documents successfully ingested. |
| `FAILED_DOCUMENTS` | number | Yes | Number of documents that failed during ingestion. |
| `OBJECT_BUCKET` | varchar2(255) | No | Default OCI bucket name used by the batch. |
| `OBJECT_NAMESPACE` | varchar2(255) | No | OCI object storage namespace. |
| `ERROR_MESSAGE` | clob | No | Batch-level error or summary message. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation timestamp. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update timestamp. |

### Relationships
- One batch can have many documents.
- One batch can have many ingestion runs.
- One batch can have many step log rows through document processing.

---

## 6.2 Table: `XXGSC_KM_DOCUMENT`

### Purpose
This is the master record for an uploaded document.
It connects the logical business document with the actual file stored in Object Storage and with all downstream ingestion records.

### Business meaning
This table represents the source document as a managed record inside the knowledge platform.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `DOCUMENT_ID` | number | Yes | Primary key. Unique identifier for the logical document record. |
| `BATCH_ID` | number | Yes | Foreign key to `XXGSC_KM_INGESTION_BATCH.BATCH_ID`. Identifies the upload batch that created the document. |
| `FILE_NAME` | varchar2(500) | Yes | Normalized filename used by the storage/process layer. |
| `ORIGINAL_FILE_NAME` | varchar2(500) | Yes | Original filename uploaded by the end user. |
| `OBJECT_NAME` | varchar2(1000) | Yes | Object storage key of the file. Typically includes folder path. |
| `OBJECT_URI` | varchar2(2000) | No | Full OCI object URI for the stored file. |
| `BUCKET_NAME` | varchar2(255) | Yes | OCI bucket name containing the document. |
| `NAMESPACE_NAME` | varchar2(255) | Yes | OCI namespace for the bucket. |
| `SOURCE_SYSTEM` | varchar2(50) | Yes | System from which the document originated, usually APEX. |
| `DOC_STATUS` | document_status | Yes | Current lifecycle state of the document. |
| `IS_ACTIVE` | yes_no_flag | Yes | Indicates whether the document is currently active for business use. |
| `CURRENT_VERSION_NO` | number | Yes | Latest version number for the document. |
| `LAST_SUCCESSFUL_RUN_ID` | number | No | Foreign key to the last successful ingestion run that processed this document. |
| `UPLOADED_BY` | varchar2(255) | Yes | Business user or application user that uploaded the document. |
| `UPLOADED_AT` | timestamp | Yes | Timestamp when the document was uploaded. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- Many documents belong to one batch.
- One document has one metadata row.
- One document can have many version rows.
- One document can have many chunks.
- One document can have many vector rows indirectly through chunks.
- One document can appear in many step log rows.

### Important note
This is the main anchor table for document traceability.

---

## 6.3 Table: `XXGSC_KM_DOCUMENT_METADATA`

### Purpose
This table stores structured business metadata collected at upload time.

### Business meaning
It captures document classification information used in search filtering, business understanding, and retrieval display.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `DOCUMENT_ID` | number | Yes | Primary key and foreign key to `XXGSC_KM_DOCUMENT.DOCUMENT_ID`. One metadata record per document. |
| `DOCUMENT_TYPE` | varchar2(100) | Yes | Type of document, such as Technical or Functional. |
| `PILLAR` | varchar2(100) | Yes | Business or solution pillar, such as ERP, HCM, SCM, or CX. |
| `GEOGRAPHY` | varchar2(100) | No | Geography or region associated with the document. |
| `BUSINESS_UNIT` | varchar2(100) | No | Business unit associated with the document. |
| `PROJECT_NAME` | varchar2(255) | No | Project, engagement, or initiative associated with the document. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- One metadata record belongs to exactly one document.
- Search results should join back to this table through `DOCUMENT_ID`.

### Important note
This is the master source of business-facing metadata for retrieval results.

---

## 6.4 Table: `XXGSC_KM_DOCUMENT_VERSION`

### Purpose
This table stores the version history of a document.

### Business meaning
It supports document replacement, correction, and re-upload use cases without losing history.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `DOCUMENT_VERSION_ID` | number | Yes | Primary key. Unique identifier for a specific document version row. |
| `DOCUMENT_ID` | number | Yes | Foreign key to the logical document. |
| `VERSION_NO` | number | Yes | Numeric version number of the uploaded file. |
| `OBJECT_NAME` | varchar2(1000) | Yes | Object storage key for the specific version file. |
| `IS_CURRENT` | yes_no_flag | Yes | Indicates whether this version is the current version. |
| `VALID_FROM` | timestamp | Yes | Start time from which this version is considered valid. |
| `VALID_TO` | timestamp | No | End time for version validity, typically populated when superseded. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- One document can have multiple version rows.
- Only one version should normally be current at a time.

### Important note
This table is critical for supporting replacement of incorrect or outdated uploaded documents.

---

## 6.5 Table: `XXGSC_KM_INGESTION_RUN`

### Purpose
This table stores each execution attempt of the ingestion process for a batch.

### Business meaning
It separates upload registration from backend processing and allows retries and operational observability.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `RUN_ID` | number | Yes | Primary key. Unique identifier of the ingestion execution. |
| `BATCH_ID` | number | Yes | Foreign key to the batch being processed. |
| `RUN_TYPE` | run_type | Yes | Indicates whether this run is automatic, manual retry, or scheduled. |
| `STARTED_AT` | timestamp | Yes | Time when the run started. |
| `ENDED_AT` | timestamp | No | Time when the run completed or terminated. |
| `STATUS` | run_status | Yes | Current or final status of the run. |
| `TRIGGERED_BY` | varchar2(255) | Yes | System or user that triggered the run. |
| `TOTAL_DOCUMENTS` | number | Yes | Number of documents expected in this run. |
| `SUCCESSFUL_DOCUMENTS` | number | Yes | Number of documents successfully completed in this run. |
| `FAILED_DOCUMENTS` | number | Yes | Number of failed documents in this run. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- One run belongs to one batch.
- One run can produce many step logs.
- One run can produce many chunks and vector rows.

---

## 6.6 Table: `XXGSC_KM_INGESTION_STEP_LOG`

### Purpose
This table records detailed step-level processing information.

### Business meaning
It is the operational audit trail of the ingestion pipeline.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `LOG_ID` | number | Yes | Primary key for the log row. |
| `RUN_ID` | number | Yes | Foreign key to the ingestion run. |
| `BATCH_ID` | number | Yes | Foreign key to the parent batch. |
| `DOCUMENT_ID` | number | Yes | Foreign key to the document being processed. |
| `STEP_NAME` | varchar2(100) | Yes | Name of the step, such as DOWNLOAD, EXTRACT, CHUNK, EMBED, or STORE_VECTOR. |
| `STEP_STATUS` | step_status | Yes | Status of the step execution. |
| `STEP_SEQUENCE` | number | Yes | Processing order of the step inside the run. |
| `MESSAGE` | clob | No | Informational or diagnostic message. |
| `STARTED_AT` | timestamp | Yes | Time when the step started. |
| `ENDED_AT` | timestamp | No | Time when the step ended. |
| `DURATION_MS` | number | No | Step duration in milliseconds. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- Many step logs can belong to one run.
- Many step logs can belong to one document.
- Many step logs can belong to one batch.

---

## 6.7 Table: `XXGSC_KM_DOCUMENT_CHUNK`

### Purpose
This table stores the chunked text generated from documents during ingestion.

### Business meaning
It is the structured text layer that bridges raw document content and embeddings.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `CHUNK_ID` | varchar2(100) | Yes | Primary key. Unique identifier of the chunk. |
| `DOCUMENT_ID` | number | Yes | Foreign key to the source document. |
| `RUN_ID` | number | Yes | Foreign key to the ingestion run that created the chunk. |
| `CHUNK_INDEX` | number | Yes | Sequential order of the chunk within the document. |
| `SECTION_HEADING` | varchar2(500) | No | Heading or logical section title associated with the chunk. |
| `CHUNK_TEXT` | clob | Yes | Actual chunk text used for embedding and retrieval. |
| `CHUNK_HASH` | varchar2(500) | Yes | Hash of the chunk content for deduplication or comparison. |
| `IS_ANONYMIZED` | yes_no_flag | Yes | Indicates whether the chunk content has been anonymized. |
| `CHUNK_STATUS` | chunk_status | Yes | Status of the chunk such as ACTIVE or SUPERSEDED. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- Many chunks belong to one document.
- Many chunks can be created in one ingestion run.
- One chunk should map to one vector row in the current model.

### Important note
This table is the main bridge from vector search back to the source document and its metadata.

---

## 6.8 Table: `XXGSC_KM_CHUNK_VECTOR`

### Purpose
This table stores the vector embedding generated from each chunk.

### Business meaning
It is the semantic search layer of the system.

### Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `VECTOR_ID` | varchar2(100) | Yes | Primary key. Unique identifier for the vector row. |
| `CHUNK_ID` | varchar2(100) | Yes | Foreign key to the chunk represented by this vector. Unique in the current model, meaning one vector per chunk. |
| `DOCUMENT_ID` | number | Yes | Foreign key to the parent document for easier joins and filtering. |
| `RUN_ID` | number | Yes | Foreign key to the ingestion run that generated the vector. |
| `EMBEDDING_VECTOR` | text | Yes | The embedding payload. In actual Oracle implementation this should be a VECTOR type, not plain text. |
| `CREATED_BY` | varchar2(255) | Yes | WHO column indicating record creator. |
| `CREATED_AT` | timestamp | Yes | WHO column indicating creation time. |
| `UPDATED_BY` | varchar2(255) | Yes | WHO column indicating last updater. |
| `UPDATED_AT` | timestamp | Yes | WHO column indicating last update time. |

### Relationships
- One vector belongs to one chunk.
- Many vectors belong to one document.
- Many vectors can be generated in one run.

### Important note
Business metadata is not mastered here. It should be obtained by joining:

`XXGSC_KM_CHUNK_VECTOR -> XXGSC_KM_DOCUMENT_CHUNK -> XXGSC_KM_DOCUMENT -> XXGSC_KM_DOCUMENT_METADATA`

---

## 7. Relationship summary

### 7.1 Entity relationships

| Parent | Child | Relationship | Meaning |
|---|---|---|---|
| `XXGSC_KM_INGESTION_BATCH` | `XXGSC_KM_DOCUMENT` | 1 to many | One upload batch can contain many documents |
| `XXGSC_KM_INGESTION_BATCH` | `XXGSC_KM_INGESTION_RUN` | 1 to many | One batch can be processed multiple times |
| `XXGSC_KM_DOCUMENT` | `XXGSC_KM_DOCUMENT_METADATA` | 1 to 1 | Each document has one structured metadata record |
| `XXGSC_KM_DOCUMENT` | `XXGSC_KM_DOCUMENT_VERSION` | 1 to many | One logical document can have many versions |
| `XXGSC_KM_DOCUMENT` | `XXGSC_KM_DOCUMENT_CHUNK` | 1 to many | One document creates many chunks |
| `XXGSC_KM_DOCUMENT` | `XXGSC_KM_CHUNK_VECTOR` | 1 to many | A document can have many embeddings via chunks |
| `XXGSC_KM_INGESTION_RUN` | `XXGSC_KM_INGESTION_STEP_LOG` | 1 to many | One run produces many step logs |
| `XXGSC_KM_INGESTION_RUN` | `XXGSC_KM_DOCUMENT_CHUNK` | 1 to many | One run can create many chunks |
| `XXGSC_KM_INGESTION_RUN` | `XXGSC_KM_CHUNK_VECTOR` | 1 to many | One run can create many vectors |
| `XXGSC_KM_DOCUMENT_CHUNK` | `XXGSC_KM_CHUNK_VECTOR` | 1 to 1 | In current model, one chunk has one vector |

---

## 8. End-to-end ingestion lifecycle

### Step 1: user uploads document(s)
APEX uploads one or more files to OCI Object Storage and captures metadata.

### Step 2: batch and document registration
Rows are created in:

- `XXGSC_KM_INGESTION_BATCH`
- `XXGSC_KM_DOCUMENT`
- `XXGSC_KM_DOCUMENT_METADATA`
- optionally `XXGSC_KM_DOCUMENT_VERSION`

### Step 3: ingestion run starts
Backend creates a row in `XXGSC_KM_INGESTION_RUN`.

### Step 4: step logs are recorded
For each document and each processing stage, rows are written into `XXGSC_KM_INGESTION_STEP_LOG`.

### Step 5: document is chunked
Rows are written into `XXGSC_KM_DOCUMENT_CHUNK`.

### Step 6: embeddings are generated
Rows are written into `XXGSC_KM_CHUNK_VECTOR`.

### Step 7: retrieval later joins the chain
When search returns a vector match, the system joins back to:

- chunk
- document
- metadata

to build a complete response with citation details.

---

## 9. How retrieval works with this design

At query time, the application should:

1. generate embedding for user query
2. search vector similarity in `XXGSC_KM_CHUNK_VECTOR`
3. resolve the matching chunk from `XXGSC_KM_DOCUMENT_CHUNK`
4. resolve the parent document from `XXGSC_KM_DOCUMENT`
5. resolve business metadata from `XXGSC_KM_DOCUMENT_METADATA`

### Retrieval join chain

```text
XXGSC_KM_CHUNK_VECTOR
    -> XXGSC_KM_DOCUMENT_CHUNK
    -> XXGSC_KM_DOCUMENT
    -> XXGSC_KM_DOCUMENT_METADATA
```

### Outcome
This allows the system to return:

- relevant text snippet
- source file name
- document type
- pillar
- geography
- business unit
- project name

---

## 10. Document replacement and versioning

The design supports replacement through controlled versioning.

### Typical replacement scenario
If a user uploaded a wrong document or wants to upload an updated file:

1. keep the same logical `DOCUMENT_ID`
2. insert a new row in `XXGSC_KM_DOCUMENT_VERSION`
3. increment `CURRENT_VERSION_NO` in `XXGSC_KM_DOCUMENT`
4. mark the previous version as no longer current
5. re-run ingestion for the new version
6. mark old chunks as `SUPERSEDED`
7. future retrieval should use only active/current records

### Why this is important
This avoids uncontrolled overwrites and preserves audit history.

### Recommended enhancement
For stronger version traceability, consider adding `DOCUMENT_VERSION_ID` into:

- `XXGSC_KM_DOCUMENT_CHUNK`
- `XXGSC_KM_CHUNK_VECTOR`

This will allow the system to know exactly which chunk and vector belong to which physical uploaded version.

### Additional recommended enhancement
Add `VECTOR_STATUS` to `XXGSC_KM_CHUNK_VECTOR` so old vectors can be retired safely during document replacement.

---

## 11. Indexing summary

The current schema includes practical indexes for common access paths.

### Key indexing intent

- `BATCH_ID` indexes support batch-driven ingestion and monitoring
- `DOC_STATUS` supports document processing filters
- metadata indexes support business filtering
- `(DOCUMENT_ID, CHUNK_INDEX)` ensures chunk ordering uniqueness
- chunk hash helps deduplication or change detection
- vector table indexes support joins by document and run

### Future indexing considerations
Depending on production scale, you may later add:

- document version lookup indexes
- active/current filtering indexes
- metadata composite indexes for search filters
- Oracle-native vector index strategy for embedding search

---

## 12. WHO columns and audit model

WHO columns appear across all core tables.

### Standard WHO fields

- `CREATED_BY`
- `CREATED_AT`
- `UPDATED_BY`
- `UPDATED_AT`

### Purpose
These support:

- operational traceability
- audit readiness
- ownership tracking
- support diagnostics

### Typical values

- `APEX_APP`
- `FASTAPI_INGEST`
- admin username
- service account name

---

## 13. Design strengths

This schema has several strong design characteristics:

1. clear separation between upload control, metadata, content, and vectors
2. supports both single and bulk ingestion with the same model
3. operationally traceable through run and step log tables
4. retrieval is explainable because vectors map back to documents and metadata
5. supports replacement/versioning with controlled evolution
6. simple enough to implement while still enterprise-friendly

---

## 14. Recommended future enhancements

For a more complete production-grade implementation, the following enhancements are recommended:

### 14.1 Add `DOCUMENT_VERSION_ID` to chunk and vector tables
This will make version-aware replacement and rollback much cleaner.

### 14.2 Add `VECTOR_STATUS` to `XXGSC_KM_CHUNK_VECTOR`
This will allow old vectors to be retired during replacement.

### 14.3 Add embedding metadata fields
Consider adding:

- `EMBEDDING_MODEL`
- `EMBEDDING_VERSION`
- `VECTOR_DIMENSION`

These are useful for model tracking and future migrations.

### 14.4 Add richer business metadata if needed
Depending on product needs, later fields may include:

- confidentiality level
- customer name
- project code
- country code
- language code

### 14.5 Add query audit tables if governance becomes important
This would help track who searched what and which filters were used.

---

## 15. Final summary

The `XXGSC_KM_*` database design provides a strong foundation for a knowledge-management ingestion and RAG retrieval platform.

It is designed so that:

- APEX can upload documents and metadata
- Object Storage can store the physical files
- ingestion can run in a controlled, observable way
- chunking and embedding can be traced to source documents
- retrieval can return both content and business metadata
- replacement and versioning can be handled cleanly

In short, this design is suitable as a product-level foundation for an enterprise document ingestion and retrieval workflow.