# Ingestion API Documentation

This document describes all ingestion-related APIs currently available in the backend.

Base URL for local testing:

```text
http://127.0.0.1:8000
```

---

## 1. `POST /ingest`

Submit a batch for ingestion.

### Purpose
- Create/find project
- Create ingestion batch
- Create ingestion run
- Accept one or more documents
- Start background ingestion
- Return immediately with `batch_id` and `run_id`

### Request Headers

```http
Content-Type: application/json
```

### Request Body

```json
{
  "requested_by": "APEX_USER",
  "source_system": "APEX",
  "anonymize_docs": true,
  "project": {
    "project_name": "KM_LOCAL_TEST_PROJECT",
    "geography_code": "IN",
    "vertical_code": "GSC",
    "engagement_type": "LOCAL_TEST",
    "confidentiality": "INTERNAL"
  },
  "documents": [
    {
      "file_name": "DOM_Global_Design_ISSUE 1.0.docx",
      "file_path": "C:\\Users\\shshrohi\\Desktop\\KM_Docs_Rag_Agent\\backend\\app\\data\\downloads\\GlobalDesign\\DOM_Global_Design_ISSUE 1.0.docx",
      "object_name": "GlobalDesign/DOM_Global_Design_ISSUE 1.0.docx",
      "bucket_name": "gsc-scm-loading-km-doc",
      "namespace_name": "ax4qsxvnsmtm",
      "module_code": "GLOBAL_DESIGN",
      "doc_type_code": "DOCX",
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "content_text": null
    }
  ]
}
```

### Field Notes

#### Top-level fields
- `requested_by` - user/system who triggered ingestion
- `source_system` - source name like `APEX`, `POSTMAN`
- `anonymize_docs` - whether text anonymization should run
- `project` - project metadata
- `documents` - array of one or more documents

### Hardcoded ingestion defaults
These are now fixed in backend code and are not part of request payload:

- chunk max tokens = `300`
- chunk overlap tokens = `40`
- no VM download is triggered by `POST /ingest`

#### Document fields
- `file_name` - logical document name
- `file_path` - local file path on server/VM
- `object_name` - object storage path/key
- `bucket_name` - OCI bucket name
- `namespace_name` - OCI namespace
- `module_code` - module classification
- `doc_type_code` - document type
- `mime_type` - MIME type
- `content_text` - use this when frontend directly sends extracted text instead of file path

### Success Response

```json
{
  "status": "accepted",
  "message": "Batch accepted for background ingestion",
  "project_id": "4D90...",
  "project_name": "KM_LOCAL_TEST_PROJECT",
  "batch_id": "4D90...",
  "run_id": "4D90..."
}
```

### Error Response

```json
{
  "detail": {
    "message": "Ingestion pipeline failed.",
    "error": "actual error message"
  }
}
```

---

## 2. `GET /batches/{batch_id}`

Get batch summary and current processing status.

### Example Request

```http
GET /batches/4D90ABC123
```

### Success Response

```json
{
  "batch_id": "4D90ABC123",
  "status": "IN_PROGRESS",
  "total_documents": 5,
  "successful_documents": 2,
  "failed_documents": 1,
  "remaining_documents": 2,
  "requested_by": "APEX_USER",
  "requested_at": "2026-03-21 14:10:01",
  "source_system": "APEX"
}
```

### Possible Status Values
- `PENDING`
- `IN_PROGRESS`
- `COMPLETED`
- `FAILED`
- `PARTIAL_SUCCESS`

### Not Found Response

```json
{
  "detail": "Batch not found."
}
```

---

## 3. `GET /batches/{batch_id}/documents`

Get all documents belonging to a batch and their statuses.

### Example Request

```http
GET /batches/4D90ABC123/documents
```

### Success Response

```json
{
  "batch_id": "4D90ABC123",
  "documents": [
    {
      "document_id": "4D90DOC001",
      "file_name": "Doc1.docx",
      "status": "COMPLETED",
      "doc_type_code": "DOCX",
      "module_code": "GLOBAL_DESIGN",
      "created_date": "2026-03-21 14:10:10",
      "last_updated_date": "2026-03-21 14:11:20"
    },
    {
      "document_id": "4D90DOC002",
      "file_name": "Doc2.docx",
      "status": "FAILED",
      "doc_type_code": "DOCX",
      "module_code": "GLOBAL_DESIGN",
      "created_date": "2026-03-21 14:10:20",
      "last_updated_date": "2026-03-21 14:10:59"
    }
  ]
}
```

### Not Found Response

```json
{
  "detail": "Batch not found."
}
```

---

## 4. `GET /documents/{document_id}/steps`

Get step-by-step processing logs for a document.

### Example Request

```http
GET /documents/4D90DOC001/steps
```

### Success Response

```json
{
  "document_id": "4D90DOC001",
  "steps": [
    {
      "log_id": "4D90LOG001",
      "step_name": "EXTRACT",
      "step_status": "COMPLETED",
      "step_sequence": 1,
      "message": "EXTRACT completed",
      "started_at": "2026-03-21 14:10:12",
      "ended_at": "2026-03-21 14:10:15",
      "duration_ms": 3000
    },
    {
      "log_id": "4D90LOG002",
      "step_name": "ANONYMIZE",
      "step_status": "COMPLETED",
      "step_sequence": 2,
      "message": "ANONYMIZE completed",
      "started_at": "2026-03-21 14:10:15",
      "ended_at": "2026-03-21 14:10:19",
      "duration_ms": 4000
    }
  ]
}
```

### Logged Steps
- `EXTRACT`
- `ANONYMIZE`
- `CHUNK`
- `EMBED`
- `STORE_VECTOR`

### Possible Step Status Values
- `STARTED`
- `COMPLETED`
- `FAILED`

---

## 5. `DELETE /documents/{document_id}`

Hard delete one document completely from all related tables.

### Purpose
Deletes document data from:
- `XXGSC_KM_CHUNK_VECTOR`
- `XXGSC_KM_DOCUMENT_CHUNK`
- `XXGSC_KM_INGESTION_STEP_LOG`
- `XXGSC_KM_DOCUMENT_VERSION`
- `XXGSC_KM_DOCUMENTS`

### Example Request

```http
DELETE /documents/4D90DOC001
```

### Success Response

```json
{
  "status": "ok",
  "document_id": "4D90DOC001",
  "deleted": {
    "chunk_vectors": 25,
    "document_chunks": 25,
    "step_logs": 5,
    "document_versions": 1,
    "documents": 1
  }
}
```

### Error Response

```json
{
  "detail": {
    "message": "Document deletion failed.",
    "error": "actual error message"
  }
}
```

---

## 6. `GET /db-test`

Simple database connectivity test.

### Example Request

```http
GET /db-test
```

### Success Response

```json
{
  "status": "ok",
  "db_user": "WKSP_GSCKMPVT2WS"
}
```

---

## 7. `GET /load-docs`

Lists locally available supported documents from the configured downloads folder.

### Example Request

```http
GET /load-docs
```

### Success Response

```json
{
  "total_documents": 2,
  "documents": [
    {
      "file": "Doc1.docx",
      "folder": "GlobalDesign",
      "path": "C:\\...\\Doc1.docx"
    }
  ]
}
```

---

# Current Ingestion Behavior Summary

## Duplicate document behavior
If same logical document is uploaded again:
- if file hash and content hash match -> status returned as `SKIPPED_DUPLICATE`
- if content changed -> old document becomes `SUPERSEDED`, old chunks become `SUPERSEDED`, old version becomes non-current, new version is created

## Step logging behavior
Each processed document logs:
- `EXTRACT`
- `ANONYMIZE`
- `CHUNK`
- `EMBED`
- `STORE_VECTOR`

---

# APEX Integration Section

This section lists the APIs APEX should use.

## APEX API 1: Submit batch

### Endpoint
```http
POST /ingest
```

### Use
APEX calls this when user submits one or multiple documents.

### Simplified request body for APEX
```json
{
  "requested_by": "APEX_USER",
  "source_system": "APEX",
  "anonymize_docs": true,
  "project": {
    "project_name": "KM_LOCAL_TEST_PROJECT",
    "geography_code": "IN",
    "vertical_code": "GSC",
    "engagement_type": "IMPLEMENTATION",
    "confidentiality": "INTERNAL"
  },
  "documents": [
    {
      "file_name": "DOM_Global_Design_ISSUE 1.0.docx",
      "file_path": "C:\\path\\to\\file.docx",
      "object_name": "GlobalDesign/DOM_Global_Design_ISSUE 1.0.docx",
      "bucket_name": "gsc-scm-loading-km-doc",
      "namespace_name": "ax4qsxvnsmtm",
      "module_code": "GLOBAL_DESIGN",
      "doc_type_code": "DOCX",
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
  ]
}
```

### What APEX should store from response
- `project_id`
- `batch_id`
- `run_id`

### Expected response
```json
{
  "status": "accepted",
  "message": "Batch accepted for background ingestion",
  "project_id": "...",
  "project_name": "...",
  "batch_id": "...",
  "run_id": "..."
}
```

---

## APEX API 2: Poll batch status

### Endpoint
```http
GET /batches/{batch_id}
```

### Use
APEX polls until batch status becomes:
- `COMPLETED`
- `FAILED`
- `PARTIAL_SUCCESS`

---

## APEX API 3: Show document-level status

### Endpoint
```http
GET /batches/{batch_id}/documents
```

### Use
Show status of each document in the uploaded batch.

---

## APEX API 4: Show per-document step logs

### Endpoint
```http
GET /documents/{document_id}/steps
```

### Use
Show where exactly a document failed or completed.

---

## APEX API 5: Delete a document

### Endpoint
```http
DELETE /documents/{document_id}
```

### Use
Completely remove a document and all related rows from the database.

---

# Suggested APEX Flow

1. User uploads one or multiple documents
2. APEX calls `POST /ingest`
3. APEX stores `batch_id` and `run_id`
4. APEX polls `GET /batches/{batch_id}`
5. APEX optionally calls `GET /batches/{batch_id}/documents`
6. APEX optionally calls `GET /documents/{document_id}/steps`
7. If user deletes a document, APEX calls `DELETE /documents/{document_id}`

---

# Notes

- Current background processing uses FastAPI `BackgroundTasks`
- This is good for current integration/testing
- If backend restarts during processing, in-progress work may be lost
- Later production enhancement can move this to a durable worker/queue model
- `POST /ingest` no longer downloads documents to the VM
- Chunk tuning is backend-controlled with fixed values optimized for current RAG flow