# Ingestion API Documentation

This document describes the finalized ingestion API design for the backend.

Base URL for local testing:

```text
http://127.0.0.1:8000
```

---

## Final Architecture

APEX is responsible for:
- creating project/batch/document metadata in DB
- uploading files to OCI Object Storage
- calling backend to start ingestion

Backend is responsible for:
- resolving document scope by `batch_id` and optional `document_id`s
- reading document metadata from DB
- downloading files from OCI Object Storage
- extracting, anonymizing, chunking, embedding, and storing results
- updating ingestion statuses and step logs

The backend uses configured Object Storage defaults from `.env`:

```env
BUCKET_NAME=gsc-scm-loading-km-doc
OCI_NAMESPACE=ax4qsxvnsmtm
```

---

## 1. `POST /ingestion/start`

Trigger ingestion for a full batch or a selected subset of documents.

### Request Headers

```http
Content-Type: application/json
```

### Request Body — Full Batch

```json
{
  "batch_id": "4D90ABC123",
  "requested_by": "APEX_USER",
  "source_system": "APEX",
  "anonymize_docs": true
}
```

### Request Body — Selected Documents

```json
{
  "batch_id": "4D90ABC123",
  "requested_by": "APEX_USER",
  "source_system": "APEX",
  "anonymize_docs": true,
  "documents": [
    { "document_id": "4D90DOC001" },
    { "document_id": "4D90DOC002" }
  ]
}
```

### Processing Rules
- `batch_id` is mandatory
- `documents` is optional
- if `documents` is omitted, backend processes all documents in the batch
- if `documents` is provided, backend processes only those `document_id`s
- all requested documents must belong to the same batch
- DB is the source of truth for document metadata

### Success Response

```json
{
  "status": "accepted",
  "message": "Ingestion request accepted. Processing started.",
  "batch_id": "4D90ABC123",
  "run_id": "4D90RUN001",
  "requested_by": "APEX_USER",
  "source_system": "APEX",
  "anonymize_docs": true,
  "selected_documents": 2
}
```

---

## 2. `GET /batches/{batch_id}`

Get batch summary and current processing status.

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

---

## 3. `GET /batches/{batch_id}/documents`

Get all documents belonging to a batch and their statuses.

---

## 4. `GET /documents/{document_id}/steps`

Get step-by-step processing logs for a document.

Logged steps:
- `EXTRACT`
- `ANONYMIZE`
- `CHUNK`
- `EMBED`
- `STORE_VECTOR`

---

## 5. `DELETE /documents/{document_id}`

Hard delete one document completely from all related tables.

---

## 6. `GET /db-test`

Simple database connectivity test.

---

## 7. `GET /sync-bucket`

Utility endpoint to download all objects from the configured bucket locally.
Not part of the main APEX ingestion trigger flow.

---

# Final APEX Flow

1. User creates/selects project in APEX
2. APEX uploads documents to OCI Object Storage
3. APEX saves batch/document metadata in DB
4. APEX calls `POST /ingestion/start`
5. Backend resolves documents from DB
6. Backend downloads objects from OCI Object Storage
7. Backend processes and stores chunks/vectors/statuses
8. APEX polls `GET /batches/{batch_id}`
9. APEX optionally calls:
   - `GET /batches/{batch_id}/documents`
   - `GET /documents/{document_id}/steps`

---

# Notes

- Current background processing uses FastAPI `BackgroundTasks`
- If backend restarts during processing, in-progress work may be lost
- Later production enhancement can move this to a durable worker/queue model
- `POST /ingestion/start` is now a lightweight trigger API
- Object Storage bucket/namespace are backend-configured defaults