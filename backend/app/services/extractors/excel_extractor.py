from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


@dataclass
class ExcelRowRecord:
    row_number: int
    values: dict[str, Any]
    text: str


@dataclass
class ExcelSheetRecord:
    sheet_name: str
    state: str
    max_row: int
    max_column: int
    headers: list[str]
    row_count: int
    preview_rows: list[ExcelRowRecord]


@dataclass
class ExcelWorkbookRecord:
    file_path: str
    file_name: str
    file_type: str
    sheet_count: int
    sheets: list[ExcelSheetRecord]


@dataclass
class ExcelIngestionRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]


GROUPED_ROW_WINDOW = 10
GROUPED_ROW_OVERLAP = 2


def _clean_cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return value


def _normalize_header(value: Any, idx: int) -> str:
    cleaned = _clean_cell_value(value)
    if cleaned is None:
        return f"column_{idx}"
    return str(cleaned)


def _is_placeholder_header(header: str) -> bool:
    return header.startswith("column_")


def _prune_row_map(row_map: dict[str, Any]) -> dict[str, Any]:
    cleaned_map: dict[str, Any] = {}
    for key, value in row_map.items():
        cleaned_value = _clean_cell_value(value)
        if cleaned_value is None:
            continue
        cleaned_map[key] = cleaned_value
    return cleaned_map


def _is_likely_template_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.startswith("*") or text.startswith("#")


def _filter_meaningful_row_map(row_map: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in row_map.items():
        if value is None:
            continue
        if _is_placeholder_header(key) and _is_likely_template_value(value):
            continue
        filtered[key] = value
    return filtered


def _should_skip_row(row_map: dict[str, Any]) -> bool:
    if not row_map:
        return True

    values = list(row_map.values())
    non_empty_count = len(values)
    if non_empty_count == 0:
        return True

    template_like_count = sum(1 for value in values if _is_likely_template_value(value))
    placeholder_header_count = sum(1 for header in row_map if _is_placeholder_header(header))

    if non_empty_count >= 3 and template_like_count / non_empty_count >= 0.6:
        return True

    if placeholder_header_count == len(row_map) and template_like_count > 0:
        return True

    return False


def _meaningful_headers(headers: list[str], row_maps: list[dict[str, Any]]) -> list[str]:
    meaningful: list[str] = []
    for header in headers:
        if any(header in row_map for row_map in row_maps):
            meaningful.append(header)
    return meaningful


def _row_to_text(row_map: dict[str, Any]) -> str:
    parts = []
    for key, value in row_map.items():
        parts.append(f"{key}: {value}")
    return ", ".join(parts)


def extract_excel_workbook(file_path: str | Path, preview_limit: int | None = None) -> ExcelWorkbookRecord:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Unsupported Excel file type: {suffix}. Supported: .xlsx, .xlsm")

    workbook = load_workbook(filename=path, data_only=True, keep_vba=(suffix == ".xlsm"))
    sheets: list[ExcelSheetRecord] = []

    for worksheet in workbook.worksheets:
        rows = list(worksheet.iter_rows(values_only=True))

        header_row_index = None
        headers: list[str] = []
        preview_rows: list[ExcelRowRecord] = []
        retained_row_maps: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            cleaned = [_clean_cell_value(cell) for cell in row]
            if any(cell is not None for cell in cleaned):
                header_row_index = idx
                headers = [_normalize_header(cell, col_idx + 1) for col_idx, cell in enumerate(cleaned)]
                break

        if header_row_index is not None:
            for row_number, row in enumerate(rows[header_row_index:], start=header_row_index + 1):
                cleaned_values = [_clean_cell_value(cell) for cell in row]
                if not any(cell is not None for cell in cleaned_values):
                    continue

                padded_headers = headers[:]
                if len(cleaned_values) > len(padded_headers):
                    padded_headers.extend(
                        [f"column_{idx}" for idx in range(len(padded_headers) + 1, len(cleaned_values) + 1)]
                    )

                row_map = {
                    padded_headers[idx]: cleaned_values[idx] if idx < len(cleaned_values) else None
                    for idx in range(len(padded_headers))
                }
                row_map = _prune_row_map(row_map)
                row_map = _filter_meaningful_row_map(row_map)
                if _should_skip_row(row_map):
                    continue
                text = _row_to_text(row_map)
                if not text:
                    continue

                retained_row_maps.append(row_map)

                preview_rows.append(
                    ExcelRowRecord(
                        row_number=row_number,
                        values=row_map,
                        text=text,
                    )
                )
                if preview_limit is not None and len(preview_rows) >= preview_limit:
                    break

        headers = _meaningful_headers(headers, retained_row_maps)

        sheets.append(
            ExcelSheetRecord(
                sheet_name=worksheet.title,
                state=worksheet.sheet_state,
                max_row=worksheet.max_row,
                max_column=worksheet.max_column,
                headers=headers,
                row_count=len(preview_rows),
                preview_rows=preview_rows,
            )
        )

    workbook.close()

    return ExcelWorkbookRecord(
        file_path=str(path.resolve()),
        file_name=path.name,
        file_type=suffix.lstrip("."),
        sheet_count=len(sheets),
        sheets=sheets,
    )


def workbook_record_to_dict(record: ExcelWorkbookRecord) -> dict[str, Any]:
    return asdict(record)


def workbook_record_to_ingestion_records(record: ExcelWorkbookRecord) -> list[ExcelIngestionRecord]:
    ingestion_records: list[ExcelIngestionRecord] = []

    for sheet in record.sheets:
        step = max(1, GROUPED_ROW_WINDOW - GROUPED_ROW_OVERLAP)
        for start in range(0, len(sheet.preview_rows), step):
            row_group = sheet.preview_rows[start : start + GROUPED_ROW_WINDOW]
            if not row_group:
                continue
            grouped_values: list[dict[str, Any]] = []
            grouped_lines: list[str] = []
            row_numbers: list[int] = []

            for row in row_group:
                non_empty_values = {
                    key: value
                    for key, value in row.values.items()
                    if value is not None and not (_is_placeholder_header(key) and _is_likely_template_value(value))
                }
                if not non_empty_values:
                    continue

                grouped_values.append(non_empty_values)
                row_numbers.append(row.row_number)
                grouped_lines.append(f"Row {row.row_number}: {row.text}")

            if not grouped_values:
                continue

            value_keys: list[str] = []
            for value_map in grouped_values:
                for key in value_map.keys():
                    if key not in value_keys:
                        value_keys.append(key)

            non_placeholder_headers = [header for header in sheet.headers if header in value_keys or not _is_placeholder_header(header)]

            metadata = {
                "document_type": "excel",
                "file_name": record.file_name,
                "file_path": record.file_path,
                "file_type": record.file_type,
                "sheet_name": sheet.sheet_name,
                "sheet_state": sheet.state,
                "row_number": row_numbers[0],
                "row_start": row_numbers[0],
                "row_end": row_numbers[-1],
                "row_numbers": row_numbers,
                "headers": non_placeholder_headers,
                "values": grouped_values,
                "value_keys": value_keys,
                "max_row": sheet.max_row,
                "max_column": sheet.max_column,
                "chunk_type": "excel_row_group",
            }
            ingestion_records.append(
                ExcelIngestionRecord(
                    record_id=f"{record.file_name}:{sheet.sheet_name}:{row_numbers[0]}-{row_numbers[-1]}",
                    text=(
                        f"Workbook: {record.file_name} | "
                        f"Sheet: {sheet.sheet_name} | "
                        f"Rows: {row_numbers[0]}-{row_numbers[-1]} | "
                        + " || ".join(grouped_lines)
                    ),
                    metadata=metadata,
                )
            )

    return ingestion_records


def ingestion_records_to_dict(records: list[ExcelIngestionRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def workbook_record_to_text(record: ExcelWorkbookRecord) -> str:
    lines = [
        f"Workbook: {record.file_name}",
        f"Path: {record.file_path}",
        f"Type: {record.file_type}",
        f"Sheets: {record.sheet_count}",
    ]

    for sheet in record.sheets:
        lines.append("")
        lines.append(f"## Sheet: {sheet.sheet_name}")
        lines.append(f"State: {sheet.state}")
        lines.append(f"Dimensions: max_row={sheet.max_row}, max_column={sheet.max_column}")
        lines.append(f"Headers: {', '.join(sheet.headers) if sheet.headers else '[none detected]'}")
        lines.append(f"Preview row count: {sheet.row_count}")

        for row in sheet.preview_rows:
            lines.append(f"- Row {row.row_number}: {row.text}")

    return "\n".join(lines)


def ingestion_records_to_text(records: list[ExcelIngestionRecord]) -> str:
    lines = [f"Ingestion records: {len(records)}"]

    for record in records:
        metadata = record.metadata
        lines.append("")
        lines.append(f"## Record ID: {record.record_id}")
        lines.append(f"LLM Text: {record.text}")
        lines.append(
            "Metadata: "
            f"sheet_name={metadata.get('sheet_name')}, "
            f"row_number={metadata.get('row_number')}, "
            f"file_name={metadata.get('file_name')}"
        )
        lines.append(f"Values: {metadata.get('values')}")

    return "\n".join(lines)