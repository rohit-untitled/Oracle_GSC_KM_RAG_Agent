import tempfile
import unittest
from pathlib import Path

import xlrd
from openpyxl import Workbook

from app.services.extractors.excel_extractor import (
    _iter_xlrd_rows,
    extract_excel_workbook,
    workbook_record_to_ingestion_records,
    workbook_record_to_text,
)


class ExcelExtractorTests(unittest.TestCase):
    def _write_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Setup"
        sheet["A1"] = "Payables Configuration"
        sheet.append([])
        sheet.append(["*Application", "Status", "Amount", "Calculated"])
        sheet.append(["Payables", "Enabled", 42, "=C4*2"])
        sheet.append([])
        sheet.append(["Name", "Name", "Description"])
        sheet.append(["Primary", "Duplicate header value", "Duplicate headers are retained"])
        workbook.save(path)

    def test_extracts_xlsx_rows_headers_formulas_and_ingestion_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "setup.xlsx"
            self._write_workbook(path)

            record = extract_excel_workbook(path)
            self.assertEqual(record.file_type, "xlsx")
            self.assertEqual(record.sheet_count, 1)

            sheet = record.sheets[0]
            self.assertIn("*Application", sheet.headers)
            self.assertIn("Name_2", sheet.headers)
            self.assertIn(3, sheet.header_row_numbers)
            self.assertGreaterEqual(sheet.formula_cell_count, 1)

            formula_rows = [row for row in sheet.preview_rows if row.formulas]
            self.assertEqual(formula_rows[0].formulas["Calculated"], "=C4*2")

            text = workbook_record_to_text(record)
            self.assertIn("Payables Configuration", text)
            self.assertIn("Calculated formula: =C4*2", text)

            chunks = workbook_record_to_ingestion_records(record)
            self.assertTrue(chunks)
            self.assertEqual(chunks[0].metadata["document_type"], "excel")
            self.assertEqual(chunks[0].metadata["file_type"], "xlsx")
            self.assertIn("cell_references", chunks[0].metadata)

    def test_extracts_xlsm_without_macro_media_dependency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "setup.xlsm"
            self._write_workbook(path)

            record = extract_excel_workbook(path)
            self.assertEqual(record.file_type, "xlsm")
            self.assertEqual(record.sheets[0].sheet_name, "Setup")

    def test_xls_row_iterator_uses_one_based_excel_rows(self):
        class FakeCell:
            def __init__(self, ctype, value):
                self.ctype = ctype
                self.value = value

        class FakeWorkbook:
            datemode = 0

        class FakeSheet:
            nrows = 2
            ncols = 2

            def cell(self, row_idx, col_idx):
                values = [
                    [FakeCell(xlrd.XL_CELL_TEXT, "Header"), FakeCell(xlrd.XL_CELL_TEXT, "Date")],
                    [FakeCell(xlrd.XL_CELL_TEXT, "Value"), FakeCell(xlrd.XL_CELL_NUMBER, 1)],
                ]
                return values[row_idx][col_idx]

        rows = list(_iter_xlrd_rows(FakeWorkbook(), FakeSheet()))
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1][0].coordinate, "A1")
        self.assertEqual(rows[1][0], 2)
        self.assertEqual(rows[1][1][1].coordinate, "B2")


if __name__ == "__main__":
    unittest.main()
