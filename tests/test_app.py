import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import app


class ParseEnvTests(unittest.TestCase):
    def test_parse_int_env_uses_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(app._parse_int_env("TEST_INT", 7), 7)

    def test_parse_int_env_rejects_non_integer(self):
        with mock.patch.dict(os.environ, {"TEST_INT": "abc"}, clear=True):
            with self.assertRaises(RuntimeError):
                app._parse_int_env("TEST_INT", 7)

    def test_parse_int_env_rejects_out_of_range(self):
        with mock.patch.dict(os.environ, {"TEST_INT": "9"}, clear=True):
            with self.assertRaises(RuntimeError):
                app._parse_int_env("TEST_INT", 7, max_value=5)

    def test_parse_choice_env_uses_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                app._parse_choice_env("TEST_PIPELINE", "range_map", {"range_map", "legacy"}),
                "range_map",
            )

    def test_parse_choice_env_rejects_invalid_value(self):
        with mock.patch.dict(os.environ, {"TEST_PIPELINE": "invalid"}, clear=True):
            with self.assertRaises(RuntimeError):
                app._parse_choice_env("TEST_PIPELINE", "range_map", {"range_map", "legacy"})


class LocalInputValidationTests(unittest.TestCase):
    def test_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                app._validate_local_input(path)

    def test_rejects_oversized_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "big.pdf"
            path.write_bytes(b"x" * (2 * 1024 * 1024))
            with mock.patch.object(app, "MAX_FILE_MB", 1):
                with self.assertRaises(ValueError):
                    app._validate_local_input(path)

    def test_accepts_supported_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            path.write_bytes(b"png")
            suffix, size = app._validate_local_input(path)
            self.assertEqual(suffix, ".png")
            self.assertEqual(size, 3)


class OutputPathValidationTests(unittest.TestCase):
    def test_rejects_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.md"
            out_path.write_text("existing", encoding="utf-8")
            with self.assertRaises(ValueError):
                app._validate_output_path(out_path, overwrite=False)

    def test_allows_existing_output_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.md"
            out_path.write_text("existing", encoding="utf-8")
            app._validate_output_path(out_path, overwrite=True)


class RangeHelperTests(unittest.TestCase):
    def test_chunk_items_even_split(self):
        items = [b"1", b"2", b"3", b"4"]
        self.assertEqual(app._chunk_items(items, 2), [[b"1", b"2"], [b"3", b"4"]])

    def test_chunk_items_with_remainder(self):
        items = [b"1", b"2", b"3", b"4", b"5"]
        self.assertEqual(app._chunk_items(items, 2), [[b"1", b"2"], [b"3", b"4"], [b"5"]])

    def test_chunk_items_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            app._chunk_items([b"1"], 0)

    def test_page_ranges_even_split(self):
        self.assertEqual(app._page_ranges(8, 2), [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_page_ranges_remainder(self):
        self.assertEqual(app._page_ranges(5, 2), [(0, 2), (2, 4), (4, 5)])

    def test_page_ranges_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            app._page_ranges(5, 0)

    def test_iter_page_blocks_range_continuity(self):
        blocks = list(app._iter_page_blocks_for_range(4, 8, ["A", "B", "C", "D"]))
        self.assertEqual(blocks[0][0], 5)
        self.assertEqual(blocks[-1][0], 8)
        self.assertEqual(blocks[0][1], "<!-- Page 5 -->\nA")
        self.assertEqual(blocks[-1][1], "<!-- Page 8 -->\nD")

    def test_iter_page_blocks_rejects_length_mismatch(self):
        with self.assertRaises(RuntimeError):
            list(app._iter_page_blocks_for_range(0, 2, ["only-one"]))


class MainPipelineTests(unittest.TestCase):
    def _make_fake_ocr(self):
        return SimpleNamespace(
            run_page=SimpleNamespace(remote=mock.Mock(return_value="image text")),
            run_page_batch=SimpleNamespace(map=mock.Mock(return_value=[])),
            run_pdf_range=SimpleNamespace(starmap=mock.Mock(return_value=iter([]))),
        )

    def test_main_uses_legacy_pipeline_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.md"
            input_path.write_bytes(b"pdf")

            fake_ocr = self._make_fake_ocr()
            fake_ocr.run_page_batch.map.return_value = [["p1", "p2", "p3"]]
            legacy_remote = mock.Mock(return_value=[b"a", b"b", b"c"])
            stage_remote = mock.Mock()

            with mock.patch.object(app, "PDF_PIPELINE", "legacy"), mock.patch.object(
                app, "TyphoonOCR", return_value=fake_ocr
            ), mock.patch.object(app, "pdf_to_page_images", SimpleNamespace(remote=legacy_remote)), mock.patch.object(
                app, "stage_pdf_input", SimpleNamespace(remote=stage_remote)
            ):
                app.main(file_path=str(input_path), output=str(output_path), overwrite=True)

            stage_remote.assert_not_called()
            legacy_remote.assert_called_once()
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("<!-- Page 1 -->", text)
            self.assertIn("<!-- Page 3 -->", text)

    def test_main_range_map_orchestration_and_output_continuity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.md"
            input_path.write_bytes(b"pdf")

            fake_ocr = self._make_fake_ocr()
            stage_remote = mock.Mock(return_value=("/inputs/run/input.pdf", 5))
            cleanup_remote = mock.Mock()
            captured = {}

            def starmap_side_effect(input_iterator, order_outputs=True):
                captured["args"] = list(input_iterator)
                captured["ordered"] = order_outputs
                return iter([["A", "B"], ["C", "D"], ["E"]])

            fake_ocr.run_pdf_range.starmap.side_effect = starmap_side_effect

            with mock.patch.object(app, "PDF_PIPELINE", "range_map"), mock.patch.object(
                app, "PAGE_BATCH_SIZE", 2
            ), mock.patch.object(app, "TyphoonOCR", return_value=fake_ocr), mock.patch.object(
                app, "stage_pdf_input", SimpleNamespace(remote=stage_remote)
            ), mock.patch.object(
                app, "cleanup_staged_pdf", SimpleNamespace(remote=cleanup_remote)
            ), mock.patch.object(
                Path, "replace", autospec=True, wraps=Path.replace
            ) as replace_mock:
                app.main(file_path=str(input_path), output=str(output_path), overwrite=True)

            stage_remote.assert_called_once()
            cleanup_remote.assert_called_once()
            self.assertTrue(captured["ordered"])
            self.assertEqual(
                captured["args"],
                [
                    ("/inputs/run/input.pdf", 0, 2, app.PDF_DPI),
                    ("/inputs/run/input.pdf", 2, 4, app.PDF_DPI),
                    ("/inputs/run/input.pdf", 4, 5, app.PDF_DPI),
                ],
            )
            self.assertEqual(replace_mock.call_count, 1)
            expected = (
                "<!-- Page 1 -->\nA\n\n---\n\n<!-- Page 2 -->\nB\n\n---\n\n<!-- Page 3 -->\nC\n\n---\n\n"
                "<!-- Page 4 -->\nD\n\n---\n\n<!-- Page 5 -->\nE\n"
            )
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected)

    def test_main_range_map_calls_cleanup_when_starmap_raises_mid_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.md"
            input_path.write_bytes(b"pdf")

            fake_ocr = self._make_fake_ocr()
            stage_remote = mock.Mock(return_value=("/inputs/run/input.pdf", 4))
            cleanup_remote = mock.Mock()

            def raising_stream():
                yield ["A", "B"]
                raise RuntimeError("ocr failed")

            fake_ocr.run_pdf_range.starmap.return_value = raising_stream()

            with mock.patch.object(app, "PDF_PIPELINE", "range_map"), mock.patch.object(
                app, "PAGE_BATCH_SIZE", 2
            ), mock.patch.object(app, "TyphoonOCR", return_value=fake_ocr), mock.patch.object(
                app, "stage_pdf_input", SimpleNamespace(remote=stage_remote)
            ), mock.patch.object(app, "cleanup_staged_pdf", SimpleNamespace(remote=cleanup_remote)):
                with self.assertRaisesRegex(RuntimeError, "ocr failed"):
                    app.main(file_path=str(input_path), output=str(output_path), overwrite=True)

            cleanup_remote.assert_called_once()
            self.assertFalse(output_path.exists())

    def test_main_range_map_preserves_primary_error_when_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.md"
            input_path.write_bytes(b"pdf")

            fake_ocr = self._make_fake_ocr()
            stage_remote = mock.Mock(return_value=("/inputs/run/input.pdf", 2))
            cleanup_remote = mock.Mock(side_effect=RuntimeError("cleanup failed"))

            def starmap_raises(_input_iterator, order_outputs=True):
                raise RuntimeError("ocr failed")

            fake_ocr.run_pdf_range.starmap.side_effect = starmap_raises

            with mock.patch.object(app, "PDF_PIPELINE", "range_map"), mock.patch.object(
                app, "TyphoonOCR", return_value=fake_ocr
            ), mock.patch.object(app, "stage_pdf_input", SimpleNamespace(remote=stage_remote)), mock.patch.object(
                app, "cleanup_staged_pdf", SimpleNamespace(remote=cleanup_remote)
            ):
                with self.assertRaisesRegex(RuntimeError, "ocr failed"):
                    app.main(file_path=str(input_path), output=str(output_path), overwrite=True)

            cleanup_remote.assert_called_once()


class ModelIntegrityTests(unittest.TestCase):
    def test_model_dir_integrity_accepts_valid_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            for filename in app.REQUIRED_MODEL_FILES:
                (model_dir / filename).write_text("{}", encoding="utf-8")
            (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")

            valid, reason = app._model_dir_integrity_status(model_dir)
            self.assertTrue(valid)
            self.assertEqual(reason, "ok")

    def test_model_dir_integrity_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            valid, reason = app._model_dir_integrity_status(model_dir)
            self.assertFalse(valid)
            self.assertIn("missing required file", reason)


if __name__ == "__main__":
    unittest.main()
