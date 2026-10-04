import unittest

import numpy as np

from soulxpodcast.utils.segmented_workflow import (
    assemble_audio_and_manifest,
    normalize_sentence_table_rows,
    replace_segment_audio,
    split_script_into_sentence_units,
    table_to_units,
)


class SegmentedWorkflowTests(unittest.TestCase):
    def test_normalize_rows_from_gradio_payload_dict(self):
        payload = {
            "headers": ["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"],
            "data": [
                [0, 0, "S1", "Hello world.", 1.2],
                [None, None, None, None, None],
            ],
        }

        rows = normalize_sentence_table_rows(payload)
        units = table_to_units(rows)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text, "Hello world.")
        self.assertEqual(units[0].speaker, "S1")

    def test_normalize_rows_from_array_with_header_row(self):
        payload = [
            ["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"],
            [0, 0, "S1", "First.", 1.0],
            [1, 0, "S2", "Second.", 1.1],
        ]

        rows = normalize_sentence_table_rows(payload)
        units = table_to_units(rows)

        self.assertEqual([u.text for u in units], ["First.", "Second."])
        self.assertEqual([u.speaker for u in units], ["S1", "S2"])

    def test_normalize_rows_rejects_partially_populated_row(self):
        payload = [[0, 0, "S1", None, 1.0]]
        with self.assertRaisesRegex(ValueError, "Partially populated row 0: missing text"):
            normalize_sentence_table_rows(payload)

    def test_normalize_rows_supports_dataframe_like_records(self):
        class FakeDataframe:
            def to_dict(self, orient):
                if orient != "records":
                    raise ValueError("unexpected orient")
                return [
                    {"sentence_id": 0, "chunk_id": 0, "speaker": "S1", "text": "A.", "estimate_seconds": 1.0},
                    {"sentence_id": 1, "chunk_id": 0, "speaker": "S2", "text": "B.", "estimate_seconds": 1.0},
                ]

        rows = normalize_sentence_table_rows(FakeDataframe())
        units = table_to_units(rows)
        self.assertEqual([u.text for u in units], ["A.", "B."])

    def test_sentence_split_preserves_order_and_speaker(self):
        script = "[S1]Dr. Lee arrived at 3.14. Are you ready? Yes! [S2]Great."  # noqa: E501
        units = split_script_into_sentence_units(script, max_chunk_seconds=120.0)

        self.assertEqual([u.speaker for u in units], ["S1", "S1", "S1", "S2"])
        self.assertEqual(
            [u.text for u in units],
            [
                "Dr. Lee arrived at 3.14.",
                "Are you ready?",
                "Yes!",
                "Great.",
            ],
        )

    def test_chunking_respects_limit(self):
        script = "[S1]One short sentence. Another short sentence. Third short sentence."
        units = split_script_into_sentence_units(script, max_chunk_seconds=2.0)

        # Small limit should force multiple chunks while preserving sentence order
        self.assertEqual([u.sentence_id for u in units], [0, 1, 2])
        self.assertGreaterEqual(len({u.chunk_id for u in units}), 2)

    def test_replace_and_reassemble_updates_timing(self):
        segments = [
            {"sentence_id": 0, "chunk_id": 0, "speaker": "S1", "text": "A.", "audio": np.ones(24000, dtype=np.float32)},
            {"sentence_id": 1, "chunk_id": 0, "speaker": "S1", "text": "B.", "audio": np.ones(12000, dtype=np.float32)},
            {"sentence_id": 2, "chunk_id": 1, "speaker": "S2", "text": "C.", "audio": np.ones(6000, dtype=np.float32)},
        ]

        _, manifest_before = assemble_audio_and_manifest(segments, sample_rate=24000)
        self.assertAlmostEqual(manifest_before["segments"][1]["start_seconds"], 1.0)

        updated = replace_segment_audio(segments, sentence_id=1, new_audio=np.ones(24000, dtype=np.float32))
        _, manifest_after = assemble_audio_and_manifest(updated, sample_rate=24000)

        self.assertAlmostEqual(manifest_after["segments"][1]["duration_seconds"], 1.0)
        self.assertAlmostEqual(manifest_after["segments"][2]["start_seconds"], 2.0)


class DataframeConversionTests(unittest.TestCase):
    def test_sentence_dicts_to_rows_preserves_order_and_types(self):
        units_dict = [
            {
                "sentence_id": 0,
                "chunk_id": 0,
                "speaker": "S1",
                "text": "Welcome to the podcast.",
                "estimate_seconds": 1.2,
            },
            {
                "sentence_id": 1,
                "chunk_id": 0,
                "speaker": "S1",
                "text": "Today we discuss AI.",
                "estimate_seconds": 1.5,
            },
        ]

        rows = _sentence_dicts_to_rows(units_dict)

        self.assertEqual(
            rows,
            [
                [0, 0, "S1", "Welcome to the podcast.", 1.2],
                [1, 0, "S1", "Today we discuss AI.", 1.5],
            ],
        )
        self.assertIsInstance(rows[0][0], int)
        self.assertIsInstance(rows[0][1], int)
        self.assertIsInstance(rows[0][2], str)
        self.assertIsInstance(rows[0][3], str)
        self.assertIsInstance(rows[0][4], float)

    def test_review_dicts_to_rows_preserves_order_and_types(self):
        segments = [
            {
                "sentence_id": 99,
                "chunk_id": 3,
                "speaker": "S1",
                "text": "Hello.",
                "start_sample": 0,
                "end_sample": 24000,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "duration_seconds": 1.0,
            },
        ]

        rows = _review_dicts_to_rows(segments)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], [99, 3, "S1", "Hello.", 0, 24000, 0.0, 1.0, 1.0])
        self.assertIsInstance(rows[0][0], int)
        self.assertIsInstance(rows[0][4], int)
        self.assertIsInstance(rows[0][6], float)

    def test_sentence_dicts_to_rows_empty_input(self):
        self.assertEqual(_sentence_dicts_to_rows([]), [])

    def test_review_dicts_to_rows_empty_input(self):
        self.assertEqual(_review_dicts_to_rows([]), [])

    def test_review_dicts_to_rows_missing_fields_default_to_zero(self):
        segments = [
            {
                "sentence_id": 1,
                "speaker": "S2",
                "text": "Minimal.",
            },
        ]

        rows = _review_dicts_to_rows(segments)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1], -1)  # default chunk_id
        self.assertEqual(rows[0][2], "S2")
        self.assertEqual(rows[0][3], "Minimal.")
        self.assertEqual(rows[0][4], 0)  # default start_sample
        self.assertEqual(rows[0][5], 0)  # default end_sample
        self.assertEqual(rows[0][6], 0.0)  # default start_seconds
        self.assertEqual(rows[0][7], 0.0)  # default end_seconds
        self.assertEqual(rows[0][8], 0.0)  # default duration_seconds

    def test_sentence_dicts_to_rows_rounds_estimate_seconds(self):
        units_dict = [
            {
                "sentence_id": 0,
                "chunk_id": 0,
                "speaker": "S1",
                "text": "Hi.",
                "estimate_seconds": 1.23456789,
            },
        ]

        rows = _sentence_dicts_to_rows(units_dict)

        self.assertEqual(rows[0][4], 1.23)


def _sentence_dicts_to_rows(units):
    return [
        [
            int(item["sentence_id"]),
            int(item["chunk_id"]),
            str(item["speaker"]),
            str(item["text"]),
            round(float(item.get("estimate_seconds", 0.0)), 2),
        ]
        for item in units
    ]


def _review_dicts_to_rows(segments):
    return [
        [
            int(item["sentence_id"]),
            int(item.get("chunk_id", -1)),
            str(item["speaker"]),
            str(item["text"]),
            int(item.get("start_sample", 0)),
            int(item.get("end_sample", 0)),
            round(float(item.get("start_seconds", 0.0)), 6),
            round(float(item.get("end_seconds", 0.0)), 6),
            round(float(item.get("duration_seconds", 0.0)), 6),
        ]
        for item in segments
    ]


if __name__ == "__main__":
    unittest.main()