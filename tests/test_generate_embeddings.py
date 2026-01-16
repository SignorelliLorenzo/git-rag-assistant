import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.embeddings.generate_embeddings import (
    ChunkRecord,
    embed_chunks,
    load_chunks,
    write_embeddings,
)


class GenerateEmbeddingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.chunks_path = Path(self.temp_dir.name) / "chunks.json"
        self.output_path = Path(self.temp_dir.name) / "embeddings.json"
        self.sample_chunks = [
            {
                "repo_id": "sample",
                "file_path": "src/main.py",
                "chunk_id": 0,
                "text": "print('hello')",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "repo_id": "sample",
                "file_path": "README.md",
                "chunk_id": 0,
                "text": "# Sample",
                "start_line": 1,
                "end_line": 1,
            },
        ]
        with self.chunks_path.open("w", encoding="utf-8") as fh:
            json.dump(self.sample_chunks, fh)

    def test_load_chunks_sorts_and_validates(self) -> None:
        records = load_chunks(self.chunks_path)
        self.assertEqual(len(records), 2)
        self.assertIsInstance(records[0], ChunkRecord)
        self.assertLessEqual(records[0].file_path, records[1].file_path)

    @mock.patch("src.embeddings.generate_embeddings.embed")
    def test_embed_and_write_embeddings(self, mock_embed) -> None:
        chunks = load_chunks(self.chunks_path)
        mock_embed.text.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        embeddings = embed_chunks(
            chunks,
            model_name="nomic-embed-text",
            task_type="search_document",
        )
        write_embeddings(chunks, embeddings, self.output_path)

        self.assertTrue(self.output_path.exists())
        output = json.loads(self.output_path.read_text())
        self.assertEqual(len(output), len(chunks))
        self.assertEqual(output[0]["embedding"], [0.1, 0.2])
        self.assertEqual(output[0]["id"], "README.md::#0")


if __name__ == "__main__":
    unittest.main()
