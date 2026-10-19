import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublicSurfaceTest(unittest.TestCase):
    def test_weights_are_not_bundled(self):
        suffixes = {".safetensors", ".bin", ".pt", ".pth"}
        self.assertFalse(
            any(
                path.suffix in suffixes and ".venv" not in path.parts
                for path in ROOT.rglob("*")
            )
        )

    def test_legacy_names_are_absent(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "dlite").rglob("*.py")
        )
        self.assertNotIn("rnn_easy", sources)
        self.assertNotIn("markov_head", sources)
        self.assertNotIn("flashmtp", sources.lower())


if __name__ == "__main__":
    unittest.main()
