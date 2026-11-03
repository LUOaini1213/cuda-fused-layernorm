"""README numbers, reproduction notebook and requirement files (CPU only)."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    spec = importlib.util.spec_from_file_location("check_readme_numbers", ROOT / "check_readme_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadmeNumberTests(unittest.TestCase):
    def test_readme_numbers_match_committed_results(self):
        self.assertEqual(load_checker().check(), [])


class NotebookTests(unittest.TestCase):
    def test_notebook_runs_repo_scripts_instead_of_embedded_copies(self):
        nb = json.loads((ROOT / "cloud_gpu_verify.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn("git clone", code)
        self.assertIn("requirements-gpu.txt", code)
        for script in ("python bench3.py", "python bench_e2e.py", "python stability.py"):
            self.assertIn(script, code)
        for embedded in ("files['", "%%writefile", "__global__", "gemm_cliff"):
            self.assertNotIn(embedded, code)


class RequirementFileTests(unittest.TestCase):
    def test_requirement_files_are_ascii_and_pinned(self):
        for name in ("requirements.txt", "requirements-gpu.txt"):
            text = (ROOT / name).read_bytes().decode("ascii")
            self.assertIn("torch==", text, name)
        gpu = (ROOT / "requirements-gpu.txt").read_text(encoding="ascii")
        header = (ROOT / "results_t4/t4_three_way_and_e2e.txt").read_text(encoding="utf-8").splitlines()[0]
        # The pins must be the versions printed by the T4 run that produced the README table.
        for pin, printed in (("torch==2.11.0+cu128", "torch 2.11.0+cu128"),
                             ("triton==3.6.0", "triton 3.6.0"),
                             ("cupy-cuda12x==14.0.1", "cupy 14.0.1")):
            self.assertIn(pin, gpu)
            self.assertIn(printed, header)


if __name__ == "__main__":
    unittest.main()
