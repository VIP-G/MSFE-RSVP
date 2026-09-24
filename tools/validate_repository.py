"""Run the complete local validation suite for the MSFE-RSVP repository."""

from pathlib import Path
import compileall
import os
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def report(name: str, passed: bool, detail: str = "") -> None:
    """Print one machine-readable validation result."""
    suffix = f": {detail}" if detail else ""
    print(f"[{ 'PASS' if passed else 'FAIL' }] {name}{suffix}")


def run_tests() -> None:
    """Run repository tests in a subprocess with this checkout on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError("repository unittest suite failed")


def check_repository_contents() -> None:
    """Reject local data, weights, outputs, and machine-specific paths."""
    forbidden_parts = {
        "opendet",
        "analysis_results",
        "prediction_results",
        "output",
    }
    forbidden_suffixes = {".pth", ".pt", ".ckpt", ".log", ".onnx"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in forbidden_parts for part in relative.parts):
            raise RuntimeError(f"forbidden repository path: {relative}")
        if path.is_file() and path.suffix in forbidden_suffixes:
            raise RuntimeError(f"forbidden artifact: {relative}")
    for path in ROOT.rglob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for machine_path in ("/ghData", "/home/ghzhang", "/workspace"):
            if machine_path in text:
                raise RuntimeError(
                    f"machine-specific path {machine_path!r} in {path}"
                )

    config_paths = list((ROOT / "configs/rec").rglob("*.yml"))
    expected = [ROOT / "configs/rec/uyghur/msfe_rsvp.yml"]
    if config_paths != expected:
        raise RuntimeError(f"unexpected recognition configs: {config_paths}")


def main() -> int:
    """Run validation and return a shell status code."""
    try:
        compiled = compileall.compile_dir(str(ROOT), quiet=1)
        report("Python compile", compiled)
        if not compiled:
            return 1

        config = yaml.safe_load(
            (ROOT / "configs/rec/uyghur/msfe_rsvp.yml").read_text()
        )
        report(
            "YAML parse",
            config["Architecture"]["Encoder"]["name"]
            == "ViTLearnableFusionRSVP",
        )
        run_tests()
        report("Component registry, CPU forward, CTC backward, checkpoint", True)
        check_repository_contents()
        report("Repository pollution check", True)
        return 0
    except Exception as error:
        report("Repository validation", False, str(error))
        return 1
    finally:
        for cache_dir in ROOT.rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
