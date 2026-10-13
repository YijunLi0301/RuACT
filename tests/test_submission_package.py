import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_submission_readme_present():
    readme = ROOT / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "Quick start" in text
    assert "experiments/main/medicine/4b/RuACT.sh" in text


def test_no_obsolete_generator_names():
    for path in ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        assert "codex" not in path.read_text().lower(), path


def test_no_unsupported_local_generator():
    assert not list(ROOT.rglob("policy_vllm_server.py"))
    for path in ROOT.rglob("refinement_sidecar.py"):
        assert "policy_server" not in path.read_text()


def test_no_constant_false_refinement_branches():
    for path in ROOT.rglob("single_pass_worker.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                assert not isinstance(node.test, ast.Constant), path
            if isinstance(node, ast.BoolOp):
                assert not any(
                    isinstance(v, ast.Constant) and v.value is False
                    for v in node.values
                ), path


def test_method_entrypoints_do_not_expose_secrets():
    spec = importlib.util.spec_from_file_location(
        "experiment_entry", ROOT / "experiments/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = json.loads((ROOT / "experiments/settings.json").read_text())
    settings["judge"]["api_key"] = "test-secret-sentinel"
    settings["generator"]["api_key"] = "test-generator-sentinel"
    for method in module.METHODS:
        base, output, command, sidecar, env = module.build(
            settings, method, preview=True
        )
        displayed = json.dumps([command, sidecar])
        assert "test-secret-sentinel" not in displayed
        assert "test-generator-sentinel" not in displayed
        assert base.is_dir()


def test_internal_symlinks():
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(ROOT.resolve())
            assert path.exists()
