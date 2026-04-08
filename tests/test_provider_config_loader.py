from pathlib import Path

from core_lib.config.provider_config_loader import resolve_config_file_path


def test_resolve_config_file_path_finds_file_in_parent_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)

    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    cfg = project_root / "llm_providers.yaml"
    cfg.write_text("providers: []\n", encoding="utf-8")

    monkeypatch.chdir(src_dir)

    resolved = resolve_config_file_path("llm_providers.yaml")

    assert resolved is not None
    assert Path(resolved) == cfg
