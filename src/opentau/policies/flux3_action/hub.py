# Copyright 2026 Black Forest Labs. All rights reserved.
"""Resolve one policy package without downloading other embodiments or variants."""

from pathlib import Path, PurePosixPath


def looks_like_local_path(source: str) -> bool:
    """A spec that reads as a filesystem path rather than a Hub ``namespace/name``: an explicit prefix,
    more than one slash, a file suffix, or a first component that exists on disk (``outputs/...``)."""
    parts = PurePosixPath(source).parts
    return (
        source.startswith(("/", ".", "~"))
        or source.count("/") != 1
        or source.endswith((".safetensors", ".json", ".pt", ".bin"))
        or (bool(parts) and Path(parts[0]).exists())
    )


def resolve_policy_directory(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Path:
    """Resolve a local package or a Hub repository at one snapshot revision."""
    folder = PurePosixPath(subfolder or "")
    if folder.is_absolute() or ".." in folder.parts:
        raise ValueError("subfolder must be a relative path within the policy repository")
    path = Path(source).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError("revision applies to a Hub repository, not a local directory")
        package = path / folder
    else:
        if isinstance(source, Path) or looks_like_local_path(str(source)):
            raise FileNotFoundError(
                f"no such policy directory: {path}. Paths are relative to the working directory; download the "
                f"package first (docs/setup.md), or pass a Hub repository as namespace/name to download it."
            )
        from huggingface_hub import snapshot_download

        prefix = f"{folder}/" if folder.parts else ""
        names = (
            "config.json",
            "config.native.json",
            "manifest.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_*.safetensors",
            "policy_postprocessor_step_*.safetensors",
        )
        snapshot = snapshot_download(
            repo_id=str(source), revision=revision, allow_patterns=[prefix + name for name in names]
        )
        package = Path(snapshot) / folder
    if not (package / "config.json").is_file():
        raise FileNotFoundError(package / "config.json")
    return package
