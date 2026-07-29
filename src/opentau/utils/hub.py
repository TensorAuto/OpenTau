# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for interacting with the Hugging Face Hub.

This module provides the HubMixin class which enables objects to be saved to
and loaded from the Hugging Face Hub, similar to ModelHubMixin but with fewer
assumptions about the object type, plus the ``repo_id@revision`` spec parser
every checkpoint loader goes through.
"""

import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Type, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args

T = TypeVar("T", bound="HubMixin")

# What the left side of an "@" must look like for us to treat the spec as a hub
# id rather than a filesystem path: an optional single "<namespace>/" followed by
# a repo name. Deliberately laxer than `huggingface_hub.utils.validate_repo_id`
# (which rejects "--" and a ".git" suffix) because a slugified wandb run name can
# legitimately produce those — the Hub itself is the authority on whether an id
# resolves; all we need here is to tell an id from a path. The leading character
# class is what rules out "/abs/path@x", "./rel@x", "~/home@x" and "..\win@x",
# and the single optional "/" rules out any deeper path.
_HUB_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._\-]*(?:/[A-Za-z0-9._\-]+)?$")


def split_repo_revision(
    pretrained_name_or_path: str | Path | None,
    revision: str | None = None,
) -> tuple[str | None, str | None]:
    """Split a ``repo_id@revision`` checkpoint spec into ``(repo_id, revision)``.

    Training checkpoints are published as one HF repo per run with each saved step
    as a git tag, so ``--policy.path=TensorAuto/<runid>-<slug>@6000`` selects the
    ``6000`` tag and a bare ``TensorAuto/<runid>-<slug>`` selects ``main`` — the
    highest step. Applies to ``--policy.path``, ``--config_path`` and
    ``policy.pretrained_path`` alike, since all three land in a ``from_pretrained``.

    A filesystem path is **never** split: local checkpoint directories legitimately
    contain "@". An existing path short-circuits outright, and a path-shaped string
    that does not exist (a moved checkpoint) still fails on the real path rather
    than as a bogus hub lookup.

    Idempotent — ``split(*split(x)) == split(x)`` — which is what lets an outer
    ``from_pretrained`` split a spec and forward ``(repo_id, revision)`` to an
    inner one that splits again.

    Args:
        pretrained_name_or_path: A hub id, a ``repo_id@revision`` spec, a local
            path, or None.
        revision: An explicitly-passed revision, if any.

    Returns:
        ``(repo_id_or_path, revision)``. The path is returned unchanged whenever
        it is not a hub spec.

    Raises:
        ValueError: If the spec pins a revision via "@" and a *different*
            ``revision`` was also passed.
    """
    if pretrained_name_or_path is None:
        return None, revision
    spec = str(pretrained_name_or_path)
    if not spec:
        return spec, revision

    # An existing local path wins outright. `exists`, not `isdir`:
    # `TrainPipelineConfig.from_pretrained` also accepts a path *to* a
    # train_config.json, not just the directory holding one.
    if os.path.exists(spec):
        return spec, revision

    # First "@" only: a repo id cannot contain one, but a git ref can
    # (e.g. "refs/pr/1", or a branch literally named "a@b").
    head, sep, tail = spec.partition("@")
    if not sep or not tail:
        return spec, revision
    if not _HUB_REPO_ID_RE.match(head) or os.path.exists(head):
        return spec, revision

    # Both an explicit `revision=` and an "@rev" suffix. Agreeing is fine (that is
    # the idempotent re-entrant case); disagreeing is not. Silently preferring
    # either one would serve weights the caller did not ask for.
    if revision is not None and revision != tail:
        raise ValueError(
            f"Conflicting revisions for {spec!r}: the path pins '@{tail}' but "
            f"revision={revision!r} was also passed. Specify only one."
        )
    return head, tail


def format_repo_revision(repo_id: str | Path | None, revision: str | None) -> str:
    """Re-join a ``(repo_id, revision)`` pair — the inverse of :func:`split_repo_revision`.

    For error messages and for naming output directories after the exact source
    that was loaded.
    """
    return f"{repo_id}@{revision}" if revision else str(repo_id)


class HubMixin:
    """
    A Mixin containing the functionality to push an object to the hub.

    This is similar to huggingface_hub.ModelHubMixin but is lighter and makes less assumptions about its
    subclasses (in particular, the fact that it's not necessarily a model).

    The inheriting classes must implement '_save_pretrained' and 'from_pretrained'.
    """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        Save object in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the object will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your object to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the card template to customize the card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~HubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save object (weights, files, etc.)
        self._save_pretrained(save_directory)

        # push to the Hub if required
        if push_to_hub:
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, card_kwargs=card_kwargs, **push_to_hub_kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your object.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the object files will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """
        Download the object from the Huggingface Hub and instantiate it.

        Args:
            pretrained_name_or_path (`str`, `Path`):
                - Either the `repo_id` (string) of the object hosted on the Hub, e.g. `lerobot/diffusion_pusht`.
                - Or a path to a `directory` containing the object files saved using `.save_pretrained`,
                    e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the files from the Hub, overriding the existing cache.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the object during initialization.
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether the repository created should be private.
                If `None` (default), the repo will be public unless the organization's default is private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.
            card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the card template to customize the card.

        Returns:
            The url of the commit of your object in the given repository.
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        if commit_message is None:
            if "Policy" in self.__class__.__name__:
                commit_message = "Upload policy"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        # Push the files to the repo in a single commit
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, card_kwargs=card_kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )
