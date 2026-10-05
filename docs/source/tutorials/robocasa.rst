.. _robocasa:

.. _robocasa_client_gist: https://gist.github.com/akshay18iitg/4d299c135c2d384ceb9a283b745baa01

RoboCasa setup and evaluation
=============================

`RoboCasa365 <https://robocasa.ai/>`_ (kitchen simulation) is a first-class
simulated-eval environment in OpenTau, alongside LIBERO. There are two ways to
evaluate a policy in RoboCasa:

#. **In-process vectorized eval** (recommended) — RoboCasa runs inside OpenTau's
   own process via ``opentau-eval``, exactly like LIBERO, with per-task success
   rates, video grids, and multi-rank task sharding for free. See
   :ref:`evaluating-robocasa`.
#. **External WebSocket server/client** — OpenTau runs a policy server and a
   separate rollout client drives a standalone RoboCasa checkout. Use this only
   when RoboCasa must run in its own process or on a different host. See
   `Alternative: external rollout client`_.

Installation
------------------------------------------

.. note::
   Complete the base :doc:`/installation` steps first.

For in-process eval the RoboCasa simulator installs **into OpenTau's
environment** as an extra — no separate upstream install is required:

.. code-block:: bash

   uv sync --extra robocasa      # or: uv sync --all-extras

This co-installs RoboCasa with LIBERO on the shared robosuite-1.5 stack (the
``robocasa`` and ``libero`` extras resolve together in one environment). Kitchen
assets (~5–10 GB) **auto-download on the first RoboCasa env build** into a
venv-external store — defaulting to ``$HF_OPENTAU_HOME/robocasa/assets``,
overridable with the ``ROBOCASA_ASSETS_ROOT`` env var or the ``env.assets_root``
config field. To warm the cache up front (e.g. before a multi-rank eval), run:

.. code-block:: bash

   python -m opentau.scripts.download_robocasa_assets

Run headless (no display) with ``MUJOCO_GL=egl``.

In-process eval
------------------------------------------

Set ``env.type`` to ``robocasa`` in your training/eval config and launch
``opentau-eval`` exactly as for LIBERO. The evaluation guide documents the
config fields and a complete example:

* :ref:`evaluating-robocasa` — the ``env`` block (task, cameras,
  ``metadata.robot_type`` / ``control_mode``) and how eval reports per-task
  results.
* ``configs/examples/pi05_robocasa_eval_config.json`` — a runnable example. It is a
  two-episode plumbing smoke and is **not** aiming for comparability (see below).

.. _robocasa_comparability:

Comparable success rates
------------------------------------------

``env.obj_registries`` defaults to ``["lightwheel"]``, because RoboCasa's own default
adds the ``objaverse`` object pack — a one-time ~30 GB download. The restriction is not
free, and the cost is not the one the download size suggests: the registry set feeds
RoboCasa's **scene generation**, so restricting it changes the generated scene, not just
which object meshes are placed in it. On ``CloseFridge`` / ``split="pretrain"`` at a fixed
reset seed, the fridge's own placement moves (``base_position`` y ``-3.100518`` with both
registries vs ``-3.262029`` with ``lightwheel`` alone — same task, same seed, same split).

A success rate measured under the default is therefore self-consistent — fine for A/B
comparisons between your own runs — but it is **not comparable to a published RoboCasa365
or leaderboard number**, for any policy. For a comparable run set:

.. code-block:: json

   "obj_registries": ["objaverse", "lightwheel"]

OpenTau prints a one-time warning at env construction when ``objaverse`` is absent, so an
incomparable run says so in its own log rather than leaving it to whoever reads the number
later.

.. warning::
   Success rates this repository produced **before** the flat-action-layout fix (see
   :ref:`the flat action layout <robocasa_action_layout>`) are lower than the same policy scores today, by a
   policy-dependent and unknown amount. Treat any earlier RoboCasa number as
   non-comparable — both to post-fix numbers and to each other.

Alternative: external rollout client
------------------------------------------

This mode runs OpenTau as a **policy WebSocket server** that serves an OpenTau
checkpoint, with a separate **rollout client** that drives a standalone RoboCasa
checkout. Use it only when RoboCasa must run in its own process or on a
different host than the policy server; for most evals the in-process path above
is simpler.

The rollout client code **is not shipped in the OpenTau repository**. Use the
reference implementation in `robocasa_client_gist`_ (RoboCasa policy client:
``client`` and ``client_async``).

.. note::
   The client runs inside *your own* RoboCasa environment (the files from the
   gist, or equivalent), which needs the ``robocasa`` package. Install OpenTau's
   ``robocasa`` extra in that venv, or follow the `upstream RoboCasa install
   <https://robocasa.ai/docs/introduction/installation.html>`_ if you keep
   RoboCasa separate. OpenTau itself provides the **policy server**.

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow is usually split across machines or terminals:

#. **OpenTau host** — runs the WebSocket policy server, loads
   ``policy.pretrained_path`` from a training config, and returns **action
   chunks** via MessagePack.
#. **RoboCasa host** — runs the kitchen sim, JPEG-encodes cameras, and talks to
   the server. Parallel rollouts use a threaded **async** client that
   **batches** observations for workers that need a new chunk.

**In this repo**

* ``opentau.scripts.robocasa.server`` — WebSocket server (single-observation or
  batched requests; replies are **action chunks** per request row).

**Outside this repo**

* ``robocasa.scripts.client`` / ``robocasa.scripts.client_async`` — reference
  rollout scripts from `robocasa_client_gist`_ (place them under your
  ``robocasa`` package tree or run them as you prefer).

Server dependencies (``websockets``, ``msgpack``) are in OpenTau's
``pyproject.toml``. The server needs **OpenCV** (``cv2``) to decode JPEG camera
inputs.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hardware and OS**

* Linux with an NVIDIA GPU is recommended for both RoboCasa (MuJoCo) and OpenTau
  inference.
* Follow GPU guidance in :doc:`/installation`.

**Python**

* OpenTau targets **Python 3.10** (see ``requires-python`` in the repo root
  ``pyproject.toml``). Match or reconcile Python versions with your RoboCasa
  environment.

Policy server (OpenTau)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The server listens on WebSocket and uses **MessagePack** for request and
response bodies.

**Inference**

* Each successful call uses ``policy.sample_actions`` (not ``select_action``):
  the model predicts a **temporal chunk** of actions. The last dimension is
  trimmed or zero-padded to ``--robocasa_action_dim``.

**Requests**

* **Single observation:** top-level dict with ``images`` (JPEG bytes per camera
  name), ``state`` (list of floats), ``prompt`` (string).
* **Batch:** ``{ "batch": true, "items": [ { ... same fields ... }, ... ] }``.

**Responses**

* **Single:** one chunk as nested lists: ``[[float, ...], ...]`` — shape
  ``(T, action_dim)`` with ``T`` equal to the policy's predicted horizon (e.g.
  ``n_action_steps``).
* **Batch:** ``[ chunk_0, chunk_1, ... ]`` — one chunk per ``items`` row, same
  order.

**Entry point**

.. code-block:: bash

   python -m opentau.scripts.robocasa.server \
       --config_path /path/to/train_config.json

**RoboCasa-specific flags** (must appear **before** normal OpenTau config flags;
they are parsed first and stripped from ``sys.argv``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Meaning
   * - ``--robocasa_host``
     - Bind address (default ``0.0.0.0``). Use ``127.0.0.1`` to listen only locally.
   * - ``--robocasa_port``
     - TCP port (default ``8765``).
   * - ``--robocasa_action_dim``
     - Flat action width for reply padding/trimming (default ``16``; align with RoboCasa env and training).
   * - ``--robocasa_torch_compile``
     - ``true`` / ``false`` — whether to compile ``sample_actions`` when supported (default ``true``).

**Example**

.. code-block:: bash

   python -m opentau.scripts.robocasa.server \
       --robocasa_host 0.0.0.0 \
       --robocasa_port 8765 \
       --robocasa_action_dim 16 \
       --config_path /path/to/train_config.json

The training config must define ``policy.pretrained_path`` and settings
compatible with your checkpoint.

Rollout client (RoboCasa environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the client sources from `robocasa_client_gist`_.

Typical layout after copying into a RoboCasa checkout:

* ``robocasa/scripts/client.py`` — single-env style client (if provided in the
  gist).
* ``robocasa/scripts/client_async.py`` — threaded client that **batches**
  observations for workers that need a **new action chunk**, sends one WebSocket
  message per batch, receives one chunk per batch row, then **steps the
  simulator for every action in each chunk** before querying the server again.

.. _robocasa_action_layout:

**Flat action layout.** RoboCasa's flat 12-D PandaOmron action is **EE-first**:

.. code-block:: text

   [0:3]   end-effector position
   [3:6]   end-effector rotation
   [6:7]   gripper
   [7:11]  base motion
   [11:12] control mode

That is the order ``robocasa.utils.env_utils.convert_action`` unpacks, and the order the
RoboCasa365 LeRobot datasets store in their ``action`` column — so it is what any policy
trained on that data emits. A client that unpacks a different order still runs: the arm
moves and episodes complete, only the success rate falls. OpenTau's in-process path
(``envs/robocasa.py::convert_action``) carried exactly that bug, routing the end-effector
command into base motion, and it is now pinned against both sources in
``tests/envs/test_robocasa_action_layout.py``.

If your PandaOmron-style env expects actions in a particular layout, the gist
may include a ``convert_action_pi05`` helper (or equivalent); wire it to match
``create_env`` / your task.

**Example (async / batched client)**

.. code-block:: bash

   python -m robocasa.scripts.client_async ENV_NAME \
       --host localhost \
       --port 8765

Replace ``ENV_NAME`` with a registered RoboCasa kitchen task. Common options
(see the gist for the exact CLI):

* ``--num-rollouts`` — total episodes.
* ``--num-parallel`` — parallel env threads (batch size is at most the count of
  workers requesting a chunk at once).
* ``--seed``, ``--split``, ``--output-dir``, ``--max-episode-steps``,
  ``--render``, ``--jpeg-quality``.

**Environment variables** (if supported by the gist client)

* ``ROBOCASA_POLICY_HOST`` — default host.
* ``ROBOCASA_POLICY_PORT`` — default port.

Protocol and outputs (summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Transport:** WebSocket binary frames, MessagePack.
* **Client → server (batch):** ``{ "batch": true, "items": [ { "images": {...}, "state": [...], "prompt": "..." }, ... ] }``.
* **Server → client (batch):** list of action chunks; each chunk is
  ``(T, action_dim)`` as nested lists.
* **Rollout output:** directory with ``rollouts.json`` and, when not rendering
  on screen, per-rollout MP4s per camera (behavior as implemented in the gist).

For server implementation details, see ``src/opentau/scripts/robocasa/server.py``.
For client behavior and options, see `robocasa_client_gist`_.

Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Import errors for** ``robocasa`` — Install the ``robocasa`` extra
  (``uv sync --extra robocasa``) in the environment running the client, or
  install RoboCasa per upstream docs.
* **Server JPEG decode errors** — Install OpenCV for Python on the server
  (``cv2``).
* **Port in use** — Change ``--robocasa_port`` / client ``--port``.
* **Action shape / chunk mismatch** — Align ``--robocasa_action_dim`` with
  training and env; ensure the client consumes **chunks** (multiple steps per
  server reply) if you use chunking inference.
