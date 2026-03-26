.. _robocasa:

.. _robocasa_client_gist: https://gist.github.com/akshay18iitg/4d299c135c2d384ceb9a283b745baa01

RoboCasa setup and rollout client
=================================

This page explains how to set up **RoboCasa** (kitchen simulation) alongside **OpenTau**, run the **policy WebSocket server** that serves an OpenTau checkpoint, and run the **rollout client** against that server.

The rollout client code **is not shipped in the OpenTau repository**. Use the reference implementation in `robocasa_client_gist`_ (RoboCasa policy client: ``client`` and ``client_async``).

.. note::
   Complete the base :doc:`/installation` steps first. RoboCasa itself is installed **outside** the OpenTau package. OpenTau provides the **policy server**; you run the **client** inside your RoboCasa install (files from the gist, or equivalent).

Overview
--------

The workflow is usually split across machines or terminals:

1. **OpenTau host** — runs the WebSocket policy server, loads ``policy.pretrained_path`` from a training config, and returns **action chunks** via MessagePack.
2. **RoboCasa host** — runs the kitchen sim, JPEG-encodes cameras, and talks to the server. Parallel rollouts use a threaded **async** client that **batches** observations for workers that need a new chunk.

**In this repo**

* ``opentau.scripts.robocasa.server`` — WebSocket server (single-observation or batched requests; replies are **action chunks** per request row).

**Outside this repo**

* ``robocasa.scripts.client`` / ``robocasa.scripts.client_async`` — reference rollout scripts from `robocasa_client_gist`_ (place them under your ``robocasa`` package tree or run them as you prefer).

Server dependencies (``websockets``, ``msgpack``) are in OpenTau’s ``pyproject.toml``. The server needs **OpenCV** (``cv2``) to decode JPEG camera inputs.


Prerequisites
-------------

**Hardware and OS**

* Linux with an NVIDIA GPU is recommended for both RoboCasa (MuJoCo) and OpenTau inference.
* Follow GPU guidance in :doc:`/installation`.

**Python**

* OpenTau targets **Python 3.10** (see ``requires-python`` in the repo root ``pyproject.toml``). Match or reconcile Python versions with your RoboCasa environment.

**RoboCasa simulation**

RoboCasa is not fully installed by ``pip install opentau``. Install the simulator and assets from upstream:

* `RoboCasa installation <https://robocasa.ai/docs/introduction/installation.html>`_

**OpenTau**

Install OpenTau as in :doc:`/installation` (e.g. ``uv sync`` or ``pip install -e .``).


Policy server (OpenTau)
-----------------------

The server listens on WebSocket and uses **MessagePack** for request and response bodies.

**Inference**

* Each successful call uses ``policy.sample_actions`` (not ``select_action``): the model predicts a **temporal chunk** of actions. The last dimension is trimmed or zero-padded to ``--robocasa_action_dim``.

**Requests**

* **Single observation:** top-level dict with ``images`` (JPEG bytes per camera name), ``state`` (list of floats), ``prompt`` (string).
* **Batch:** ``{ "batch": true, "items": [ { ... same fields ... }, ... ] }``.

**Responses**

* **Single:** one chunk as nested lists: ``[[float, ...], ...]`` — shape ``(T, action_dim)`` with ``T`` equal to the policy’s predicted horizon (e.g. ``n_action_steps``).
* **Batch:** ``[ chunk_0, chunk_1, ... ]`` — one chunk per ``items`` row, same order.

**Entry point**

.. code-block:: bash

   python -m opentau.scripts.robocasa.server \
       --config_path /path/to/train_config.json

**RoboCasa-specific flags** (must appear **before** normal OpenTau config flags; they are parsed first and stripped from ``sys.argv``):

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

The training config must define ``policy.pretrained_path`` and settings compatible with your checkpoint.


Rollout client (RoboCasa environment)
-------------------------------------

Get the client sources from `robocasa_client_gist`_.

Typical layout after copying into a RoboCasa checkout:

* ``robocasa/scripts/client.py`` — single-env style client (if provided in the gist).
* ``robocasa/scripts/client_async.py`` — threaded client that **batches** observations for workers that need a **new action chunk**, sends one WebSocket message per batch, receives one chunk per batch row, then **steps the simulator for every action in each chunk** before querying the server again.

If your PandaOmron-style env expects actions in a particular layout, the gist may include a ``convert_action_pi05`` helper (or equivalent); wire it to match ``create_env`` / your task.

**Example (async / batched client)**

.. code-block:: bash

   python -m robocasa.scripts.client_async ENV_NAME \
       --host localhost \
       --port 8765

Replace ``ENV_NAME`` with a registered RoboCasa kitchen task. Common options (see the gist for the exact CLI):

* ``--num-rollouts`` — total episodes.
* ``--num-parallel`` — parallel env threads (batch size is at most the count of workers requesting a chunk at once).
* ``--seed``, ``--split``, ``--output-dir``, ``--max-episode-steps``, ``--render``, ``--jpeg-quality``.

**Environment variables** (if supported by the gist client)

* ``ROBOCASA_POLICY_HOST`` — default host.
* ``ROBOCASA_POLICY_PORT`` — default port.


Protocol and outputs (summary)
------------------------------

* **Transport:** WebSocket binary frames, MessagePack.
* **Client → server (batch):** ``{ "batch": true, "items": [ { "images": {...}, "state": [...], "prompt": "..." }, ... ] }``.
* **Server → client (batch):** list of action chunks; each chunk is ``(T, action_dim)`` as nested lists.
* **Rollout output:** directory with ``rollouts.json`` and, when not rendering on screen, per-rollout MP4s per camera (behavior as implemented in the gist).

For server implementation details, see ``src/opentau/scripts/robocasa/server.py``. For client behavior and options, see `robocasa_client_gist`_.


Troubleshooting
---------------

* **Import errors for ``robocasa``** — Install RoboCasa per upstream docs; run the client from that environment.
* **Server JPEG decode errors** — Install OpenCV for Python on the server (``cv2``).
* **Port in use** — Change ``--robocasa_port`` / client ``--port``.
* **Action shape / chunk mismatch** — Align ``--robocasa_action_dim`` with training and env; ensure the client consumes **chunks** (multiple steps per server reply) if you use chunking inference.
