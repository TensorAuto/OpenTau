.. _robocasa:

RoboCasa setup and rollout client
=================================

This page explains how to set up **RoboCasa** (kitchen simulation) alongside **OpenTau**, run the **policy WebSocket server** that serves an OpenTau checkpoint, and run the **batched rollout client** that steps parallel MuJoCo environments and queries the policy in batches.

.. note::
   Complete the base :doc:`/installation` steps first. RoboCasa itself is installed **outside** the OpenTau package; OpenTau provides the client and server glue only.

Overview
--------

The workflow is split across machines or terminals:

1. **Simulation + client** — RoboCasa environments, JPEG encoding, and episode logging run where you have the ``robocasa`` Python package and assets (often the same machine as the server for local testing).
2. **Policy server** — Loads ``policy.pretrained_path`` from an OpenTau training config and answers WebSocket requests with MessagePack-encoded actions.

OpenTau ships:

* ``opentau.scripts.robocasa.server`` — async WebSocket server (single-observation and **batched** requests).
* ``opentau.scripts.robocasa.client`` — threaded client that batches observations from multiple env workers per timestep.

Dependencies used by these modules (``websockets``, ``msgpack``) are declared in OpenTau’s ``pyproject.toml``. The server also needs **OpenCV** for JPEG decode on the policy side (``opencv-python`` or ``opencv-python-headless`` is already a core OpenTau dependency).


Prerequisites
-------------

**Hardware and OS**

* Linux with an NVIDIA GPU is recommended for both RoboCasa (MuJoCo) and OpenTau inference.
* Follow GPU guidance in :doc:`/installation`.

**Python**

* OpenTau currently targets **Python 3.10** (see ``requires-python`` in the repo root ``pyproject.toml``). Use the same interpreter for OpenTau and for the environment where you install RoboCasa, or ensure compatibility between the two stacks.

**RoboCasa simulation**

RoboCasa is not installed by ``pip install opentau``. Install the simulator and assets from the **upstream project**:

* `RoboCasa installation <https://robocasa.ai/docs/introduction/installation.html>`_

Typical steps include installing ``robosuite`` (often from source), then ``robocasa`` in editable mode, then running asset download scripts (kitchen assets can be large). Always refer to the official docs for the version you use.

**OpenTau**

Install OpenTau from source or PyPI as in :doc:`/installation`. Ensure your environment has the packages required by the scripts above (sync with ``uv sync`` or ``pip install -e .`` from the repo).


Policy server (OpenTau)
-----------------------

The server listens on a WebSocket port and speaks MessagePack. It accepts either a **single** observation dict or a **batch** payload ``{ "batch": true, "items": [ ... ] }`` for parallel clients.

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
     - Flat action size passed to the policy and validation (default ``16``; align with your RoboCasa / training setup).
   * - ``--robocasa_torch_compile``
     - ``true`` / ``false`` — whether to compile ``sample_actions`` when supported (default ``true``).
   * - ``--robocasa_use_stub``
     - ``true`` to use a small random policy instead of loading ``policy.pretrained_path`` (useful for wiring tests without weights).

**Example** with explicit host and port:

.. code-block:: bash

   python -m opentau.scripts.robocasa.server \
       --robocasa_host 0.0.0.0 \
       --robocasa_port 8765 \
       --robocasa_action_dim 16 \
       --config_path /path/to/train_config.json

The training config must define ``policy.pretrained_path`` and compatible policy settings unless you use ``--robocasa_use_stub=true``.


Rollout client (RoboCasa + OpenTau)
-----------------------------------

Copy the code from `opentau.scripts.robocasa.client` to `robocasa.scripts.client` and modify the code to fit the needs. Run the following command in robocasa environment:

Add this function in `robocasa.utils.env_utils`:

.. code-block:: python

    def convert_action_pi05(action):
        """
        Converts input action (np.array) to format expected by gym env (dict)
        """
        action = action.copy()
        output_action = {
            "action.end_effector_position": action[5:8],
            "action.end_effector_rotation": action[8:11],
            "action.gripper_close": action[11:12],
            "action.base_motion": action[0:4],
            "action.control_mode": action[4:5],
        }
        return np.concatenate([v for k,v in output_action.items()], axis=-1)

Run the client **after** the server is listening. It registers a RoboCasa task name, spawns one thread per parallel worker (up to ``--num-parallel``), batches observations for each timestep, and writes ``rollouts.json`` plus optional per-camera videos.

**Entry point**

.. code-block:: bash

   python -m opentau.scripts.robocasa.client ENV_NAME \
       --host localhost \
       --port 8765

Replace ``ENV_NAME`` with a registered RoboCasa kitchen task class name (same as other RoboCasa tooling).

**Useful options**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Meaning
   * - ``--num-rollouts``
     - Total episodes (default ``1``).
   * - ``--num-parallel``
     - Parallel env threads (capped by ``--num-rollouts``); batch size per step is at most this value.
   * - ``--seed``
     - Base seed; rollout ``i`` uses ``seed + i``.
   * - ``--split``
     - Dataset split for ``create_env`` (``all``, ``pretrain``, or ``target``).
   * - ``--output-dir``
     - Root for ``rollouts.json`` and ``rollout_*_seed_*`` video folders (default: auto-generated under cwd).
   * - ``--max-episode-steps``
     - Step cap per episode (default ``1500``).
   * - ``--render``
     - On-screen rendering; disables saved videos.

**Environment variables**

* ``ROBOCASA_POLICY_HOST`` — default for ``--host`` (default ``localhost``).
* ``ROBOCASA_POLICY_PORT`` — default for ``--port`` (default ``8765``).


Protocol and outputs (short)
------------------------------

* **Transport:** WebSocket binary frames, MessagePack payloads.
* **Client → server (batched):** ``{ "batch": true, "items": [ { "images": { camera_name: jpeg_bytes, ... }, "state": [...], "prompt": "..." }, ... ] }``.
* **Server → client:** A list of flat action lists, one per item, same order as ``items``.
* **Client output:** A directory containing ``rollouts.json`` (summary and per-rollout ``seed``, ``length``, ``success``) and, when not using ``--render``, MP4 files per camera under ``rollout_*`` subfolders.

For full behavioral details (variable batch size as workers finish, JPEG quality, ``ping_timeout``), see the module docstrings in ``src/opentau/scripts/robocasa/client.py`` and ``src/opentau/scripts/robocasa/server.py``.


Troubleshooting
---------------

* **Import errors for ``robocasa``** — Install and register RoboCasa per upstream docs; the client imports ``robocasa`` and ``robocasa.utils.env_utils``.
* **Server fails on JPEG decode** — Install OpenCV for Python on the server host (``cv2``); without it, JPEG decoding raises at runtime.
* **Port already in use** — Change ``--robocasa_port`` / ``--port`` or stop the conflicting process.
* **Action dimension mismatches** — Align ``--robocasa_action_dim`` with the policy and environment (e.g. PandaOmron / ``convert_action_pi05`` expectations in the client).
