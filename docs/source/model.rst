Models
======

This is the documentation for the supported models in OpenTau. Every model is a
LeRobot-compliant policy registered in
`src/opentau/policies/factory.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/factory.py>`_;
the name in the ``Config selector`` line below is the value you pass as
``--policy.type=...`` (or set as ``policy.type`` in a JSON config).

The action models fall into two backbone families:

- **PaliGemma family** (``pi0``, ``pi05``, ``pi05_mem``, ``pi07_paligemma``) — built on the
  ``paligemma_with_expert.py`` wrapper: a PaliGemma VLM (SigLIP image tower + Gemma text
  model) paired with a Gemma action expert.
- **Gemma 3 family** (``pi06``, ``pi07``) — built on the ``gemma3_with_expert.py`` wrapper:
  a Gemma 3 4B VLM (SigLIP-400m/14 vision + Gemma 3 text) paired with a Gemma action expert,
  at 448×448 image resolution.

Models that emit robot actions do so by **flow matching** (continuous action chunks),
**FAST discrete-action tokens** (autoregressive cross-entropy), or both jointly.


pi0
---
- π0 is a vision-language-action flow model that only supports flow-matching continuous actions, built on the PaliGemma backbone.
- More details can be found in the `pi0 paper <https://www.pi.website/download/pi0.pdf>`_.
- See the implementation in `src/opentau/policies/pi0/modeling_pi0.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi0/modeling_pi0.py>`_.
- This model can be turned into π0-star (π*₀) by setting the ``advantage`` flag in the config file; it is the policy used with the RECAP framework (see the ``moka_pot_RECAP_*`` checkpoints in the README).
- Config selector: ``--policy.type=pi0``.


pi05
----
- π0.5 is a state-of-the-art vision-language-action flow model for general robot control, on the PaliGemma backbone. It supports both autoregressive discrete (FAST) actions and flow-matching continuous actions.
- More details can be found in the `pi05 paper <https://www.pi.website/download/pi05.pdf>`_.
- See the implementation in `src/opentau/policies/pi05/modeling_pi05.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi05/modeling_pi05.py>`_.
- A checkpoint finetuned on the LIBERO dataset (discrete actions + knowledge insulation) is available on Hugging Face: `TensorAuto/tPi0.5-libero <https://huggingface.co/TensorAuto/tPi0.5-libero>`_. Additional RoboCasa and LIBERO checkpoints are linked from the README.
- The ``pi05_continuous_state`` policy name is a deprecated alias — use ``pi05`` with ``state_type='continuous'`` instead (this projects raw proprioceptive state into the model's latent dimension; see the `TensorAuto/pi05_libero_continuous_state <https://huggingface.co/TensorAuto/pi05_libero_continuous_state>`_ checkpoint).
- Config selector: ``--policy.type=pi05``.
- Disclaimer: Our implementation doesn't support sub-task prediction yet, as mentioned in the paper.


pi05_mem
--------
- π0.5-mem is a memory-augmented variant of π0.5. It keeps the same PaliGemma backbone and the same hybrid flow-matching + FAST-discrete action heads, but lets multiple past video frames and a temporal state sequence inform the current observation.
- The architecture follows the intuition of the `π-mem (MEM) paper <https://www.pi.website/download/Mem.pdf>`_, implemented on the PaliGemma backbone: the SigLIP image tower is wrapped with a ``SpaceTimeSiglipVideoEncoder`` that inserts causal space-time separable attention every few ViT layers. Crucially it **reuses the existing per-layer Q/K/V/O projections**, so the memory mechanism adds **zero new learnable parameters** — a standard ``pi05`` checkpoint loads directly with unchanged state-dict keys. Past-timestep tokens are dropped after the encoder, so the prefix keeps the same image-token budget as a single-frame VLA.
- It sees ``n_obs_steps`` historical frames (default 8) at a configurable temporal stride, and projects each timestep of robot state into its own continuous token.
- See the implementation in `src/opentau/policies/pi05_mem/modeling_pi05.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi05_mem/modeling_pi05.py>`_.
- Config selector: ``--policy.type=pi05_mem``.


pi06
----
- π0.6 inherits the π0.5 recipe but upgrades the architecture to the **Gemma 3 4B** backbone with a ~860M-parameter action expert, 448×448 image resolution, and 5-step flow matching (halved from π0.5's 10). It is co-trained with flow-matching continuous actions and FAST discrete-action cross-entropy, and uses knowledge insulation so action-loss gradients don't corrupt the pretrained VLM.
- It optionally restores π0.5's hierarchical high-level-subtask + low-level-action design via ``predict_response`` (off by default, since most LeRobot-style datasets carry no subtask annotations).
- More details can be found in the `π*0.6 paper <https://arxiv.org/abs/2511.14759>`_ ("π*0.6: a VLA That Learns From Experience"), and the π0.6 model card from Physical Intelligence.
- See the implementation in `src/opentau/policies/pi06/modeling_pi06.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi06/modeling_pi06.py>`_.
- To spin up a training run, start from `configs/examples/pi06_training_config.json <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/pi06_training_config.json>`_.
- Config selector: ``--policy.type=pi06``.
- Disclaimer: No TensorAuto-published π0.6 checkpoint exists yet ("coming soon" in the README); the policy is implemented and ready to train from scratch.


pi07
----
π0.7 splits the model into two components that are trained independently: a **high-level
planner** that proposes subgoals, and a **low-level controller** that executes them. The
current implementation pairs the Gemma 3 backbone with a SpaceTime SigLIP video encoder so
the controller can attend over temporal context. More details can be found in the
`π0.7 paper <https://www.pi.website/download/pi07.pdf>`_.

pi07_high_level
^^^^^^^^^^^^^^^
- The high-level planner is a Gemma 3 vision-language model that, given camera images, the language task, robot state, and past memory, autoregressively predicts an **updated memory string and the next subtask string**. It issues no robot actions — ``select_action`` / ``predict_action_chunk`` raise ``NotImplementedError`` — and its training loss is purely cross-entropy over the memory and subtask text. (The paired Gemma action expert is disabled to save memory.)
- See the implementation in `src/opentau/policies/pi07/high_level_planner/modeling_pi07_high_level.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi07/high_level_planner/modeling_pi07_high_level.py>`_.
- Config selector: ``--policy.type=pi07_high_level``.

pi07_low_level
^^^^^^^^^^^^^^
- The low-level controller is the vision-language-action half of the hierarchy. On the Gemma 3 backbone with a ``SpaceTimeSiglipVideoEncoder``, it turns multi-camera video history, a language prompt, optional high-level subtask / subgoal-image / metadata conditioning, and a temporal proprioceptive state sequence into continuous action chunks by flow matching, while also predicting FAST discrete-action tokens through the VLM backbone. It uses knowledge insulation and supports heterogeneous-dataset co-training (per-group normalization and projection heads).
- See the implementation in `src/opentau/policies/pi07/low_level/modeling_pi07_low_level.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi07/low_level/modeling_pi07_low_level.py>`_.
- To train the controller, start from `configs/examples/pi07_low_level_libero.json <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/pi07_low_level_libero.json>`_.
- Config selector: ``--policy.type=pi07_low_level``.
- Disclaimer: No TensorAuto-published π0.7 checkpoint exists yet ("coming soon" in the README); the policies are implemented and ready to train from scratch.


pi07_paligemma (legacy)
-----------------------
This is the older variant of π0.7. It follows the same
`π0.7 paper <https://www.pi.website/download/pi07.pdf>`_ intuition — a high-level
planner plus a low-level controller — but **swaps the Gemma 3 backbone for the PaliGemma
backbone** (``paligemma_with_expert.py``). It is kept for compatibility with older
checkpoints; new π0.7 work should generally target the Gemma 3 ``pi07`` implementation above.
The current ``pi07`` loaders can warm-start from these checkpoints by remapping
``paligemma_with_expert.*`` keys to ``gemma3_with_expert.*``.

pi07_paligemma_high_level_planner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The PaliGemma-backbone high-level planner: same role as ``pi07_high_level`` (autoregressively generate updated memory + the next subtask, emit no actions), on the PaliGemma VLM.
- See the implementation in `src/opentau/policies/pi07_paligemma/high_level_planner/modeling_pi07_high_level.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi07_paligemma/high_level_planner/modeling_pi07_high_level.py>`_.
- Config selector: ``--policy.type=pi07_paligemma_high_level_planner``.

pi07_paligemma_low_level
^^^^^^^^^^^^^^^^^^^^^^^^^
- The PaliGemma-backbone low-level controller: same role as ``pi07_low_level`` (flow-matching continuous actions + FAST discrete tokens, SpaceTime SigLIP video encoder, hierarchical subtask/subgoal/metadata conditioning, knowledge insulation), on the PaliGemma VLM. Its flow-matching action expert is inherited from `π0.5 <https://www.pi.website/download/pi05.pdf>`_.
- See the implementation in `src/opentau/policies/pi07_paligemma/low_level/modeling_pi07_low_level.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi07_paligemma/low_level/modeling_pi07_low_level.py>`_.
- Config selector: ``--policy.type=pi07_paligemma_low_level``.


cosmos3
-------
- cosmos3 is the π0.5 flow-matching recipe on a **frozen Qwen3-VL-32B backbone** — the **reasoning tower of NVIDIA `Cosmos3-Super <https://huggingface.co/nvidia/Cosmos3-Super>`_**, extracted into a standalone Qwen3-VL-32B checkpoint by ``src/opentau/scripts/extract_cosmos3_reasoner.py`` — paired with a custom **sub-1B Qwen3-style action expert** (``qwen3vl_with_expert.py``). Given camera images and a language prompt, the frozen reasoner encodes the observation once (prefix); the trainable expert cross-attends to the reasoner's per-layer key/value cache to denoise a continuous action chunk by flow matching.
- Continuous actions only (MSE flow matching) — no FAST discrete-action tokens and no subtask/response head. The backbone (vision tower + text tower) is fully frozen; only the action expert and its projections train (~0.9B parameters).
- The expert's KV heads (8) and head dim (128) match the Qwen3-VL text tower so its keys/values concatenate with the cached backbone KV at every layer; its query-head count is free. The shared multimodal RoPE (MRoPE) is computed by the backbone and reused by the expert.
- More details on the backbone: `Cosmos 3 technical report <https://arxiv.org/abs/2606.02800>`_. ``Cosmos3-Super`` is an interleaved Mixture-of-Transformers (a shared-attention autoregressive **reasoner** tower whose text config is Qwen3-VL-32B, plus a diffusion generation tower); cosmos3 keeps only the reasoner tower (text path ``mlp`` + the ``Qwen3VLVisionModel`` ``vision_encoder/``) and drops the generation tower.
- The extracted reasoner backbone is published at `TensorAuto/cosmos3-reason-32b <https://huggingface.co/TensorAuto/cosmos3-reason-32b>`_ (**private**; the default ``pretrained_backbone_repo_id``), so training pulls it directly given an HF token with TensorAuto read access. To reproduce or re-host it, run ``python -m opentau.scripts.extract_cosmos3_reasoner --cosmos3-path <Cosmos3-Super snapshot> --out-dir <reasoner-dir>`` (Cosmos3-Super is ungated; the script remaps the reasoner weights to a standard Qwen3-VL-32B checkpoint) and point ``--policy.pretrained_backbone_repo_id`` at the result.
- See the implementation in `src/opentau/policies/cosmos3/modeling_cosmos3.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/cosmos3/modeling_cosmos3.py>`_.
- To spin up a training run, start from `configs/examples/cosmos3_training_config.json <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/cosmos3_training_config.json>`_.
- Requires ``transformers>=4.57`` (the ``qwen3_vl`` model class). The extracted reasoner backbone is ~64 GB in bf16.
- Config selector: ``--policy.type=cosmos3``.
- Disclaimer: the reasoner *backbone* is published (private ``TensorAuto/cosmos3-reason-32b``), but no full cosmos3 *policy* checkpoint exists yet — the action expert is randomly initialized on top of the frozen reasoner and produced by training.

.. note::

   **Inference latency.** Because latency is weight-independent (it depends only
   on shapes / dtype / compile, not on trained values), a random-init cosmos3
   policy benches the same as a fully-trained one on the same hardware and
   config. Measured with ``benchmark_inference.py`` on **1× NVIDIA B200, bf16,
   batch 1**, real ``TensorAuto/cosmos3-reason-32b`` backbone (~33B, full 64
   layers; the action expert mirrors this depth with ``condition_on_layer=None``),
   3 cameras at 224×224 + prompt → a 50-action chunk with ``num_steps=10``
   flow-matching steps, one ``policy.sample_actions`` call costs:

   - **314.6 ms** eager (no ``torch.compile``);
   - **129.9 ms** with ``torch.compile`` (2.42× faster); and
   - **106.9 ms** with ``torch.compile(mode="reduce-overhead")`` / CUDA graphs
     (2.94× faster — the fastest setting).

   The call is one 32B backbone prefill plus 10 action-expert passes; the expert
   loop dominates and is what ``torch.compile`` accelerates (~25.8 → ~8.0 ms per
   denoising step), while the compute-bound prefill barely changes. See
   :doc:`tutorials/benchmarking` for the full methodology, the ``num_steps``
   decomposition, and the ``BENCH_*`` env-var knobs.

.. note::

   **Choosing the DeepSpeed ZeRO stage (ZeRO-2 vs ZeRO-3).** Because the ~33B
   reasoner backbone is *frozen*, ZeRO-2 shards only the ~0.9B action expert's
   optimizer state and **replicates** the bf16 backbone (~66 GB/rank), whereas
   ZeRO-3 also **shards** the backbone (~8 GB/rank) at the cost of a per-step
   all-gather of those 33B params. Benchmarked with
   ``opentau/scripts/profile_step.py`` on **8×A100-80GB** using
   ``configs/examples/cosmos3_training_config.json`` (frozen backbone + vision
   tower, ``gradient_checkpointing=True``, ``sdpa`` attention, ``chunk_size=10``,
   two 224 px cameras, ``gradient_accumulation_steps=1``).

   At a *matched* per-rank batch ZeRO-2 is ~1.3–1.4× faster — it skips ZeRO-3's
   per-step backbone all-gather (e.g. 4.3 vs 3.0 samples/s at batch 2; 15.7 vs
   11.8 at batch 8). But the fair question is throughput when **each stage is
   pushed to its own limit**, since they hit very different walls:

   .. list-table:: each stage at its limit (8×A100-80GB, peak reserved memory)
      :header-rows: 1
      :widths: 14 16 16 26 26

      * - stage
        - max per-rank batch
        - peak mem/rank
        - throughput
        - limited by
      * - ZeRO-2
        - ~32
        - 78.6 GB (≈ ceiling)
        - 43 samples/s
        - **memory** — replicated 66 GB backbone leaves no room to grow
      * - ZeRO-3
        - 192 (room to spare)
        - 55.4 GB
        - 58 samples/s (≈83 GPU-compute)
        - **compute**, not memory

   ZeRO-2 is *memory-bound*: the replicated backbone OOMs past per-rank batch ~32.
   ZeRO-3 *shards* the backbone, so it stays compute-bound long before memory
   fills — at batch 192 it still uses only ~55 GB while its GPU-compute throughput
   has already plateaued at ~83 samples/s (per-sample forward cost is flat from
   batch 64→192, so filling the remaining memory with an even larger batch does
   not raise it). On this tiny libero smoke dataset the pyav dataloader caps the
   *end-to-end* rate at ~58 samples/s (``dataload_wait`` ≈30 % of the step at
   batch 192); a production data pipeline removes that ceiling. Net: at full
   memory utilisation ZeRO-3 delivers ~1.3× (end-to-end here) to ~1.8×
   (GPU-compute) ZeRO-2's throughput **and** keeps headroom.

   **Recommendation:** for maximum training throughput on 8×A100-80GB prefer
   **ZeRO-3** (``configs/examples/accelerate_deepspeed_zero3_config.yaml``) — it
   reaches a higher ceiling because the sharded backbone lets you scale the batch
   until compute (not memory) saturates, and it leaves headroom for longer
   prompts, higher image resolution, larger ``chunk_size``, an unfrozen backbone,
   or GPUs smaller than 80 GB. **ZeRO-2**
   (``configs/examples/accelerate_deepspeed_config.yaml``) is the faster choice
   only when you are pinned to a small per-rank batch (it has lower per-step
   overhead), but the replicated backbone keeps it at ~95–98 % of memory and caps
   its peak throughput.

The training-time tensor flow, from the raw batch through the frozen reasoner
prefix and the trainable action expert to the flow-matching velocity head
(shapes shown at config defaults: ``chunk_size=50``, ``max_action_dim=32``,
``expert_hidden_size=1024``, 64 layers, 8 KV heads, ``head_dim=128``;
``S`` = prefix length, ``B`` = batch):

.. code-block:: text

     raw batch                          normalize_inputs / normalize_targets
     +--------------------------------------------------------------------+
     | images   (B, C, H, W)        VISUAL: identity   (per-dataset       |
     | state    (B, state_dim)      STATE : mean-std     stats, keyed     |
     | actions  (B, 50, act_dim)    ACTION: mean-std     by dataset_idx)  |
     | prompt   list[str]                                                 |
     +--------------------------------------------------------------------+
                |
                v   prepare_multimodal_inputs / prepare_state
     +--------------------------------------------------------------------+
     | each image -> resize 224x224 -> uint8 ; prompt -> <=256 tokens     |
     | Qwen3-VL chat template + processor (image tokens interleaved       |
     | with text):                                                        |
     |     input_ids (B, S) , attention_mask (B, S)                       |
     |     pixel_values , image_grid_thw                                  |
     | state -> pad to max_state_dim=32  ->  (B, 32)                      |
     +--------------------------------------------------------------------+
                |
                v
     ====== FROZEN Qwen3-VL-32B reasoner : prefix forward (no_grad) =======
     +--------------------------------------------------------------------+
     | get_rope_index -> MRoPE positions            (3, B, S)             |
     | run_prefix = stock Qwen3VLModel.forward:                           |
     |   vision tower -> image embeds scattered into the token            |
     |   sequence, deepstack, MRoPE, QK-norm, native causal mask          |
     | => per-layer KV cache: 64 x (K, V), each (B, 8, S, 128) --+        |
     +--------------------------------------------------------------------+
                |  (detached; backbone never reads the expert)   | cached_kv
                v                                                |
     ------------ flow-matching interpolation (suffix targets) ------------
     +--------------------------------------------------------------------+
     | noise ~ N(0,1)                      (B, 50, 32)                    |
     | time  ~ Beta(1.5,1.0)*0.999+0.001 in [0.001,1]  (B,) -> (B,50)     |
     |    (real-time-inference `delay` can pin the first `delay`          |
     |     chunk steps to t=0; max_delay defaults to 0 = none pinned)     |
     | x_t = t*noise + (1-t)*actions       (B, 50, 32)  expert input      |
     | u_t = noise - actions               (B, 50, 32)  MSE target        |
     +--------------------------------------------------------------------+
                |
                v   embed_suffix
     +--------------------------------------------------------------------+
     | action_in_proj(x_t)        (B, 50, 1024)                           |
     | state_proj(state)          (B,  1, 1024)  <- prepended token       |
     | embs = [state ; actions]   (B, 51, 1024)                           |
     | time -> sinusoid -> MLP -> adarms_proj -> adarms_cond (B,51,256)   |
     |    (state-token slot uses fixed t=1.0; 50 action slots use t)      |
     +--------------------------------------------------------------------+
                |
                v
     ============= TRAINABLE Qwen3 action expert  (64 layers) =============
     +--------------------------------------------------------------------+
     | for each layer i = 0..63:                                          |
     |   AdaRMS(hidden, adarms_cond) -> q/k/v, QK-norm, MRoPE             |
     |   K,V = concat( cached_kv[i] , expert K,V )  <- reasoner cache     |
     |           (B,8,S,128)        (B,8,51,128)                          |
     |   attn: expert queries over [prefix ; suffix] -> gated resid.      |
     |   AdaRMS -> SwiGLU MLP -> gated residual                           |
     | final AdaRMS norm                            (B, 51, 1024)         |
     +--------------------------------------------------------------------+
                |
                v   drop state token -> last 50 ; action_out_proj  [HEAD]
     +--------------------------------------------------------------------+
     | v_t = action_out_proj(out[:, -50:])    (B, 50, 32)  velocity       |
     +--------------------------------------------------------------------+
                |
                v
          flow_matching_masked_mse(v_t, u_t)  ->  MSE loss   (CE = 0)

     Inference swaps the interpolation block for Euler integration:
     x_t = noise at t=1, run the expert num_steps (10) times,
     x_t += dt*v_t, then unnormalize -> executed action chunk.


cosmos3_nano
------------
- cosmos3_nano is the cosmos3 recipe on the smaller **frozen Qwen3-VL-8B backbone** — the reasoning tower of NVIDIA `Cosmos3-Nano <https://huggingface.co/nvidia/Cosmos3-Nano>`_, extracted by the same ``src/opentau/scripts/extract_cosmos3_reasoner.py`` (the script reads the reasoner geometry from the snapshot's root ``config.json``, so it handles both family members). All modeling code is shared with cosmos3; the package only carries the nano defaults.
- The Nano reasoner keeps the exact KV interface the action expert cross-attends to — the same **8 KV heads × head_dim 128** as the 32B tower — so the expert architecture and every hard constraint carry over unchanged. The only geometry change is the backbone depth: **36 layers** instead of 64, which the default ``expert_num_hidden_layers=36`` mirrors (per-layer conditioning). With the inherited expert widths the trainable expert + projections total ~0.51B parameters. ``condition_on_layer`` works exactly as in cosmos3 (range ``[-36, 35]``).
- The extracted reasoner backbone is published at `TensorAuto/cosmos3-reason-8b <https://huggingface.co/TensorAuto/cosmos3-reason-8b>`_ (**private** — the default ``pretrained_backbone_repo_id``; the training environment needs an HF token with TensorAuto org read access, picked up from the ambient ``HF_TOKEN`` / ``~/.cache/huggingface/token``), ~17.5 GB in bf16 — it fits comfortably on a single 24–32 GB GPU for inference, where the 32B backbone needs an 80 GB-class card. To reproduce or re-host it, run the extraction script on the ungated ``nvidia/Cosmos3-Nano``.
- See the implementation in `src/opentau/policies/cosmos3_nano/modeling_cosmos3_nano.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/cosmos3_nano/modeling_cosmos3_nano.py>`_.
- To spin up a training run, start from `configs/examples/cosmos3_nano_training_config.json <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/cosmos3_nano_training_config.json>`_.
- Config selector: ``--policy.type=cosmos3_nano``.
- Disclaimer: as with cosmos3, only the reasoner *backbone* is published — no trained cosmos3_nano *policy* checkpoint exists yet; the action expert is randomly initialized and produced by training.


value
-----
- The value model is a vision-language model used to predict the value of the current state. It is used to train VLA policies with the RECAP framework.
- More details can be found in the `pi*06 paper <https://www.pi.website/download/pistar06.pdf>`_.
- See the implementation in `src/opentau/policies/value/modeling_value.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/value/modeling_value.py>`_.
- Config selector: ``--policy.type=value``.
