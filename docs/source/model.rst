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


value
-----
- The value model is a vision-language model used to predict the value of the current state. It is used to train VLA policies with the RECAP framework.
- More details can be found in the `pi*06 paper <https://www.pi.website/download/pistar06.pdf>`_.
- See the implementation in `src/opentau/policies/value/modeling_value.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/value/modeling_value.py>`_.
- Config selector: ``--policy.type=value``.
