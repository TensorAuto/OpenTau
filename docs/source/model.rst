Models
======

This is the documentation for the supported models in OpenTau.

pi05
----
- Pi05 is a state of the art Vision-language-action flow model for general robot control. It supports both autoregressive discrete actions and flow matching continuous actions.
- More details can be found in the `paper <https://www.pi.website/download/pi05.pdf>`_.
- See the implementation in `src/opentau/policies/pi05/modeling_pi05.py`.
- Checkpoint of the model finetuned on the LIBERO dataset is available on Hugging Face: `TensorAuto/tPi05-Libero <https://huggingface.co/TensorAuto/tPi05-Libero>`_
- Disclaimer: Our implementation doesn't support sub-task prediction yet, as mentioned in the paper.


pi0
----
- Pi0 is a Vision-language-action flow model that only supports flow matching continuous actions.
- More details can be found in the `paper <https://www.pi.website/download/pi0.pdf>`_.
- See the implementation in `src/opentau/policies/pi0/modeling_pi0.py`.
- This model can be changed to pi0-star by changing the `advantage_always_on` flag to `on`/'use' in the config file.
- Checkpoint of the model finetuned on the LIBERO dataset is available on Hugging Face: `TensorAuto/tPi0-Libero <https://huggingface.co/TensorAuto/tPi0-Libero>`_

value
-----
- Value model is a Vision-language model used to predict the value of the current state. Its used to train VLA policies with RECAP framework.
- More details can be found in the `paper <https://www.pi.website/download/pistar06.pdf>`_.
- See the implementation in `src/opentau/policies/value/modeling_value.py`.
