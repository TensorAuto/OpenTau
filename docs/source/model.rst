Models
======

This is the documentation for the supported models in OpenTau.

pi05
----
- Pi05 is a state of the art vision-language-action flow model for general robot control. It supports both autoregressive discrete actions and flow matching continuous actions.
- More details can be found in the `pi05 paper <https://www.pi.website/download/pi05.pdf>`_.
- See the implementation in `src/opentau/policies/pi05/modeling_pi05.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi05/modeling_pi05.py>`_.
- Checkpoint of the model finetuned on the LIBERO dataset is available on Hugging Face: `TensorAuto/tPi0,5-libero <https://huggingface.co/TensorAuto/tPi0.5-libero>`_
- Disclaimer: Our implementation doesn't support sub-task prediction yet, as mentioned in the paper.


pi0
----
- Pi0 is a vision-language-action flow model that only supports flow matching continuous actions.
- More details can be found in the `pi0 paper <https://www.pi.website/download/pi0.pdf>`_.
- See the implementation in `src/opentau/policies/pi0/modeling_pi0.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/pi0/modeling_pi0.py>`_.
- This model can be changed to pi0-star by changing the `advantage` flag in the config file.

value
-----
- Value model is a vision-language model used to predict the value of the current state. Its used to train VLA policies with RECAP framework.
- More details can be found in the `pi*06 paper <https://www.pi.website/download/pistar06.pdf>`_.
- See the implementation in `src/opentau/policies/value/modeling_value.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/policies/value/modeling_value.py>`_.
