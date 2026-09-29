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

"""FLUX 3 Action (Black Forest Labs) ported as an OpenTau policy.

Every module besides ``configuration_flux3_action`` / ``modeling_flux3_action`` is
vendored from ``black-forest-labs/flux-action`` and kept byte-identical to upstream
apart from a three-line import rewrite -- see ``VENDOR.md`` for the provenance commit
and the exact delta, which is what keeps an upstream re-sync a mechanical diff.
"""
