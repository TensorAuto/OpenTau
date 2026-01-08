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

"""Module for patching transformers metadata lookup.

This module monkey patches `importlib.metadata.distribution` to redirect
'transformers' lookups to 'opentau-transformers'. This ensures that the custom
transformers fork is correctly recognized by libraries checking for transformers
installation.
"""

import importlib.metadata

# Keep a reference to the original distribution function
_orig_distribution = importlib.metadata.distribution


def distribution(distribution_name):
    """Monkey patch to redirect 'transformers' metadata lookups to 'opentau-transformers'.

    This function intercepts calls to `importlib.metadata.distribution`. If the
    requested distribution is "transformers", it redirects the lookup to
    "opentau-transformers". Otherwise, it delegates to the original implementation.

    Args:
        distribution_name: The name of the distribution to retrieve metadata for.

    Returns:
        The distribution metadata object.
    """
    if distribution_name == "transformers":
        distribution_name = "opentau-transformers"
    return _orig_distribution(distribution_name)


# Apply the patch
importlib.metadata.distribution = distribution
