#!/usr/bin/env python

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

"""Pre-commit hook to add license headers to Python files.

This script checks if edited Python files have the required license header.
If not present, it automatically adds the Apache License 2.0 header with
copyright notices.
"""

import re
import sys
from pathlib import Path

# License header templates
LICENSE_HEADER_HUGGINGFACE = """# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License."""

LICENSE_HEADER_EXISTING = """# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License."""

LICENSE_HEADER_NEW = """# Copyright 2026 Tensor Auto Inc. All rights reserved.
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
# limitations under the License."""

# Pattern to detect existing license header (with both copyrights)
LICENSE_PATTERN_EXISTING = re.compile(
    r"#\s*Copyright.*HuggingFace.*?\n#\s*Copyright.*Tensor Auto.*?\n#\s*\n#\s*Licensed under the Apache License",
    re.DOTALL | re.IGNORECASE,
)

# Pattern to detect new license header (Tensor Auto only)
LICENSE_PATTERN_NEW = re.compile(
    r"#\s*Copyright.*Tensor Auto.*?\n#\s*\n#\s*Licensed under the Apache License",
    re.DOTALL | re.IGNORECASE,
)

# Pattern to detect HuggingFace-only license header
LICENSE_PATTERN_HUGGINGFACE = re.compile(
    r"#\s*Copyright.*HuggingFace.*?\n#\s*\n#\s*Licensed under the Apache License",
    re.DOTALL | re.IGNORECASE,
)


def has_existing_license_header(content: str) -> bool:
    """Check if file content has the existing license header (both copyrights)."""
    # Check first 20 lines for license header
    first_lines = "\n".join(content.split("\n")[:20])
    return bool(LICENSE_PATTERN_EXISTING.search(first_lines))


def has_new_license_header(content: str) -> bool:
    """Check if file content has the new license header (Tensor Auto only)."""
    # Check first 20 lines for license header
    first_lines = "\n".join(content.split("\n")[:20])
    return bool(LICENSE_PATTERN_NEW.search(first_lines))


def has_huggingface_license_header(content: str) -> bool:
    """Check if file content has the HuggingFace license header."""
    # Check first 20 lines for license header
    first_lines = "\n".join(content.split("\n")[:20])
    return bool(LICENSE_PATTERN_HUGGINGFACE.search(first_lines))


def add_license_header(file_path: Path) -> bool:
    """Add license header to file if missing. Returns True if file was modified."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    # Skip if existing license header (both copyrights) already present
    if has_existing_license_header(content):
        return False

    # Skip if new license header (Tensor Auto only) already present
    if has_new_license_header(content):
        return False

    # Determine which license header to add
    if has_huggingface_license_header(content):
        # File has HuggingFace license, add LICENSE_HEADER_EXISTING
        license_header = LICENSE_HEADER_EXISTING
        # Need to replace the existing HuggingFace header
        needs_replacement = True
    else:
        # File has no license or different license, add LICENSE_HEADER_NEW
        license_header = LICENSE_HEADER_NEW
        needs_replacement = False

    # Handle shebang line
    lines = content.split("\n")
    shebang = None
    start_idx = 0

    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        start_idx = 1
        # Skip empty line after shebang if present
        if len(lines) > 1 and not lines[1].strip():
            start_idx = 2

    # If replacing HuggingFace header, find and remove it
    if needs_replacement:
        # Find where the license header starts and ends
        license_start_idx = None
        license_end_idx = None

        for i in range(start_idx, min(start_idx + 20, len(lines))):
            line = lines[i]
            # Find the start of the license (first Copyright line with HuggingFace)
            if license_start_idx is None and "Copyright" in line and "HuggingFace" in line:
                license_start_idx = i
            # Find the end of the license (last line with "limitations under the License")
            elif license_start_idx is not None and "limitations under the License" in line:
                license_end_idx = i + 1
                # Skip empty line after license if present
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    license_end_idx = i + 2
                break

        # Remove the old license header
        if license_start_idx is not None and license_end_idx is not None:
            lines = lines[:license_start_idx] + lines[license_end_idx:]

    # Build new content
    new_lines = []
    if shebang:
        new_lines.append(shebang)
        new_lines.append("")  # Empty line after shebang

    # Add license header
    new_lines.append(license_header)
    new_lines.append("")  # Empty line after license

    # Add rest of content
    new_lines.extend(lines[start_idx:])

    # Write back
    new_content = "\n".join(new_lines)
    # Preserve original trailing newline
    if content.endswith("\n"):
        new_content += "\n"
    file_path.write_text(new_content, encoding="utf-8")
    return True


def main():
    """Process files passed as arguments."""
    if len(sys.argv) < 2:
        print("Usage: add_license_header.py <file1> [file2] ...", file=sys.stderr)
        sys.exit(1)

    modified_files = []
    for file_path_str in sys.argv[1:]:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping", file=sys.stderr)
            continue

        # Only process Python files
        if file_path.suffix != ".py":
            continue

        if add_license_header(file_path):
            modified_files.append(file_path)
            print(f"Added license header to {file_path}")

    if modified_files:
        print(f"\nModified {len(modified_files)} file(s)")
        sys.exit(1)  # Exit with error to indicate files were modified
    else:
        sys.exit(0)  # All files already have license headers


if __name__ == "__main__":
    main()
