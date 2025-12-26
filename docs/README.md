# LeRobot Documentation

This directory contains the Sphinx documentation for the LeRobot project.

## Installation

To build the documentation, you need to install the project with the `doc` extra dependencies. We recommend using `uv` for dependency management.

```bash
# Sync dependencies including documentation tools
uv sync --extra doc
```

## Building the Documentation

You can build the HTML documentation using the provided Makefile or directly with `sphinx-build`.

### Using Makefile (Linux/macOS)

From the `docs` directory:

```bash
source ../.venv/bin/activate
make html
```

### Using sphinx-build (Universal)

From the `docs` directory:

```bash
source ../.venv/bin/activate
sphinx-build -b html source _build/html
```

## Viewing the Documentation

After building, the HTML files will be located in `docs/_build/html`. You can open `docs/_build/html/index.html` in your web browser to view the documentation.

On Linux, you can use `xdg-open`:

```bash
xdg-open docs/_build/html/index.html
```

## Writing Documentation

The documentation uses **reStructuredText (.rst)** and **Markdown (.md)** (via MyST-Parser).

-   **reStructuredText**: The standard for Sphinx. Good for complex directives and autodoc.
-   **Markdown**: Supported via MyST. Useful for narrative documentation.

To update the API documentation, edit the docstrings in the Python code. The `index.rst` file is configured to automatically document the `lerobot` package.
