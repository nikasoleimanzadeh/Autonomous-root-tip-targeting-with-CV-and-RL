# Shared OT-2 Source Package

The [`ot2/`](ot2/) Python package holds reusable simulator helpers, including workspace-envelope logic, control utilities, and environment adapters. Folder-level demonstration scripts import from this package rather than duplicating the same lower-level logic.

The existing `pyproject.toml` builds the `ot2` package. Before claiming a reproducible installation, verify the external simulator dependency and the import paths used by individual scripts. A future improvement would be a small package test suite and one documented `uv sync` workflow.
