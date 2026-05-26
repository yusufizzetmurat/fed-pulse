"""Repository-level scripts as an importable package.

Empty marker so tests can ``from scripts.<module> import …`` without
falling back to the ``sys.path.insert(_SCRIPTS_DIR)`` trick used by
``tests/unit/test_garch_baseline.py``.
"""
