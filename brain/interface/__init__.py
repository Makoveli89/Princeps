"""
Brain Interface Module
======================

CLI and API endpoints.

Exports:
    - BrainCLI: Command-line interface
    - BrainAPI: FastAPI application
    - Test scenarios
"""

from .brain_api import (
    BrainAPI,
    create_app,
)
from .brain_cli import (
    BrainCLI,
    cli_main,
)
from .test_scenarios import (
    TestScenario,
    run_integration_tests,
)

__all__ = [
    "BrainCLI",
    "cli_main",
    "create_app",
    "BrainAPI",
    "run_integration_tests",
    "TestScenario",
]
