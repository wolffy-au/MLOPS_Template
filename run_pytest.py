# 7. Automated Testing Script (test.py):
# Contains unit tests, integration tests, or end-to-end tests for MLOps components.
# Ensures that each part of the pipeline functions correctly and prevents regressions.
# Triggered as part of the CI/CD process to validate changes.

import os
import pytest

def run_pytest():
    # os.chdir(".")
    
    # Run Pytest
    pytest.main(["-v", "tests"])

if __name__ == "__main__":
    run_pytest()
