"""
Test Configuration Module

This module sets up the testing environment by:
    - Configuring Python path for test discovery
    - Setting up test fixtures
    - Providing common test utilities
    - Configuring warnings
"""

import os
import sys
import warnings
from pydantic import PydanticDeprecatedSince20

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Filter out Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20, module="pydantic")
