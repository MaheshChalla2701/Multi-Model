"""
conftest.py
-----------
Shared pytest fixtures for the backend test suite.
"""

import os

# Ensure the env is populated before any app module is imported.
# These dummy values prevent ValueError on missing keys during import.
os.environ.setdefault("GEMINI_API_KEY", "test-api-key")
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("NEO4J_URI", "")
os.environ.setdefault("NEO4J_USERNAME", "")
os.environ.setdefault("NEO4J_PASSWORD", "")
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
