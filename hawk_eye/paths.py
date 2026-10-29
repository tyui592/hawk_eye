"""Locate repository-owned data independently of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
