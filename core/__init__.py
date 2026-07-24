"""Portable Aletheia core: config, models, storage contract, memory logic.

Hard rule (the project plan §7): this package must never import boto3 or psycopg.
Infrastructure enters through the StorageAdapter protocol or injected callbacks.
Enforced by tests/test_architecture.py.
"""
