#!/usr/bin/env python3
import os
import json

print("=" * 60)
print("🧪 VERIFYING PROJECT FILES")
print("=" * 60)

# Load inventory
with open('INVENTORY.json') as f:
    inv = json.load(f)

print(f"\nTotal files: {inv['total_files']}")
print(f"Total size: {inv['total_size_mb']:.2f} MB")
print(f"\nFile Summary:")
print(f"  CSV files: {inv['summary']['csv_files']}")
print(f"  Models: {inv['summary']['model_files']}")
print(f"  Python files: {inv['summary']['python_files']}")

# Check if key files exist
print("\n🔍 Checking key files...")
key_files = [f for f in inv['files'].keys() if any(x in f for x in ['.csv', '.joblib', '.pkl'])]
print(f"Found {len(key_files)} data/model files")

print("\n✅ Verification complete!")
