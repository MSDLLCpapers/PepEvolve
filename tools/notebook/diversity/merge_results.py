#!/usr/bin/env python3
"""Merge per-mode output pickles into combined dfs_diversity.pkl"""
import os
import pickle as pkl

output_dir = "output"
labels = [
    "crbp_neighbor_multi",
    "crbp_neighbor_single",
    "crbp_self_multi",
    "crbp_self_single",
    "crbp_pepinvent",
    "traditional_ga",
]

dfs_div = {}
for label in labels:
    path = os.path.join(output_dir, f"{label}_diversity.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            dfs_div[label] = pkl.load(f)
        print(f"  {label}: {len(dfs_div[label])} steps")
    else:
        print(f"  MISSING: {path}")

with open("dfs_diversity.pkl", "wb") as f:
    pkl.dump(dfs_div, f)

print(f"Merged {len(dfs_div)} modes into dfs_diversity.pkl")
