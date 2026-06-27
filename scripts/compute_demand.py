#!/usr/bin/env python3
"""
Standalone demand pre-computation script.
Usage: python compute_demand.py --community alquaa
"""
import argparse
import json
import os
import sys

CONFIG_DIR = "config"


def main():
    parser = argparse.ArgumentParser(description="Compute Basira demand scores for a community.")
    parser.add_argument("--community", required=True, help="Community ID (e.g. alquaa)")
    args = parser.parse_args()

    config_path = os.path.join(CONFIG_DIR, f"{args.community}.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        community = json.load(f)

    print(f"[compute_demand] Community: {community['community_name_en']} ({args.community})")
    print(f"[compute_demand] Bounding box: {community['bbox']}")
    print()

    from demand_engine import compute_all_demand_scores
    results = compute_all_demand_scores(community)

    print("\n========== RESULTS ==========")
    for cat_id, data in sorted(results.items()):
        print(f"  {cat_id}: score={data['demand_score']}/100")
        for sig, val in data["signals"].items():
            print(f"         {sig}: {val:.3f}")
    print("==============================")
    print(f"\n[compute_demand] Done. Results saved to outputs/demand_scores_{args.community}.json")


if __name__ == "__main__":
    main()