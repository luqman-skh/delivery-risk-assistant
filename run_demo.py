"""
Interactive / Demonstration CLI for Phase 1 Delivery Risk Assistant.
Usage:
    python run_demo.py --order-id <ORDER_ID>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.agents.coordinator import DeliveryRiskCoordinator


def main():
    parser = argparse.ArgumentParser(description="Delivery Risk Assistant CLI")
    parser.add_argument("--order-id", type=int, default=77202, help="Order ID to evaluate")
    parser.add_argument("--policy", type=str, default="v1.0", help="Policy version")
    args = parser.parse_args()

    print(f"--- Evaluating Order ID: {args.order_id} under Policy {args.policy} ---")
    coordinator = DeliveryRiskCoordinator(policy_version=args.policy)
    
    assessment = coordinator.evaluate_order(order_id=args.order_id)
    print("\n=== AGENT DECISION PAYLOAD ===")
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
