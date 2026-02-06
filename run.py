#!/usr/bin/env python
"""Simple script to run a research query from the command line."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import run_research


def main():
    parser = argparse.ArgumentParser(description="Run a research query")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the latest developments in large language models?",
        help="The research query",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"\n{'='*80}")
    print(f"Researching: {args.query}")
    print(f"{'='*80}\n")

    result = run_research(args.query)

    print(f"\n{'='*80}")
    print("RESEARCH REPORT")
    print(f"{'='*80}\n")
    print(result.get("final_report", "No report generated"))

    print(f"\n{'='*80}")
    print("METADATA")
    print(f"{'='*80}")
    print(f"Query Type: {result.get('query_type')}")
    print(f"Complexity: {result.get('query_complexity')}")
    print(f"Sources Retrieved: {len(result.get('sources', []))}")
    print(f"Facts Extracted: {len(result.get('extracted_facts', []))}")
    print(f"Contradictions Found: {len(result.get('contradictions', []))}")

    if result.get("errors"):
        print(f"\nErrors: {result['errors']}")


if __name__ == "__main__":
    main()
