import asyncio
import sys
from pathlib import Path
import logging

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_graph import run_workflow_graph

# Reduce log verbosity to avoid console encoding issues with unicode symbols
logging.getLogger().setLevel(logging.ERROR)

QUERIES = [
    "Finance analyst for budgeting and forecasting, under 45 minutes",
    "Hiring a data analyst with SQL, Python and Tableau skills",
    "Presales engineer for cloud solutions and demos",
    "Research analyst for market research and reporting",
    "Business analyst needing process mapping and stakeholder management",
    "Security analyst with vulnerability management focus",
    "Java developer who collaborates with teams",
    "Marketing manager with campaign analytics focus",
    "Customer service representative role",
    "Operations manager with supply chain exposure",
]


async def main() -> None:
    for i, q in enumerate(QUERIES, 1):
        try:
            results = await run_workflow_graph(q)
            print(f"\n=== {i}. {q} ===")
            for j, r in enumerate(results[:3], 1):
                name = r.get("name", "")
                dur = r.get("duration", r.get("duration_minutes", ""))
                t = r.get("test_types", "")
                print(f"{j}) {name} | dur={dur} | types={t}")
        except Exception as e:
            print(f"Error for query: {q}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

