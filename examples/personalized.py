"""Run with: uv run python examples/personalized.py"""

import asyncio
from pathlib import Path

from personal_harness import Agent


async def main() -> None:
    config = Path("personal-harness.toml")
    for profile in ("desktop", "mobile"):
        async with Agent.from_config(config, profile=profile) as agent:
            result = await agent.run("Help me plan today")
            print(result.output)
            print(f"Cost: {result.currency} {result.cost}\n")


if __name__ == "__main__":
    asyncio.run(main())
