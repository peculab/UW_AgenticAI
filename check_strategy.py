#!/usr/bin/env python3
import json
from pathlib import Path

context_file = Path("results/agentic_context.json")
if context_file.exists():
    data = json.loads(context_file.read_text())
    strategy = data.get("latest_dqn_strategy", {})
    print("=" * 60)
    print("DQN STRATEGY WITH PATTERN RECOMMENDATIONS")
    print("=" * 60)
    print(json.dumps(strategy, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60)
    print("PATTERN-SPECIFIC RECOMMENDATION:")
    print("=" * 60)
    if "pattern_info" in strategy:
        print(json.dumps(strategy["pattern_info"], indent=2, ensure_ascii=False))
    else:
        print("No pattern-specific recommendation in output")
else:
    print(f"File not found: {context_file}")
