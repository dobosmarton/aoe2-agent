# `data/` — AoE2 Game Knowledge

A self-contained SQLite database of AoE2 game knowledge — buildings, units,
techs, civs, counter relationships — plus the loaders that populated it.
Consumed by `gameplay-agent.providers.claude` for dynamic context injection
(the executor can be asked "what's the cost of a Castle?" and look it up
without re-prompting the LLM).

## What's here

```
packages/data/src/
├── aoe2.db                # SQLite database (ships inside the wheel)
├── game_knowledge.py      # GameKnowledge accessor class + get_db() singleton
├── fetch_aoe2_data.py     # Populates the DB from public AoE2 data sources
├── populate_db.py         # `python -m data.populate_db` to (re)build aoe2.db
├── _halfon_schema.py      # Pydantic schema for the halfon-format data dump
├── _narrow.py             # Type-narrowing helpers (as_int, as_str)
└── knowledge_base/        # *.json + summary.md — source documents the DB is built from
```

## Common usage

```python
from data import GameKnowledge

kb = GameKnowledge()
castle = kb.get_building("castle")
print(castle.cost)  # {"food": 0, "wood": 0, "gold": 0, "stone": 650}
```

The DB is read-only at runtime. Regenerate with:

```bash
uv run --package data python -m data.populate_db
```

## Where to read more

- [Chapter 10 — Knowledge Database](../../docs/part4-game-knowledge/10-knowledge-database.md) — schema, query patterns, dynamic context injection.
- [Chapter 6 — Context Injection](../../docs/part2-llm-integration/06-context-injection.md) — how the agent threads knowledge-base lookups into LLM context.
