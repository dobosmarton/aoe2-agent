"""Build the LLM context string from a scenario fixture.

Mirrors `gameplay_agent.game_loop._build_llm_context` so a fixture's `inputs:`
block produces the same shape of prompt the agent sees in production. Each
helper renders one section (entities → goals → resources → state → recent
turns) and `_build_context` stitches them together in the production order.

These helpers also populate the executor's detected-entities side channel via
`_entity_dict` — the shape must match what `gameplay_agent.executor` expects.
"""

from __future__ import annotations

ENTITY_BBOX_HALF_SIZE = 20
DEFAULT_ENTITY_CONFIDENCE = 0.9
PRIORITY_HIGH_THRESHOLD = 8
PRIORITY_MED_THRESHOLD = 5
RECENT_TURN_REASONING_PREVIEW = 100
DEFAULT_AGE = "Dark Age"


def _entity_dict(entity: dict, index: int) -> dict:
    """Convert a fixture entity into the executor's internal dict shape."""
    class_name = entity.get("class", "unknown")
    x = entity.get("x", 0)
    y = entity.get("y", 0)
    return {
        "id": entity.get("id", f"{class_name}_{index}"),
        "class": class_name,
        "center": (x, y),
        "bbox": (
            x - ENTITY_BBOX_HALF_SIZE,
            y - ENTITY_BBOX_HALF_SIZE,
            x + ENTITY_BBOX_HALF_SIZE,
            y + ENTITY_BBOX_HALF_SIZE,
        ),
        "confidence": entity.get("confidence", DEFAULT_ENTITY_CONFIDENCE),
    }


def _format_entity_line(entity: dict, index: int) -> str:
    class_name = entity.get("class", "unknown")
    x = int(entity.get("x", 0))
    y = int(entity.get("y", 0))
    confidence = float(entity.get("confidence", DEFAULT_ENTITY_CONFIDENCE))
    entity_id = entity.get("id", f"{class_name}_{index}")
    return f"  {entity_id}: {class_name} at ({x},{y}) [{confidence:.0%}]"


def _build_entity_summary(entities: list[dict]) -> str:
    """Mirror gameplay_agent.entity_utils.build_entity_summary's output shape."""
    if not entities:
        return ""
    return "\n".join(_format_entity_line(entity, index) for index, entity in enumerate(entities))


def _priority_tier(priority: int) -> str:
    if priority >= PRIORITY_HIGH_THRESHOLD:
        return "HIGH"
    if priority >= PRIORITY_MED_THRESHOLD:
        return "MED"
    return "LOW"


def _build_resource_block(resources: dict, age: str) -> str:
    return "\n".join(
        [
            "## Resource Status (from strategist)",
            f"- Food: {resources.get('food', '?')}",
            f"- Wood: {resources.get('wood', '?')}",
            f"- Gold: {resources.get('gold', '?')}",
            f"- Stone: {resources.get('stone', '?')}",
            f"- Population: {resources.get('population', '0/0')}",
            f"- Age: {age}",
        ]
    )


def _build_goal_block(goals: list[dict]) -> str:
    if not goals:
        return ""
    lines = ["## Active Goals"]
    for goal in goals:
        priority = goal.get("priority", PRIORITY_MED_THRESHOLD)
        tier = _priority_tier(priority)
        lines.append(
            f"  {tier} (P{priority}): {goal.get('name', '?')} → "
            f"{goal.get('metric')} target {goal.get('target')}"
        )
    return "\n".join(lines)


def _build_state_block(resources: dict, age: str, under_attack: bool) -> str:
    population = resources.get("population", "0/0")
    pop_now, _, pop_cap = population.partition("/")
    non_pop_resources = " ".join(
        f"{key}={value}" for key, value in resources.items() if key != "population"
    )
    lines = [
        "## Current Game State",
        f"- Resources: {non_pop_resources}",
        f"- Population: {pop_now or 0}/{pop_cap or 0}",
        f"- Age: {age}",
    ]
    if under_attack:
        lines.append("- under_attack: true")
    return "\n".join(lines)


def _build_entity_block(entities: list[dict]) -> str:
    summary = _build_entity_summary(entities)
    if not summary:
        return ""
    return (
        "\n## Detected Entities (from YOLO)\n"
        "Use target_class or target_id to interact with these:\n"
        f"{summary}\n"
    )


def _build_recent_turns_block(recent_turns: list[dict]) -> str:
    if not recent_turns:
        return ""
    lines = ["## Recent Turns (last 3)"]
    for turn in recent_turns:
        iteration = turn.get("iteration", "?")
        reasoning_preview = turn.get("reasoning", "")[:RECENT_TURN_REASONING_PREVIEW]
        lines.append(f"Turn {iteration}: {reasoning_preview}")
    return "\n".join(lines)


def _apply_strategist_overrides(inputs: dict) -> dict:
    """Merge `strategist_overrides:` on top of base inputs.

    The strategist normally provides resource readings and goals to the
    executor. This helper lets a fixture express "what if the strategist
    output something different?" without running the real strategist.

    Merge semantics:
      resources  — shallow merge (override individual fields, preserve others)
      goals      — replace entirely (lists have no canonical partial-merge)

    Returns a new dict; the original is never mutated.
    """
    overrides = inputs.get("strategist_overrides") or {}
    if not overrides:
        return inputs

    merged = {**inputs}
    if "resources" in overrides:
        merged["resources"] = {
            **inputs.get("resources", {}),
            **overrides["resources"],
        }
    if "goals" in overrides:
        merged["goals"] = overrides["goals"]
    return merged


def _build_context(fixture: dict) -> str:
    """Assemble the context string the same way game_loop._build_llm_context does.

    Order matches the production assembly: entities → goals → resources → state → recent.
    """
    inputs = fixture.get("inputs", {})
    inputs = _apply_strategist_overrides(inputs)
    resources = inputs.get("resources", {})
    age = inputs.get("age", DEFAULT_AGE)

    blocks = [
        _build_entity_block(inputs.get("detected_entities", [])),
        _build_goal_block(inputs.get("goals", [])),
        _build_resource_block(resources, age),
        _build_state_block(resources, age, bool(inputs.get("under_attack"))),
        _build_recent_turns_block(inputs.get("recent_turns", [])),
    ]
    return "\n\n".join(block for block in blocks if block)
