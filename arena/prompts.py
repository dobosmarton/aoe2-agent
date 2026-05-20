"""Prompt variants for the synth arena (Phase 6 extension).

Each profile in `arena/profiles/*.yaml` names a variant via its
`prompt_variant` field. The race harness looks up the system prompt
here at run time. Adding a new variant is a 1-key addition to `PROMPTS`.
"""

from __future__ import annotations

_ACTION_LIST = """\
Available actions:
  {"type": "queue_villager"}               — train villager (50 food; pop < pop_cap)
  {"type": "build", "building_key": "q"}   — house      (25 wood; +5 pop_cap)
  {"type": "build", "building_key": "w"}   — mill       (100 wood)
  {"type": "build", "building_key": "r"}   — lumber camp (100 wood)
  {"type": "build", "building_key": "e"}   — mining camp (100 wood)
  {"type": "build", "building_key": "a"}   — farm        (60 wood)
  {"type": "build", "building_key": "s"}   — blacksmith  (150 wood)
  {"type": "press", "key": "z"}            — start age-up (500 food; needs mill + lumber_camp + pop≥22)

Resources tick automatically each turn: +20 food, +15 wood."""

_STRATEGY_NOTES = """\
Goal: reach Feudal Age efficiently. Strategy notes:
  - Age-up has FOUR prerequisites: mill built, lumber_camp built, pop≥22, AND food≥500.
    The 'press z' action silently no-ops if ANY prerequisite is missing — verify all four
    are met before pressing.
  - When food < 500 but other prereqs are met, STOP queueing villagers (each costs 50 food
    and delays the age-up). Wait for food to tick up to 500, then press z. Spending food on
    villagers in this window directly delays the age-up by 2-3 turns per villager.
  - Build mill + lumber_camp early. Houses only when pop_cap is the bottleneck."""

_RESPONSE_FORMAT = (
    "Each turn you receive the current game state. Respond with ONLY a valid JSON "
    'array of actions (e.g. [{"type": "queue_villager"}]) or [] to do nothing. '
    "No markdown, no explanation — just the JSON array."
)

_INTRO = "You are an Age of Empires 2 economy strategist."


PROMPTS: dict[str, str] = {
    "bare": f"""\
{_INTRO}

{_RESPONSE_FORMAT}

{_ACTION_LIST}

Goal: grow economy and reach Feudal Age efficiently.""",
    "strategy": f"""\
{_INTRO}

{_RESPONSE_FORMAT}

{_ACTION_LIST}

{_STRATEGY_NOTES}""",
}


def get_prompt(variant: str) -> str:
    """Look up a prompt by variant name. Raises KeyError on unknown variant."""
    return PROMPTS[variant]
