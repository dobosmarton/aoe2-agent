"""Typed schema for `classes.yaml` (the canonical detection class list).

The training schema can grow before a matching detector is trained.
The inference server bundles the class mapping for its deployed checkpoint.
Both YAML files use the same shape:

```yaml
classes:
  - { id: 0, name: tree }
  - { id: 1, name: gold_mine }
  ...
```

Modelling that here as a `TypedDict` means key lookups are statically
checked: a misspelled `"clases"` or `"id_"` fails at type-check time
rather than at the next test run.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ClassEntry(TypedDict):
    id: int
    name: str
    # Civ-specific unit names (e.g. unique_cavalry -> Cataphract, Boyar, ...),
    # used to build open-vocabulary detection prompts.
    examples: NotRequired[list[str]]


class ClassesYaml(TypedDict):
    classes: list[ClassEntry]
