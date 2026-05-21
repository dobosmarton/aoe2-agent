"""Typed schema for `classes.yaml` (the canonical detection class list).

A single `classes.yaml` is bundled both with the inference server
(`server/classes.yaml`) and the training config
(`detection/training/config/classes.yaml`). All three loaders
(server, detector, class_mapping) parse the same shape:

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

from typing import TypedDict


class ClassEntry(TypedDict):
    id: int
    name: str


class ClassesYaml(TypedDict):
    classes: list[ClassEntry]
