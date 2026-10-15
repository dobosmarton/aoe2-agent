# Historical agent knowledge snapshot

This directory preserves an August 2026 snapshot of run findings, prompt rules, memory notes, and baseline scores. It is **not** loaded as a runtime policy.

The retired YAML rule registry has been removed. Current feasibility and fallback decisions live in `apps/agent/src/policy/` and share the action catalog used by execution. The remaining `findings.md`, `prompt_rules.md`, `archive/`, and `baseline.tsv` are historical research artifacts; their rule names and thresholds may no longer describe live behavior.
