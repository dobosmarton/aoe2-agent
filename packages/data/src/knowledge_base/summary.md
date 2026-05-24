# AoE2 Strategic Reference

Quick reference for the LLM agent's strategic decisions.

## Unit Counter Matrix

| Unit Type | Strong Against | Weak Against |
|-----------|---------------|--------------|
| archer | infantry, spearman | skirmisher, mangonel, cavalry, eagle_warrior |
| cavalry | archer, siege, monk, skirmisher | spearman, camel, monk_large_groups |
| infantry | building, eagle_warrior, trash_unit | archer, hand_cannoneer, cavalry |
| spearman | cavalry, camel, elephant | archer, infantry, mangonel |
| skirmisher | archer, spearman | cavalry, infantry, mangonel |
| knight | archer, siege, villager, monk | pikeman, camel, halberdier |
| siege | infantry, archer, building | cavalry, bombard_cannon |
| monk | knight, elephant, expensive_units | light_cavalry, eagle_warrior, archer_mass |
| camel | cavalry, knight | archer, infantry, monk |
| eagle_warrior | monk, siege, archer | infantry, cavalry |

## Key Strategic Principles

1. **Rock-Paper-Scissors**: Archers > Infantry > Cavalry > Archers
2. **Economy wins games**: More villagers = more resources = bigger army
3. **Never stop producing villagers** until 120+ (or you're all-in)
4. **Scout early**: Know what your opponent is making, then counter it
5. **Composition matters**: Mix unit types to cover weaknesses
6. **Upgrades > Numbers**: Upgraded units beat larger un-upgraded armies

## Common Build Orders

### Fast Castle (Knights)
- 6 sheep, 4 wood, 1 boar, 3 berries, 1 boar, 2 farms
- Click Feudal at ~22 pop, build 2 stables, click Castle
- Make Knights immediately

### Scouts into Knights
- 6 sheep, 3 wood, 1 boar, 1 berries, 1 boar, 3 farms
- Click Feudal at ~21 pop, build stable + blacksmith
- Make Scouts, transition to Knights in Castle Age

### Archers (Feudal Rush)
- 6 sheep, 4 wood, 1 boar, 3 gold, 1 boar, 2 farms
- Click Feudal at ~22 pop, build 2 archery ranges
- Mass archers, add skirmishers if they counter with archers

## Age Advancement Costs

| Age | Cost | Time |
|-----|------|------|
| Feudal | 500F | 130s |
| Castle | 800F + 200G | 160s |
| Imperial | 1000F + 800G | 190s |

## Statistics

- Total units in database: 1102
- Total buildings in database: 670
- Total civilizations: 50
