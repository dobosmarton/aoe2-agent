## Hotkey Reference

### Navigation & Selection
- NEVER press Escape: with nothing to cancel it OPENS the game menu and pauses
  the game. To clear an open build menu or placement ghost, press H (select
  TC) instead — switching selection cancels them.
- Coordinates go stale when the camera moves: after pressing `.`, `,` or H,
  never click x/y computed from the previous frame — name a target_class or
  target_id instead (the executor resolves it against the fresh view).
- H: Go to Town Center
- .: Select next idle villager (moves camera)
- ,: Select next idle military unit (moves camera)
- Space: Go to selected object
- Home: Go to last notification
- Shift-.: Select ALL idle villagers
- Alt-,: Select ALL idle military units
- Ctrl-B: Go to Barracks
- Ctrl-A: Go to Archery Range
- Ctrl-L: Go to Stable
- Ctrl-V: Go to Castle
- Ctrl-Z: Go to Lumber Camp
- Ctrl-I: Go to Mill
- Ctrl-G: Go to Mining Camp
- Ctrl-M: Go to Market
- Ctrl-Y: Go to Monastery
- Ctrl-S: Go to Blacksmith

### Town Center (after pressing H)
- Q: Queue Villager (50 food)
- B: Ring Town Bell (garrison all nearby villagers)
- Z: Research Age Up
- A: Research Loom
- V: All Back to Work (ungarrison)
- F: Go Back to Work

A build menu lists only the buildings available right now, and the rest of the
grid shifts up to fill the gap — so a slot pressed before its building exists
lands on a DIFFERENT building. The Farm/Outpost note below is that hazard, and
it is why age-gated entries (Blacksmith, Market) are refused until the HUD
confirms the age.

Use the `build` action rather than pressing these yourself: it opens the right
menu, presses the slot, places on open ground and leaves the UI clean.

### Villager Build — Economic (select villager, press Q) — VERIFIED
- Q: House (25 wood)
- W: Mill (100 wood)
- E: Mining Camp (100 wood)
- R: Lumber Camp (100 wood)
- A: Farm (60 wood) — ONLY when a Mill exists; without a Mill this slot is the OUTPOST (pressing A builds a tower!)
- T: Dock (150 wood)
- S: Blacksmith (150 wood) — Feudal Age and later only

### Villager Build — Military (select villager, press W) — UNVERIFIED
- Q: Barracks (175 wood) — a Dark Age building; does NOT count toward the Castle Age
- W: Archery Range (175 wood) — needs a Barracks, so its slot moves with one
- E: Stable (175 wood) — needs a Barracks, so its slot moves with one
- R: Siege Workshop (200 wood)
- F: Monastery (175 wood)

### Villager Build — More Buildings (select villager, press V) — UNVERIFIED
- D: Market (175 wood) — Feudal Age and later only
- F: Tower (125 wood, 25 stone)
- S: Palisade Wall
- Z: Town Center (275 wood, 100 stone)
- G: University (200 wood)
- C: Castle (650 stone)

Only the economic menu's layout has been confirmed in-game. The other two stay
refused until someone runs the VM check and sets `AOE2_VERIFIED_BUILD_MENUS`
(see the runbook) — an unverified slot doesn't no-op, it builds whatever
occupies that position.

### Age advancement (select TC with H, then Z)
- Feudal Age: 500 food + TWO Dark Age buildings (houses don't count)
- Castle Age: 800 food + 200 gold + TWO Feudal Age buildings (Blacksmith,
  Market, Archery Range, Stable — Dark Age buildings don't count)

### Unit Commands (when unit is selected)
- A: Drop Off Resources
- G: Stop
- T: Garrison into building
- R: Repair (villagers only)

### Barracks Units (after Ctrl-B)
- Q: Militia-line
- W: Spearman-line
- R: Eagle Warrior

### Archery Range Units (after Ctrl-A)
- Q: Archer-line
- W: Skirmisher
- E: Cavalry Archer
- R: Hand Cannoneer

### Stable Units (after Ctrl-L)
- Q: Scout Cavalry / Hussar
- W: Knight-line
- E: Camel Rider
- R: Battle Elephant

### Siege Workshop Units
- Q: Battering Ram
- W: Mangonel / Onager
- E: Scorpion
- R: Bombard Cannon / Trebuchet
- V: Siege Tower

### Castle (after Ctrl-V)
- Q: Unique Unit
- W: Trebuchet
- E: Petard

### Monastery (after Ctrl-Y)
- Q: Monk

### Market (after Ctrl-M)
- Q: Trade Cart
- C: Buy 100 Food
- X: Buy 100 Wood
- V: Buy 100 Stone
- D: Sell 100 Food
- S: Sell 100 Wood
- F: Sell 100 Stone

### Mill (after Ctrl-I)
- R: Reseed Farm
- T: Toggle Auto Farm Reseeding

### Military Stances (when military unit selected)
- G: Auto Scout (scout explores map automatically)
- R: Attack Move
- Q: Patrol
- D: Stand Ground
- S: Defensive stance
- A: Aggressive stance
