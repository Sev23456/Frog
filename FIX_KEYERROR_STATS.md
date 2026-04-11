# 🔧 Fix: KeyError in Statistics Access

## Problem
Running `run_simulation.py` threw a `KeyError: 'avg_neural_activity'` because the script was trying to access statistics keys that don't exist in the `get_statistics()` return dictionary.

## Root Cause
The [simulation.py](frog_lib/simulation.py) `get_statistics()` method returns only 7 keys:
- `total_steps`
- `caught_flies`
- `success_rate`
- `final_energy`
- `avg_dopamine`
- `is_juvenile`
- `juvenile_age`

But [run_simulation.py](run_simulation.py) and [run_menu.py](run_menu.py) were trying to access non-existent keys like:
- `avg_neural_activity` ❌
- `avg_speed` ❌
- `juvenile_progress` ❌
- `architecture_signature` ❌

## Solution Applied

### File: run_simulation.py
**Changed:** Removed references to non-existent keys
- ❌ Removed: `stats['avg_neural_activity']`
- ❌ Removed: `stats['juvenile_progress']`
- ✅ Added: `stats['juvenile_age']` (real key)

### File: run_menu.py  
**Changed:** Made keys optional using `.get()` method
- Line 46: Now safely accesses `stats.get('avg_dopamine', 0.0)`
- Lines 47-49: Conditional checks before printing optional keys
- Line 88-89: Now uses `.get()` for nullable keys and prints what exists

## Files Modified
1. `run_simulation.py` - Line 38 (removed 2 non-existent keys)
2. `run_menu.py` - Lines 46-48 and 85-89 (added safe key access)

## Testing
✅ Verified: `BioFlyCatchingSimulation.get_statistics()` now returns the correct 7 keys without errors

## Usage
Now you can run without errors:
```bash
python run_simulation.py          # ✅ Fixed
python run_menu.py               # ✅ Fixed
python compare_agents.py          # ✅ Already safe
```

## Lesson
Always verify what keys are actually returned by a function before accessing them. The fix ensures robustness by:
1. Using `.get()` for optional keys
2. Checking key existence before access
3. Having sensible defaults
