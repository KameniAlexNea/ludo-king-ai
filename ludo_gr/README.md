# Ludo Gradio Visualizer (`ludo_gr`)

Interactive Gradio UI to watch AI vs AI Ludo games.

## Features
- Select a heuristic strategy for each color (dropdowns)
- Step the game one move at a time or run N auto steps
- Visual board: circular path + home columns + token IDs
- Shows captures, finishes, extra turns, winner announcement

## Run
```bash
python -m ludo_gr.app
```
Then open the local Gradio link.

## Architecture
- `board_viz.py` draws a lightweight approximate board (not pixel‑perfect official layout) focusing on clarity.
- `app.py` wires strategies via existing `StrategyFactory` and drives turns using the core `LudoGame` API.

## Future Enhancements
- Animation / per-move delay playback
- Highlight last moved token & capture flashes
- Download game log / JSON replay
- Human move input mode
- PPO policy integration hooks

