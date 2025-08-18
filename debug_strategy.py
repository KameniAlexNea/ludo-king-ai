#!/usr/bin/env python3
"""Debug strategy interface"""

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory

# Create real game context
game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
context = game.get_ai_decision_context(6)

print("REAL CONTEXT STRUCTURE:")
print("Keys:", list(context.keys()))
print()

print("Valid moves structure:")
if context["valid_moves"]:
    print("First move keys:", list(context["valid_moves"][0].keys()))
    print("First move:", context["valid_moves"][0])
print()

# Test strategy with real context
strategy = StrategyFactory.create_strategy("killer")
try:
    decision = strategy.decide(context)
    print("SUCCESS: Strategy decision =", decision)
except Exception as e:
    print("ERROR:", str(e))
    print("Exception type:", type(e).__name__)
    import traceback

    traceback.print_exc()
