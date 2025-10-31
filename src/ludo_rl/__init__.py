"""Ludo RL: Independent PPO multi-agent training package.

This package provides a clean, IPPO-style multi-agent trainer with
- 4 distinct PPO policies (one per seat)
- a single-seat self-play environment that exposes only the learner's turns
  and executes all opponent turns internally using other policies or scripted bots
- tournament-style evaluation to pick a champion policy

It is intentionally decoupled from previous training scripts to provide
clear attribution and visibility into "who is learning".
"""
