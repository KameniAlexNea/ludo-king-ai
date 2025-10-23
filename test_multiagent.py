"""Quick test script to verify multi-agent environment is working correctly."""

from __future__ import annotations

import numpy as np

from src.models.config import EnvConfig
from src.models.ludo_env_aec import env as make_aec_env


def test_basic_episode():
    """Run a basic episode to verify environment functionality."""
    print("Creating multi-agent Ludo environment...")
    cfg = EnvConfig(max_turns=50, seed=42)
    env = make_aec_env(cfg)
    
    print(f"Possible agents: {env.possible_agents}")
    print(f"Number of agents: {len(env.possible_agents)}")
    
    # Reset environment
    env.reset(seed=42)
    print(f"\nStarted episode with {len(env.agents)} active agents")
    print(f"First agent to act: {env.agent_selection}")
    
    # Run a few steps
    steps = 0
    max_steps = 20
    
    while env.agents and steps < max_steps:
        agent = env.agent_selection
        
        # Get observation and action mask
        obs = env.observe(agent)
        action_mask = env.action_mask(agent)
        
        # Sample valid action
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 0  # Fallback if no valid moves
        
        # Step environment
        env.step(action)
        
        steps += 1
        
        # Print info every 5 steps
        if steps % 5 == 0:
            print(f"Step {steps}: Agent {agent} took action {action}")
            if agent in env.rewards:
                print(f"  Reward: {env.rewards[agent]:.2f}")
    
    print(f"\nCompleted {steps} steps")
    print("✓ Multi-agent environment test passed!")
    
    return True


def test_observation_spaces():
    """Test that observation spaces are correctly defined."""
    print("\nTesting observation spaces...")
    cfg = EnvConfig()
    env = make_aec_env(cfg)
    env.reset(seed=42)
    
    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)
    
    print(f"Observation space type: {type(obs_space)}")
    print(f"Action space: {action_space}")
    
    # Get actual observation
    obs = env.observe(env.agent_selection)
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Sample observation shapes:")
    for key in list(obs.keys())[:5]:  # Show first 5
        print(f"  {key}: {obs[key].shape}")
    
    print("✓ Observation spaces test passed!")
    return True


def test_turn_based_mechanics():
    """Test that turn-based mechanics work correctly."""
    print("\nTesting turn-based mechanics...")
    cfg = EnvConfig(max_turns=100, seed=123)
    env = make_aec_env(cfg)
    env.reset(seed=123)
    
    agent_turns = {agent: 0 for agent in env.possible_agents}
    steps = 0
    max_steps = 50
    
    while env.agents and steps < max_steps:
        agent = env.agent_selection
        agent_turns[agent] += 1
        
        # Take action
        action_mask = env.action_mask(agent)
        valid_actions = np.where(action_mask)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        
        env.step(action)
        steps += 1
    
    print(f"Turn distribution after {steps} steps:")
    for agent, turns in agent_turns.items():
        print(f"  {agent}: {turns} turns")
    
    # Verify turns are reasonably distributed (not perfectly equal due to extra turns)
    turn_counts = list(agent_turns.values())
    min_turns = min(turn_counts)
    max_turns = max(turn_counts)
    
    # Should not have huge disparities unless one agent gets many extra turns
    assert max_turns - min_turns < 30, "Turn distribution is very unbalanced"
    
    print("✓ Turn-based mechanics test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Multi-Agent Ludo Environment Test Suite")
    print("=" * 70)
    
    try:
        test_basic_episode()
        test_observation_spaces()
        test_turn_based_mechanics()
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print("\nYour multi-agent environment is ready to use.")
        print("Try: python src/train_multiagent.py --total-steps 100000")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
