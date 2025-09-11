import os

from ludo.player import PlayerColor


def load_ppo_wrapper(env_kind):
    """Dynamically import the correct EnvConfig and PPOStrategy based on env_kind."""
    if env_kind == "classic":
        from ludo_rl.envs.model import EnvConfig as ClassicEnvConfig
        from ludo_rl.ppo_strategy import PPOStrategy as ClassicPPOStrategy

        EnvConfigClass = ClassicEnvConfig
        PPOStrategyClass = ClassicPPOStrategy
    else:  # single-seat
        from ludo_rls.envs.model import EnvConfig as SingleEnvConfig
        from ludo_rls.ppo_strategy import PPOStrategy as SinglePPOStrategy

        EnvConfigClass = SingleEnvConfig
        PPOStrategyClass = SinglePPOStrategy

    return EnvConfigClass, PPOStrategyClass


def select_best_ppo_model(models_dir, model_preference):
    """Select the best PPO model based on configured preference."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory {models_dir} not found")

    model_files: list[str] = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    if not model_files:
        raise FileNotFoundError(f"No PPO model files found in {models_dir}/")
    
    # Define preference order
    if model_preference == "best":
        prefs = ["best", "final", "steps"]
    elif model_preference == "final":
        prefs = ["final", "best", "steps"]
    elif model_preference == "steps":
        prefs = ["steps", "best", "final"]
    
    # Try each preference in order
    for pref in prefs:
        if pref == "best":
            best_model = next((f for f in model_files if "best" in f.lower()), None)
            if best_model:
                return best_model.replace(".zip", "")
        elif pref == "final":
            final_model = next((f for f in model_files if "final" in f.lower()), None)
            if final_model:
                return final_model.replace(".zip", "")
        elif pref == "steps":
            # Extract step numbers and find highest
            step_models = []
            for f in model_files:
                try:
                    # Extract number from filename like "ppo_ludo_1000000_steps"
                    parts = f.replace(".zip", "").split("_")
                    for part in parts:
                        if part.isdigit():
                            step_models.append((int(part), f.replace(".zip", "")))
                            break
                except Exception:
                    continue
            if step_models:
                step_models.sort(reverse=True)
                return step_models[0][1]

    # Fallback to first model
    return model_files[0].replace(".zip", "")


def load_ppo_strategy(env_kind: str, models_dir: str, player_name: str = "ppo", agent_color: PlayerColor = PlayerColor.RED, model_preference: str = "last"):
    """Load and return the PPO strategy instance."""
    EnvConfigClass, PPOStrategyClass = load_ppo_wrapper(env_kind)
    model_name = select_best_ppo_model(models_dir, model_preference)
    model_path = os.path.join(models_dir, f"{model_name}.zip")
    
    try:
        strategy = PPOStrategyClass(
            model_path,
            player_name,
            EnvConfigClass(agent_color=agent_color.value),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize PPO strategy for model '{model_name}': {e}"
        ) from e
    
    return strategy