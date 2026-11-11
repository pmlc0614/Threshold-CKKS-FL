"""Mode-checked FULL training entry point."""
from .encrypted_training import main as run_training

def main():
    run_training(expected_mode="FULL")

def main_one_round():
    run_training(default_rounds=1, expected_mode="FULL")

if __name__ == "__main__":
    main()
