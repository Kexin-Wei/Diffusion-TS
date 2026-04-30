from pathlib import Path


def get_results_folder(config, cfg, seq_length):
    return Path("checkpoints") / f"{cfg}_seq_{seq_length}"
