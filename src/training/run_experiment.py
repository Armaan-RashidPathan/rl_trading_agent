# src/training/run_experiment.py
"""
Full Experiment Runner
- Longer training
- Walk-forward validation
- Saves all results for visualization
"""

import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.trainer import TrainingPipeline


def run_quick_experiment():
    """Quick experiment for verification (5-10 min)"""
    print("=" * 60)
    print("QUICK EXPERIMENT (50,000 steps)")
    print("=" * 60)
    
    pipeline = TrainingPipeline()
    results, sentiment_agent, baseline_agent = pipeline.run(
        total_timesteps=50000,
        test_ratio=0.2,
    )
    return results


def run_full_experiment():
    """Full experiment for final results (30-60 min)"""
    print("=" * 60)
    print("FULL EXPERIMENT (200,000 steps)")
    print("=" * 60)
    
    pipeline = TrainingPipeline()
    results, sentiment_agent, baseline_agent = pipeline.run(
        total_timesteps=200000,
        test_ratio=0.2,
    )
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trading experiment")
    parser.add_argument(
        '--mode', type=str, default='quick',
        choices=['quick', 'full'],
        help='quick=50k steps (~10min), full=200k steps (~45min)'
    )
    args = parser.parse_args()
    
    if args.mode == 'full':
        results = run_full_experiment()
    else:
        results = run_quick_experiment()