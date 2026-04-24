#!/usr/bin/env python3
"""
batch_train_dynamic.py - Batch training with dynamic complexity for all architectures.
Runs run.py train subprocesses for each network type (lstm, transformer, multimemory)
both with and without auxiliary tasks, using dynamic complexity by default.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def find_existing_model(model_dir, arch, auxiliary):
    """Check if a best.pt model already exists for this architecture/aux setting"""
    if not model_dir.exists():
        return None
    # Look for any _best.pt file matching the pattern (timestamp varies)
    # We'll just check existence of any best.pt, assuming it's a complete model.
    best_files = list(model_dir.glob(f"{arch}_*_best.pt"))
    if auxiliary:
        # For auxiliary models, the filenames contain "_aux" before _best.pt
        best_files = [f for f in best_files if "_aux_best.pt" in str(f)]
    else:
        best_files = [f for f in best_files if "_aux" not in str(f)]
    return best_files[0] if best_files else None


def train_model(arch, auxiliary, epochs, batch_size, lr, force, quiet):
    """Train one model by calling run.py train with dynamic complexity."""
    # Determine save directory (consistent with run.py's default)
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    # Check existing model (optional)
    if not force and find_existing_model(save_dir, arch, auxiliary):
        if not quiet:
            print(f"✓ Existing model found for {arch}{' (aux)' if auxiliary else ''}, skipping.")
        return True

    # Build timestamped experiment name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{arch}_{batch_size}b_{lr}lr_{timestamp}"
    if auxiliary:
        exp_name += "_aux"

    # Base command
    cmd = [
        sys.executable, "run.py", "train",
        "--network-type", arch,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--experiment-name", exp_name,
        "--save-dir", str(save_dir),
        "--dynamic-complexity"
    ]
    if auxiliary:
        cmd.append("--auxiliary-tasks")

    if not quiet:
        print(f"\n{'='*70}")
        print(f"Launching: {' '.join(cmd)}")
        print(f"{'='*70}")

    # Run the subprocess
    try:
        if quiet:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error training {arch}{' (aux)' if auxiliary else ''}: {result.stderr}")
                return False
        else:
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            process.wait()
            if process.returncode != 0:
                return False
        return True
    except Exception as e:
        print(f"Exception while training {arch}{' (aux)' if auxiliary else ''}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch train all architectures with dynamic complexity curriculum."
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs per training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed per‑run output")
    parser.add_argument("--skip-aux", action="store_true", help="Train without auxiliary tasks only")
    parser.add_argument("--skip-no-aux", action="store_true", help="Train with auxiliary tasks only")
    parser.add_argument("--networks", nargs="+", default=["lstm", "transformer", "multimemory"],
                        choices=["lstm", "transformer", "multimemory"],
                        help="Which network types to train (default: all three)")

    args = parser.parse_args()

    # Determine auxiliary settings to run
    if args.skip_aux and args.skip_no_aux:
        print("Error: Cannot skip both auxiliary and non‑auxiliary.")
        return 1
    aux_settings = []
    if not args.skip_aux:
        aux_settings.append(False)
    if not args.skip_no_aux:
        aux_settings.append(True)

    experiments = [(net, aux) for net in args.networks for aux in aux_settings]
    if not experiments:
        print("No experiments to run.")
        return 1

    print(f"Batch training {len(experiments)} experiment(s) with dynamic complexity:")
    for net, aux in experiments:
        print(f"  - {net}{' (aux)' if aux else ''}")

    results = []
    for net, aux in tqdm(experiments, desc="Overall progress", disable=args.quiet):
        success = train_model(
            arch=net,
            auxiliary=aux,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            force=args.force,
            quiet=args.quiet
        )
        results.append((net, aux, success))
        time.sleep(2)  # short pause between runs

    # Summary
    print("\n" + "="*70)
    print("BATCH TRAINING SUMMARY (dynamic complexity)")
    print("="*70)
    for net, aux, success in results:
        status = "✓" if success else "✗"
        aux_str = " (aux)" if aux else ""
        print(f"  {status} {net}{aux_str}")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())