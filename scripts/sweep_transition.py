"""
Sweep catalysis parameter across the RAF transition.

For each system size n, sweeps the average catalysis level lambda
through the transition region and measures:
    - P(RAF exists)
    - <|maxRAF|> / |R|  (order parameter)
    - Var(|maxRAF|)     (susceptibility proxy)

Outputs CSV files for downstream analysis.

Usage:
    python sweep_transition.py [--n_values 5 8 10] [--n_samples 500]
                               [--n_lambda 40] [--output_dir ../data]
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from binary_polymer_model import build_crs, raf_size, generate_molecules, generate_reactions


def compute_lambda_range(n: int, n_points: int = 40) -> np.ndarray:
    """Compute lambda sweep range centered on expected transition.

    From Hordijk et al. (2011): lambda_c ~ 1.097 + 0.019*n
    We sweep from 0.5 * lambda_c to 2.5 * lambda_c.
    """
    lambda_c_est = 1.097 + 0.019 * n
    return np.linspace(0.3 * lambda_c_est, 3.0 * lambda_c_est, n_points)


def lambda_to_p(lam: float, n_reactions: int) -> float:
    """Convert average catalysis level lambda to probability p.

    lambda = p * |R|, so p = lambda / |R|.
    """
    return lam / n_reactions


def sweep_single_n(n: int, n_samples: int, n_lambda: int,
                   seed: int = 42) -> list:
    """Run the sweep for a single system size.

    Returns:
        List of dicts with keys: n, lambda, p, n_reactions, n_molecules,
        p_raf, mean_raf_size, var_raf_size, mean_raf_frac.
    """
    # Precompute model structure (shared across samples)
    molecules = generate_molecules(n)
    reactions = generate_reactions(molecules)
    n_mol = len(molecules)
    n_rxn = len(reactions)

    print(f"n={n}: {n_mol} molecules, {n_rxn} reactions")

    lambda_values = compute_lambda_range(n, n_lambda)
    results = []

    rng = np.random.default_rng(seed)

    for i, lam in enumerate(lambda_values):
        p = lambda_to_p(lam, n_rxn)
        raf_sizes = []

        t0 = time.time()
        for _ in range(n_samples):
            crs = build_crs(n, p, rng=rng)
            size = raf_size(crs)
            raf_sizes.append(size)

        raf_sizes = np.array(raf_sizes)
        elapsed = time.time() - t0

        p_raf = np.mean(raf_sizes > 0)
        mean_size = np.mean(raf_sizes)
        var_size = np.var(raf_sizes)
        mean_frac = mean_size / n_rxn if n_rxn > 0 else 0

        results.append({
            'n': n,
            'lambda': lam,
            'p': p,
            'n_molecules': n_mol,
            'n_reactions': n_rxn,
            'n_samples': n_samples,
            'p_raf': p_raf,
            'mean_raf_size': mean_size,
            'var_raf_size': var_size,
            'mean_raf_frac': mean_frac,
        })

        print(f"  lambda={lam:.3f} p={p:.2e} P(RAF)={p_raf:.3f} "
              f"<|RAF|>={mean_size:.1f} ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Sweep catalysis parameter across RAF transition.')
    parser.add_argument('--n_values', type=int, nargs='+',
                        default=[5, 6, 7, 8],
                        help='System sizes to simulate')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of random instances per (n, lambda)')
    parser.add_argument('--n_lambda', type=int, default=40,
                        help='Number of lambda values to sweep')
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Output directory for CSV files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for n in args.n_values:
        results = sweep_single_n(n, args.n_samples, args.n_lambda,
                                 seed=args.seed + n)
        all_results.extend(results)

    # Write combined CSV
    outfile = os.path.join(args.output_dir, 'raf_transition_sweep.csv')
    fieldnames = ['n', 'lambda', 'p', 'n_molecules', 'n_reactions',
                  'n_samples', 'p_raf', 'mean_raf_size', 'var_raf_size',
                  'mean_raf_frac']
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {outfile}")


if __name__ == '__main__':
    main()
