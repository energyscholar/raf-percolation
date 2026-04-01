"""
Plot RAF transition curves and finite-size scaling analysis.

Reads sweep data from CSV and produces paper figures:
    Fig 1: P(RAF) vs lambda for multiple system sizes
    Fig 2: Order parameter (mean RAF fraction) vs lambda
    Fig 3: Susceptibility (variance) vs lambda
    Fig 4: Finite-size scaling collapse

Usage:
    python plot_transition.py [--data ../data/raf_transition_sweep.csv]
                              [--output_dir ../figures]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (5, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def sigmoid(x, x0, k):
    """Logistic sigmoid for fitting P(RAF) curves."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def find_lambda_c(df_n: pd.DataFrame) -> float:
    """Estimate lambda_c from sigmoid fit to P(RAF) vs lambda."""
    lam = df_n['lambda'].values
    p_raf = df_n['p_raf'].values
    try:
        popt, _ = curve_fit(sigmoid, lam, p_raf,
                            p0=[np.median(lam), 5.0],
                            maxfev=5000)
        return popt[0]
    except RuntimeError:
        # Fallback: lambda where P(RAF) crosses 0.5
        idx = np.argmin(np.abs(p_raf - 0.5))
        return lam[idx]


def fig1_praf_vs_lambda(df: pd.DataFrame, output_dir: str):
    """Fig 1: P(RAF) vs lambda for multiple system sizes."""
    fig, ax = plt.subplots()
    n_values = sorted(df['n'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    for n_val, color in zip(n_values, colors):
        sub = df[df['n'] == n_val].sort_values('lambda')
        ax.plot(sub['lambda'], sub['p_raf'], 'o-', color=color,
                markersize=3, linewidth=1.2, label=f'$n = {n_val}$')

    # Reference line: Hordijk et al. linear fit
    lam_ref = np.array([1.097 + 0.019 * n for n in n_values])
    for lc in lam_ref:
        ax.axvline(lc, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'Average catalysis level $\lambda$')
    ax.set_ylabel(r'$P(\mathrm{RAF\ exists})$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_title('RAF emergence transition')

    outpath = os.path.join(output_dir, 'fig1_praf_vs_lambda.pdf')
    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


def fig2_order_parameter(df: pd.DataFrame, output_dir: str):
    """Fig 2: Order parameter (mean RAF fraction) vs lambda."""
    fig, ax = plt.subplots()
    n_values = sorted(df['n'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    for n_val, color in zip(n_values, colors):
        sub = df[df['n'] == n_val].sort_values('lambda')
        ax.plot(sub['lambda'], sub['mean_raf_frac'], 'o-', color=color,
                markersize=3, linewidth=1.2, label=f'$n = {n_val}$')

    ax.set_xlabel(r'Average catalysis level $\lambda$')
    ax.set_ylabel(r'$\langle |\mathrm{maxRAF}| \rangle / |\mathcal{R}|$')
    ax.legend(loc='lower right')
    ax.set_title('Order parameter')

    outpath = os.path.join(output_dir, 'fig2_order_parameter.pdf')
    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


def fig3_susceptibility(df: pd.DataFrame, output_dir: str):
    """Fig 3: Susceptibility (variance of RAF size) vs lambda."""
    fig, ax = plt.subplots()
    n_values = sorted(df['n'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    for n_val, color in zip(n_values, colors):
        sub = df[df['n'] == n_val].sort_values('lambda')
        # Normalized susceptibility
        chi = sub['var_raf_size'] / (sub['n_reactions'] ** 2)
        ax.plot(sub['lambda'], chi, 'o-', color=color,
                markersize=3, linewidth=1.2, label=f'$n = {n_val}$')

    ax.set_xlabel(r'Average catalysis level $\lambda$')
    ax.set_ylabel(r'$\chi = \mathrm{Var}(|\mathrm{maxRAF}|) / |\mathcal{R}|^2$')
    ax.legend(loc='upper right')
    ax.set_title('Susceptibility')

    outpath = os.path.join(output_dir, 'fig3_susceptibility.pdf')
    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()


def fig4_scaling_collapse(df: pd.DataFrame, output_dir: str):
    """Fig 4: Finite-size scaling collapse of P(RAF) curves.

    Attempt collapse: P(RAF) = f((lambda - lambda_c(n)) * n^{1/nu})
    Scan over nu to find best collapse.
    """
    n_values = sorted(df['n'].unique())
    if len(n_values) < 3:
        print("Need >= 3 system sizes for scaling collapse. Skipping Fig 4.")
        return

    # Estimate lambda_c for each n
    lambda_c = {}
    for n_val in n_values:
        sub = df[df['n'] == n_val]
        lambda_c[n_val] = find_lambda_c(sub)

    # Scan nu values, find best collapse via minimizing spread
    best_nu = 1.0
    best_cost = np.inf

    for nu_trial in np.linspace(0.3, 4.0, 100):
        # For each n, compute scaled variable
        all_x = []
        all_y = []
        for n_val in n_values:
            sub = df[df['n'] == n_val].sort_values('lambda')
            x_scaled = (sub['lambda'].values - lambda_c[n_val]) * n_val**(1.0/nu_trial)
            all_x.append(x_scaled)
            all_y.append(sub['p_raf'].values)

        # Measure collapse quality: interpolate onto common x grid,
        # measure variance across curves
        x_all = np.concatenate(all_x)
        x_min, x_max = np.percentile(x_all, [10, 90])
        x_grid = np.linspace(x_min, x_max, 50)

        y_interp = []
        for x_s, y_s in zip(all_x, all_y):
            sort_idx = np.argsort(x_s)
            y_i = np.interp(x_grid, x_s[sort_idx], y_s[sort_idx])
            y_interp.append(y_i)

        y_interp = np.array(y_interp)
        # Cost = mean variance across curves at each x point
        cost = np.mean(np.var(y_interp, axis=0))
        if cost < best_cost:
            best_cost = cost
            best_nu = nu_trial

    print(f"Best collapse: nu = {best_nu:.2f} (cost = {best_cost:.6f})")
    print(f"lambda_c estimates: {lambda_c}")

    # Plot with best nu
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    for n_val, color in zip(n_values, colors):
        sub = df[df['n'] == n_val].sort_values('lambda')
        x_scaled = (sub['lambda'].values - lambda_c[n_val]) * n_val**(1.0/best_nu)
        ax.plot(x_scaled, sub['p_raf'], 'o', color=color,
                markersize=4, alpha=0.7, label=f'$n = {n_val}$')

    ax.set_xlabel(r'$(\lambda - \lambda_c) \cdot n^{1/\nu}$'
                  f'  ($\\nu = {best_nu:.2f}$)')
    ax.set_ylabel(r'$P(\mathrm{RAF\ exists})$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_title('Finite-size scaling collapse')

    outpath = os.path.join(output_dir, 'fig4_scaling_collapse.pdf')
    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close()

    # Save extracted parameters
    params_path = os.path.join(output_dir, 'scaling_parameters.txt')
    with open(params_path, 'w') as f:
        f.write(f"Best nu: {best_nu:.4f}\n")
        f.write(f"Collapse cost: {best_cost:.6f}\n")
        for n_val in n_values:
            f.write(f"lambda_c(n={n_val}): {lambda_c[n_val]:.4f}\n")
    print(f"Saved {params_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot RAF transition figures.')
    parser.add_argument('--data', type=str,
                        default='../data/raf_transition_sweep.csv')
    parser.add_argument('--output_dir', type=str, default='../figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows, n values: {sorted(df['n'].unique())}")

    fig1_praf_vs_lambda(df, args.output_dir)
    fig2_order_parameter(df, args.output_dir)
    fig3_susceptibility(df, args.output_dir)
    fig4_scaling_collapse(df, args.output_dir)


if __name__ == '__main__':
    main()
