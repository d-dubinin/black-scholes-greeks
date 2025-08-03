# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs_option import BlackScholesOption

sns.set(style='darkgrid', context='notebook', palette='deep')

# Parameters
K = 100
T = 1.0
t = 0
r = 0.05
sigma = 0.2
S_vals = np.linspace(50, 150, 200)

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

def plot_greek(S_vals, greek_name, greek_func):
    call_values = []
    put_values = []
    for S in S_vals:
        call = BlackScholesOption(S_t=S, K=K, T=T, t=t, r=r, sigma=sigma, option_type='call')
        put = BlackScholesOption(S_t=S, K=K, T=T, t=t, r=r, sigma=sigma, option_type='put')
        call_values.append(greek_func(call))
        put_values.append(greek_func(put))

    plt.figure(figsize=(8, 5))
    plt.plot(S_vals, call_values, label='Call', linewidth=2)
    if greek_name not in ['Gamma', 'Vega']:
        plt.plot(S_vals, put_values, label='Put', linestyle='--', linewidth=2)

    plt.axvline(x=K, color='gray', linestyle=':', linewidth=1.5, label=f'Strike (K = {K})')
    plt.title(f'{greek_name} vs Stock Price (t = {t}, T = {T})')
    plt.xlabel('Stock Price (S)')
    plt.ylabel(greek_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{greek_name.lower()}.png'), dpi=300)
    plt.close()
    print(f"Saved {greek_name.lower()}.png")

# Run tests and save plots
if __name__ == '__main__':
    plot_greek(S_vals, 'Delta', lambda opt: opt.delta())
    plot_greek(S_vals, 'Gamma', lambda opt: opt.gamma())
    plot_greek(S_vals, 'Vega', lambda opt: opt.vega())
    plot_greek(S_vals, 'Theta', lambda opt: opt.theta())
