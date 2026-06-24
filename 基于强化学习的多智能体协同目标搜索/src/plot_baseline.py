import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

categories = ['Heuristic Baseline', 'PPO RL (Final)']
success_rates = [56.0, 100.0]
avg_steps = [105.9, 8.4]

# Subplot 1: Success Rate
ax = axes[0]
bars = ax.bar(categories, success_rates, color=['#ff7f0e', '#2ca02c'], width=0.5, edgecolor='black')
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, 120)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
for bar, val in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Subplot 2: Average Steps
ax = axes[1]
bars = ax.bar(categories, avg_steps, color=['#ff7f0e', '#2ca02c'], width=0.5, edgecolor='black')
ax.set_ylabel('Average Steps to Find Target', fontsize=12)
ax.set_title('Search Efficiency Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, 130)
for bar, val in zip(bars, avg_steps):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

fig.suptitle('Heuristic Baseline vs PPO RL Performance', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assets/comparison/baseline_vs_rl.png', dpi=200, bbox_inches='tight')
print('[VIS] Saved baseline_vs_rl.png (split subplots)')
plt.close()
