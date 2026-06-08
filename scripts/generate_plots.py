import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
REPORTS_DIR = 'reports'

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set publication-quality plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
BIT_ORDER = ['2.0', '2.5', '3.0', '3.5', '4.0', 'BF16']
PALETTE = sns.color_palette("Spectral", n_colors=6)

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
print("Loading data...")
# Load master experiment sheet
exp_file = os.path.join(RESULTS_DIR, 'experiment_results.csv')
exp_df = pd.read_csv(exp_file)
exp_df['bit_width'] = exp_df['bit_width'].fillna('BF16').astype(str)

# Load all individual trial sheets
trial_files = glob.glob(os.path.join(RESULTS_DIR, 'trials_results_exp_*.csv'))
trials_df = pd.concat([pd.read_csv(f) for f in trial_files], ignore_index=True)
trials_df['bit_width'] = trials_df['bit_width'].fillna('BF16').astype(str)

# Force categorical ordering
trials_df['bit_width'] = pd.Categorical(trials_df['bit_width'], categories=BIT_ORDER, ordered=True)
exp_df['bit_width'] = pd.Categorical(exp_df['bit_width'], categories=BIT_ORDER, ordered=True)
exp_df = exp_df.sort_values('bit_width')

# ==========================================
# 2. DELIVERABLE 3: DISTRIBUTION PLOTS (PDF & CDF)
# ==========================================
print("Generating Distribution Plots...")

def plot_distributions(df, metric, title_prefix, filename_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: PDF (Kernel Density Estimate)
    sns.kdeplot(data=df, x=metric, hue='bit_width', fill=True, common_norm=False, 
                palette=PALETTE, alpha=0.3, ax=axes[0], linewidth=2)
    axes[0].set_title(f'{title_prefix} - Probability Density (PDF)')
    axes[0].set_xlabel(title_prefix)
    axes[0].set_ylabel('Density')
    
    # Right: CDF (Cumulative Distribution Function)
    sns.ecdfplot(data=df, x=metric, hue='bit_width', palette=PALETTE, ax=axes[1], linewidth=2.5)
    axes[1].set_title(f'{title_prefix} - Cumulative Distribution (CDF)')
    axes[1].set_xlabel(title_prefix)
    axes[1].set_ylabel('Cumulative Probability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{filename_prefix}_distribution.pdf'), dpi=300)
    plt.close()

# Avoid plotting BF16 for RMSE since it is identically 0.0 (causes KDE errors)
quant_trials = trials_df[trials_df['bit_width'] != 'BF16']

plot_distributions(trials_df, 'perplexity', 'Sequence Perplexity', 'PPL')
plot_distributions(quant_trials, 'rmse_key', 'Attention Key RMSE', 'RMSE_Key')
plot_distributions(quant_trials, 'rmse_value', 'Attention Value RMSE', 'RMSE_Value')

# ==========================================
# 3. DELIVERABLE 4: MOMENT SUMMARY TABLES
# ==========================================
print("Calculating Moment Summaries (Mean, Variance, Skewness)...")

def calculate_skew(x):
    return stats.skew(x.dropna())

moments_df = trials_df.groupby('bit_width').agg({
    'perplexity': ['mean', 'var', calculate_skew],
    'rmse_key': ['mean', 'var', calculate_skew],
    'rmse_value': ['mean', 'var', calculate_skew]
}).round(5)

# Flatten MultiIndex columns
moments_df.columns = ['_'.join(col).replace('calculate_skew', 'skew') for col in moments_df.columns.values]
moments_df.to_csv(os.path.join(REPORTS_DIR, 'moments_summary.csv'))

# ==========================================
# 4. DELIVERABLE 5: 95% CONFIDENCE INTERVALS
# ==========================================
print("Calculating exact 95% Clopper-Pearson Confidence Intervals...")

def clopper_pearson(k, n, alpha=0.05):
    """Calculate exact binomial confidence intervals."""
    if n == 0: return 0.0, 0.0
    lower = stats.beta.ppf(alpha/2, k, n - k + 1) if k > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha/2, k + 1, n - k) if k < n else 1.0
    return lower, upper

ci_records = []
for bw in BIT_ORDER:
    bw_data = trials_df[trials_df['bit_width'] == bw]
    n_total = len(bw_data)
    
    if n_total == 0: continue
        
    correct = bw_data['evaluation'].sum()
    lower, upper = clopper_pearson(correct, n_total)
    mean_acc = correct / n_total
    
    ci_records.append({
        'bit_width': bw,
        'accuracy': mean_acc,
        'ci_lower': lower,
        'ci_upper': upper,
        'yerr_lower': mean_acc - lower,
        'yerr_upper': upper - mean_acc
    })

ci_df = pd.DataFrame(ci_records)

# Plot Zero-Shot Accuracy with Error Bars
plt.figure(figsize=(10, 6))
plt.errorbar(ci_df['bit_width'], ci_df['accuracy'], 
             yerr=[ci_df['yerr_lower'], ci_df['yerr_upper']], 
             fmt='-o', capsize=5, capthick=2, markersize=8, 
             color='black', ecolor='red', linewidth=2)
plt.title('Global Zero-Shot Accuracy with 95% Confidence Intervals')
plt.xlabel('Quantization Bit-Width')
plt.ylabel('Accuracy Rate')
plt.ylim(0, 1.05)
plt.axhline(ci_df[ci_df['bit_width'] == 'BF16']['accuracy'].values[0], color='gray', linestyle='--', label='BF16 Baseline')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_95CI.pdf'), dpi=300)
plt.close()

# ==========================================
# 5. DELIVERABLE 6: FREQUENTIST SIGNIFICANCE REPORT
# ==========================================
print("Running ANOVA Tests...")

with open(os.path.join(REPORTS_DIR, 'anova_significance_report.txt'), 'w') as f:
    f.write("=== FREQUENTIST SIGNIFICANCE REPORT (ANOVA) ===\n\n")
    
    metrics = ['perplexity', 'rmse_key', 'rmse_value']
    for metric in metrics:
        # Group data by bit_width
        groups = [group[metric].dropna().values for name, group in trials_df.groupby('bit_width') if name != 'BF16']
        
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            f.write(f"Metric: {metric.upper()}\n")
            f.write(f"F-Statistic: {f_stat:.4f}\n")
            f.write(f"P-Value: {p_val:.4e}\n")
            if p_val < 0.05:
                f.write("Conclusion: Statistically SIGNIFICANT variance across bit-widths.\n")
            else:
                f.write("Conclusion: NO statistically significant variance across bit-widths.\n")
            f.write("-" * 40 + "\n")

# ==========================================
# 6. MASTER RECOVERY CURVE (The "Money Plot")
# ==========================================
print("Plotting the Dual-Axis Recovery Curve...")

fig, ax1 = plt.subplots(figsize=(12, 7))

# X-axis mapping
x_labels = exp_df['bit_width']
x_pos = np.arange(len(x_labels))

# Axis 1: Accuracies (Bars)
width = 0.2
ax1.bar(x_pos - width, exp_df['fqa_accuracy'], width, label='Factual QA', color='#4C72B0', alpha=0.8)
ax1.bar(x_pos, exp_df['crqa_accuracy'], width, label='Cross-Reference QA', color='#55A868', alpha=0.8)
ax1.bar(x_pos + width, exp_df['oosqa_accuracy'], width, label='Out-of-Scope QA', color='#C44E52', alpha=0.8)

ax1.set_xlabel('Quantization Bit-Width', fontweight='bold')
ax1.set_ylabel('Accuracy Rate', fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Axis 2: RMSE Key (Line)
ax2 = ax1.twinx()
ax2.plot(x_pos, exp_df['mean_rmse_key'], color='black', marker='D', markersize=8, 
         linewidth=3, label='Mean Key RMSE (Geometric Error)')
ax2.set_ylabel('Root Mean Square Error (RMSE)', color='black', fontweight='bold')
ax2.set_ylim(0, max(exp_df['mean_rmse_key']) * 1.2)
ax2.tick_params(axis='y', labelcolor='black')

# Combined Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)

plt.title('TurboQuant Phase Transition: Accuracy Recovery vs. RMSE Degradation', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'turboquant_recovery_curve.pdf'), dpi=300)
plt.savefig(os.path.join(PLOTS_DIR, 'turboquant_recovery_curve.png'), dpi=300)
plt.close()

print("✅ Analysis Complete! Check the 'plots/' and 'reports/' directories.")
