"""
Visualize Evaluation Results for Supervisor Presentation
Creates clear charts and graphs from validation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11


def load_results(json_path):
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(results, ax):
    """Plot confusion matrix as heatmap."""
    cm = np.array(results['confusion_matrix'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NOT Asian Hornet', 'Asian Hornet'],
                yticklabels=['NOT Asian Hornet', 'Asian Hornet'],
                cbar_kws={'label': 'Count'},
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\n(True Negatives: 188, False Positives: 9,\nFalse Negatives: 5, True Positives: 95)',
                 fontsize=13, fontweight='bold', pad=15)


def plot_metrics_bar(results, ax):
    """Plot key metrics as bar chart."""
    metrics = results['metrics']

    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    metric_values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['specificity'] * 100,
        metrics['f1_score'] * 100
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.barh(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value + 1, i, f'{value:.2f}%',
                va='center', fontweight='bold', fontsize=11)

    ax.set_xlim(0, 105)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics', fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=90, color='green', linestyle='--', alpha=0.3, linewidth=2, label='90% Target')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)


def plot_error_rates(results, ax):
    """Plot false positive and false negative rates."""
    metrics = results['metrics']

    error_types = ['False Positive\nRate', 'False Negative\nRate']
    error_values = [
        metrics['false_positive_rate'] * 100,
        metrics['false_negative_rate'] * 100
    ]

    colors = ['#e67e22', '#c0392b']
    bars = ax.bar(error_types, error_values, color=colors, edgecolor='black', linewidth=1.5, width=0.5)

    # Add value labels on bars
    for bar, value in zip(bars, error_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Error Rates (Lower is Better)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(error_values) + 2)
    ax.grid(axis='y', alpha=0.3)


def plot_class_distribution(results, ax):
    """Plot dataset class distribution."""
    counts = results['counts']

    labels = ['Asian Hornets\n(Positive)', 'European Hornets +\nWasps (Negative)']
    sizes = [counts['positive_samples'], counts['negative_samples']]
    colors = ['#e74c3c', '#3498db']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})

    # Bold the percentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax.set_title(f'Validation Set Distribution\n(Total: {counts["total_samples"]} images)',
                 fontsize=13, fontweight='bold', pad=15)


def plot_failure_breakdown(results, ax):
    """Plot breakdown of failure cases."""
    failure_cases = results['failure_cases']

    fp_cases = failure_cases['false_positives']
    fn_cases = failure_cases['false_negatives']

    # Count false positives by actual class
    fp_european = sum(1 for case in fp_cases if 'European' in case['actual'])
    fp_wasp = sum(1 for case in fp_cases if 'Wasp' in case['actual'])

    # Data for stacked bar
    categories = ['False Positives\n(9 total)', 'False Negatives\n(5 total)']
    european = [fp_european, 0]
    wasp = [fp_wasp, 0]
    asian = [0, len(fn_cases)]

    x = np.arange(len(categories))
    width = 0.5

    p1 = ax.bar(x, european, width, label='European Hornets', color='#f39c12', edgecolor='black')
    p2 = ax.bar(x, wasp, width, bottom=european, label='Wasps', color='#9b59b6', edgecolor='black')
    p3 = ax.bar(x, asian, width, label='Asian Hornets', color='#e74c3c', edgecolor='black')

    # Add value labels
    for i, (e, w, a) in enumerate(zip(european, wasp, asian)):
        if e > 0:
            ax.text(i, e/2, str(e), ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        if w > 0:
            ax.text(i, e + w/2, str(w), ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        if a > 0:
            ax.text(i, a/2, str(a), ha='center', va='center', fontweight='bold', color='white', fontsize=12)

    ax.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax.set_title('Failure Case Breakdown', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_probability_distribution(results, ax):
    """Plot distribution of prediction probabilities for failure cases."""
    failure_cases = results['failure_cases']

    fp_probs = [case['probability'] for case in failure_cases['false_positives']]
    fn_probs = [case['probability'] for case in failure_cases['false_negatives']]

    # Create bins
    bins = np.arange(0, 1.1, 0.1)

    ax.hist(fp_probs, bins=bins, alpha=0.7, label=f'False Positives (n={len(fp_probs)})',
            color='#e67e22', edgecolor='black', linewidth=1.5)
    ax.hist(fn_probs, bins=bins, alpha=0.7, label=f'False Negatives (n={len(fn_probs)})',
            color='#c0392b', edgecolor='black', linewidth=1.5)

    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold (0.5)')

    ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.set_title('Failure Case Probability Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def create_summary_text(results):
    """Create text summary for the visualization."""
    metrics = results['metrics']
    counts = results['counts']

    summary = f"""
    MODEL PERFORMANCE SUMMARY
    ========================

    Overall Accuracy: {metrics['accuracy']*100:.2f}%

    Asian Hornet Detection (CRITICAL):
      • Recall: {metrics['recall']*100:.2f}% (caught {counts['true_positives']}/{counts['positive_samples']} hornets)
      • Missed: {counts['false_negatives']} Asian hornets ({metrics['false_negative_rate']*100:.2f}%)

    Precision (When model says "Asian hornet"):
      • Correct: {metrics['precision']*100:.2f}%
      • False alarms: {counts['false_positives']} ({metrics['false_positive_rate']*100:.2f}%)

    Specificity (Correctly reject non-hornets):
      • {metrics['specificity']*100:.2f}%

    Dataset:
      • Total images: {counts['total_samples']}
      • Asian hornets: {counts['positive_samples']}
      • Non-hornets: {counts['negative_samples']}
    """

    return summary


def main():
    print("=" * 70)
    print("VISUALIZATION GENERATOR FOR VALIDATION RESULTS")
    print("=" * 70)
    print()

    # Paths
    model_dir = Path(__file__).parent / "models"
    json_path = model_dir / "validation_evaluation_detailed.json"
    output_path = model_dir / "validation_results_visualization.png"

    # Check if results exist
    if not json_path.exists():
        print(f"Error: Results file not found at {json_path}")
        print("Please run 'evaluate_full_validation.py' first!")
        return

    # Load results
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    print("Results loaded successfully!")
    print()

    # Create figure with subplots
    print("Generating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Plot 1: Confusion Matrix (top left, larger)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(results, ax1)

    # Plot 2: Metrics Bar Chart (top middle-right, larger)
    ax2 = fig.add_subplot(gs[0, 1:])
    plot_metrics_bar(results, ax2)

    # Plot 3: Error Rates (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_error_rates(results, ax3)

    # Plot 4: Class Distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_class_distribution(results, ax4)

    # Plot 5: Failure Breakdown (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_failure_breakdown(results, ax5)

    # Plot 6: Probability Distribution (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_probability_distribution(results, ax6)

    # Add title
    fig.suptitle('Asian Hornet Detector - EfficientNet-B3 Validation Results',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    print()

    # Print summary
    print(create_summary_text(results))

    # Show plot
    print("=" * 70)
    print("Opening visualization...")
    print("=" * 70)
    plt.show()


if __name__ == "__main__":
    main()
