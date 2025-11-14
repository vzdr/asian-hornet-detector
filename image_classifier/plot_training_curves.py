"""
Generate Training/Validation Curve Plots
Shows overfitting analysis for supervisor documentation
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves(results_file, output_dir):
    """Generate training/validation loss and accuracy plots"""

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    history = results['history']
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    asian_recall = [h['asian_recall'] for h in history]
    negative_acc = [h['negative_accuracy'] for h in history]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=6, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 6)')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].axvline(x=6, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 6)')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='lower right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Asian Hornet Recall
    axes[1, 0].plot(epochs, asian_recall, 'g-', linewidth=2)
    axes[1, 0].axvline(x=6, color='green', linestyle='--', alpha=0.7, label='Best Model')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Asian Hornet Recall (Sensitivity)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='lower right', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.9, 1.0])

    # Plot 4: Negative Class Accuracy (Specificity proxy)
    axes[1, 1].plot(epochs, negative_acc, 'm-', linewidth=2)
    axes[1, 1].axvline(x=6, color='green', linestyle='--', alpha=0.7, label='Best Model')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Negative Class Accuracy (Specificity)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='lower right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.9, 1.0])

    # Overall title
    fig.suptitle('30k Balanced Model - Training History (120k samples, 30k per class)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_file = output_dir / 'training_validation_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Create separate overfitting analysis plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s')

    # Highlight best epoch
    best_epoch = 6
    best_val_acc = val_acc[5]  # epoch 6 = index 5
    ax.scatter([best_epoch], [best_val_acc], color='green', s=200, zorder=5,
               label=f'Best Model (Epoch {best_epoch}, Val Acc: {best_val_acc:.2f}%)')

    # Show overfitting region
    ax.axvspan(6, 20, alpha=0.2, color='red', label='Overfitting Region')

    # Add annotations
    ax.annotate('Peak validation accuracy\n(Early stopping point)',
                xy=(6, best_val_acc), xytext=(10, best_val_acc - 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')

    ax.annotate('Training accuracy continues improving\nwhile validation declines',
                xy=(15, 99.95), xytext=(12, 97.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis: Train vs Validation Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([94, 100])

    plt.tight_layout()

    output_file2 = output_dir / 'overfitting_analysis.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file2}")

    plt.close('all')

    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Best Validation Accuracy: {max(val_acc):.2f}% (Epoch {val_acc.index(max(val_acc)) + 1})")
    print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_acc[-1]:.2f}%")
    print(f"Train-Val Gap at Best Epoch: {train_acc[5] - val_acc[5]:.2f}%")
    print(f"Train-Val Gap at Final Epoch: {train_acc[-1] - val_acc[-1]:.2f}%")
    print(f"\nBest Model Metrics (Epoch 6):")
    print(f"  Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"  Asian Hornet Recall: {results['best_recall']*100:.2f}%")
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  Specificity: {results['specificity']*100:.2f}%")
    print(f"  F1-Score: {results['f1_score']*100:.2f}%")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    results_file = Path(__file__).parent / "models" / "balanced_30k_efficientnet_results.json"
    output_dir = Path(__file__).parent / "training_analysis"

    print("Generating training/validation curve plots...")
    plot_training_curves(results_file, output_dir)
    print("\nDone! Plots saved for supervisor documentation.")
