import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_kfold(cv, X, y, ax, n_splits, xlim_max=100):
    """
    Plots the indices for a cross-validation object.

    Parameters:
    cv: Cross-validation object
    X: Feature set
    y: Target variable
    ax: Matplotlib axis object
    n_splits: Number of folds in the cross-validation
    xlim_max: Maximum limit for the x-axis
    """

    # Define colors
    training_color = 'gold'
    validation_color = 'forestgreen'
    
    cv_split = cv.split(X=X, y=y)

    for i_split, (train_idx, test_idx) in enumerate(cv_split):
        # Create an array of NaNs and fill in training/testing indices
        indices = np.full(len(X), np.nan)
        indices[test_idx], indices[train_idx] = 1, 0

        # Create colors array
        colors = np.array([validation_color if idx == 1 else training_color for idx in indices])
        colors[np.isnan(indices)] = 'none'  # Set NaN indices to transparent
        
        # Plot the training and testing indices
        ax_x = range(len(indices))
        ax_y = [i_split + 0.5] * len(indices)
        ax.scatter(
            ax_x, ax_y, c=colors, marker="_", lw=10
        )

    # Set y-ticks and labels
    y_ticks = np.arange(n_splits) + 0.5
    ax.set(
        yticks=y_ticks,
        yticklabels=range(n_splits),
        xlabel="X index",
        ylabel="Fold",
        ylim=[n_splits, -0.2],
        xlim=[0, xlim_max],
    )

    # Set plot title and create legend
    ax.set_title("KFold", fontsize=14)
    legend_patches = [
        Patch(color=validation_color, label="Validation set"),
        Patch(color=training_color, label="Training set"),
    ]
    ax.legend(handles=legend_patches, loc=(1.03, 0.8))
