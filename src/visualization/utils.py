import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown


def get_default_title(feature_name):
    core_feature_name = ' '.join(feature_name.split("_")[1:]).title()
    return 'Winner distribution by Difference of ' + core_feature_name


def comp_featured_based_on_winner(data, feature, sns_kws={}, num_bins=30, title=None):
    # Create new figure if there isn't one created
    if not plt.get_fignums():
        plt.figure(figsize=(8, 6))

    num_bins = min(data[feature].nunique(), num_bins)
    ax = sns.histplot(
        data,
        x=feature, hue="winner",
        hue_order=[0, 1],
        multiple="fill",
        bins=num_bins,
        stat="probability",
        **sns_kws
    )
    ax.axhline(0.5, linewidth=2, color='white', linestyle='--')

    min_xlim = min(abs(data[feature].min()), abs(data[feature].max()))
    ax.set_xlim([-min_xlim, min_xlim])

    if not title:
        title = get_default_title(feature)
    ax.set_title(title)

    handles = ax.get_legend().legendHandles
    ax.legend(handles, ['A', 'B'], title="Winner team")

    avg_ft_values = data.groupby("winner").mean()[feature]
    avg_values_annotation = f"Average values per Winner\nWinner A: {avg_ft_values[0]:.3f}\nWinner B: {avg_ft_values[1]:.3f}"
    ax.annotate(avg_values_annotation,
                xy=(0.5, -0.25),
                xycoords='axes fraction',
                ha='center',
                va="center",
                fontsize=14,
                fontweight="bold",
                fontfamily="Play")
    return ax
