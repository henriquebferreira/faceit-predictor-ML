import matplotlib.pyplot as plt
import seaborn as sns


def set_visualization_defaults():
    sns.set_style("dark")
    # Create an array with the colors you want to use
    faceit_colors = ["#ff5500", "#181818"]
    # Set your custom color palette
    sns.set_palette(sns.color_palette(faceit_colors))
    sns.set_context("notebook")

    plt.rcParams["axes.facecolor"] = "#2D3441"
    plt.rcParams["figure.facecolor"] = "#1f1f1f"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["legend.facecolor"] = "#565c66"
    plt.rcParams["legend.edgecolor"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["patch.linewidth"] = 1
    plt.rcParams["patch.edgecolor"] = "gray"
    plt.rcParams["font.family"] = "Play"


def get_default_title(feature_name):
    feature_name_words = feature_name.split("_")
    if 'dif' == feature_name_words[0]:
        core_feature_name = ' '.join(feature_name_words[1:]).title()
        return 'Winner distribution by Difference of ' + core_feature_name
    else:
        core_feature_name = ' '.join(feature_name_words).title()
        return 'Winner distribution by ' + core_feature_name


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
