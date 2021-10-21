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
