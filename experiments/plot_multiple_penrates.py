from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
import experiments.build_NYC_subway_net as nyc
import src.tnet as tnet
import pickle
from typing import Dict, List
from src.solvers import *
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import fields
import matplotlib.pyplot as plt
import matplotlib.axes as mpl
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 15,                     # IEEE style prefers 8–10 pt
    "pdf.fonttype": 42,   # Important: embed fonts correctly in PDF
    "ps.fonttype": 42,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "text.latex.preamble": r'\usepackage{dsfont}',
    "axes.labelsize": 13,
})


cost_legend = [
    Line2D([0], [0], color='C0', lw=2, label='Average all users'),
    Line2D([0], [0], color='C1', lw=2, label='I-AMoD'),
    Line2D([0], [0], color='C2', lw=2, label='Private')
]

marker_legend = [
    Line2D([0], [0], marker='o', color='gray', lw=0, label=r"$\phi = 0.5$", markersize=8),
    Line2D([0], [0], marker='s', color='gray', lw=0, label=r"$\phi = 1$", markersize=8),
    Line2D([0], [0], marker='^', color='gray', lw=0, label=r"$\phi = 1.5$", markersize=8),
]

def add_line_plot(ax:mpl.Axes, penrate_metrics, marker, color_set):
    x = np.linspace(0.01, 0.99, len(penrate_metrics["totCost"]))
    ax.plot(x, [i * 60 for i in penrate_metrics["totCost"]], marker=marker, color=color_set[0])
    ax.plot(x, [i * 60 for i in penrate_metrics["IAMoDCosts"]], marker=marker, color=color_set[1])
    ax.plot(x, [i * 60 for i in penrate_metrics["privateCosts"]], marker=marker, color=color_set[2])

def plot_penrate_comparison() -> None:
    fig, ax = plt.subplots()
    metrics1 = pd.read_csv("results/penrate/penRate_NYC_2025-07-18_08_29_12/penrate_metrics_099.csv")
    metrics2 = pd.read_csv("results/penrate/penRate_NYC_2025-07-18_08_29_49/penrate_metrics_0.99.csv")
    metrics3 = pd.read_csv("results/penrate/penRate_NYC_2025-07-18_08_30_47/penrate_metrics_0.99.csv")
    # Plot lines with different markers/colors
    add_line_plot(ax, metrics1, marker='o', color_set=['C0', 'C1', 'C2'])
    add_line_plot(ax, metrics2, marker='s', color_set=['C0', 'C1', 'C2'])
    add_line_plot(ax, metrics3, marker='^', color_set=['C0', 'C1', 'C2'])

    # Add the two separate legends
    # first_legend = ax.legend(handles=cost_legend, loc='upper left')
    # ax.add_artist(first_legend)  # Need this to keep both legends
    ax.set_xlabel(r'Penetration Rate ($\mathrm{\%}$)')
    ax.set_ylabel(r'Avgerage User Travel Time ($\mathrm{min}$)')
    cost_legend.extend(marker_legend)
    ax.legend(handles=cost_legend, loc='upper right')
    # plt.title("Effects of Penetration Rate on Costs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/penrate/penRate_comparison.pdf", format="pdf")
    plt.show()

if __name__=="__main__":
    plot_penrate_comparison()