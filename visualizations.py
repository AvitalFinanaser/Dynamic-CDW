import os
import re
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from itertools import cycle
from colorsys import rgb_to_hsv, hsv_to_rgb
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
import matplotlib.lines as mlines
from adjustText import adjust_text


# Help functions


def clean_label(dir_name: str) -> str:
    """
    Produce an upright-math legend label for:
      • Smoothing rules: EXP_{α,prop}-<agg>(threshold)
      • Harsh    rules: t_{prop}-<agg>(threshold)
      • Static   rules: APS→APS(threshold), APS_r→RAPS(threshold), AM→RAMS_{β}(threshold)
    """
    # aggregator renaming
    fam_map = {"AM": "RAMS", "APS_r": "RAPS", "APS": "APS"}

    # parse fields
    s = dir_name.strip("()")
    parts = [p.strip().lstrip("_") for p in s.split(",")]
    fields = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            fields[k] = v

    csf_value = fields.get("CSF", "")
    beta = fields.get("beta")
    threshold = fields.get("threshold", "0.5")  # default if not found

    # Handle the case where CSF contains "_beta="
    if "_beta=" in csf_value:
        csf, beta_value = csf_value.split("_beta=")
        csf = csf.strip()
        beta = beta_value.strip()
    else:
        csf = csf_value

    func = fields.get("Function")

    # 1) Smoothing (exp_alpha=α_prop)
    if func and func.startswith("exp_alpha"):
        m = re.match(r"exp_alpha=([\d.]+)_(events|paragraphs)", func)
        if m:
            α, prop = m.groups()
            exp_lbl = f"EXP_{{{α},{prop}}}"
            if csf == "AM" and beta:
                agg_lbl = f"RAMS_{{{beta}}}"
            else:
                agg_lbl = fam_map.get(csf, csf)
            return (
                    "$\\mathrm{" + exp_lbl + "} - \\mathrm{" + agg_lbl + "}$"
            )

    if func and func.startswith("linear_alpha"):
        m = re.match(r"linear_alpha=([\d.]+)_(events|paragraphs)", func)
        if m:
            α, prop = m.groups()
            exp_lbl = f"LIN_{{{α},{prop}}}"
            if csf == "AM" and beta:
                agg_lbl = f"RAMS_{{{beta}}}"
            else:
                agg_lbl = fam_map.get(csf, csf)
            return (
                    "$\\mathrm{" + exp_lbl + "} - \\mathrm{" + agg_lbl + "}$"
            )

    # 2) Harsh (numeric_property)
    if re.match(r"^\d+_(events|paragraphs)$", csf):
        t, prop = csf.split("_")
        agg_key = next((x for x in parts if x in fam_map), "")
        agg_lbl = fam_map.get(agg_key, agg_key)
        return (
                "$\\mathrm{" + t + "_{" + prop + "}} - \\mathrm{" + agg_lbl + "}$"
        )

    # 3) Static APS / RAPS
    if csf in ("APS", "APS_r"):
        return "$\\mathrm{" + fam_map[csf] + "}$"

    # 4) Static AM with β
    if csf == "AM" and beta:
        return "$\\mathrm{" + fam_map["AM"] + "_{" + beta + "}}$"

    # 5) Fallback
    esc = csf.replace("_", "\\_")
    return "$\\mathrm{" + esc + "}$"


def determine_rule_family(dir_name: str) -> str:
    """
    Determine the rule family: 'smooth', 'harsh', or 'static' based on directory name.
    """
    # parse fields
    s = dir_name.strip("()")
    parts = [p.strip().lstrip("_") for p in s.split(",")]
    fields = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            fields[k] = v

    # Handle the case where CSF contains both AM and beta value (CSF=AM_beta=0.1)
    csf_value = fields.get("CSF", "")
    if "_beta=" in csf_value:
        csf, _ = csf_value.split("_beta=")
        csf = csf.strip()
    else:
        csf = csf_value

    func = fields.get("Function")

    # Determine family:
    # 1. Smoothing rules have Function starting exp or linear
    if func and func.startswith("exp_alpha"):
        return "smooth"

    if func and func.startswith("linear_alpha"):
        return "smooth"

    # 2. Harsh rules have CSF matching digits followed by events/paragraphs
    if re.match(r"^\d+_(events|paragraphs)$", csf):
        return "harsh"

    # 3. Static rules are all others (APS, APS_r, AM with beta)
    return "static"


# Filtering rules

# Snake plot

def plot_filtered_snake(agent_type, rule_filter_list=None, results_dir="results/all_rules",
                        figures_dir="results/figures"):
    """
    Create a snake plot showing the evolution of stability and satisfaction for filtered rules.
    """
    # Create output directory if it doesn't exist
    agent_path = os.path.join(results_dir, agent_type)
    out_path = os.path.join(figures_dir, agent_type)
    os.makedirs(out_path, exist_ok=True)

    # Define family base colors
    family_base_colors = {
        "smooth": "#4472C4",  # Blue
        "harsh": "#ED7D31",  # Orange
        "static": "#70AD47"  # Green
    }

    # Use dashed line style for all rules
    line_style = "--"

    # Get all rule directories
    all_rules = sorted(os.listdir(agent_path))

    # Filter rules if a filter list is provided
    if rule_filter_list is not None:
        seen = set()
        rule_dirs = [x for x in rule_filter_list if not (x in seen or seen.add(x))]
    else:
        rule_dirs = all_rules

    # Categorize rules by family
    rules_by_family = {
        "smooth": [],
        "harsh": [],
        "static": []
    }

    for rule_dir in rule_dirs:
        # Check if CSV exists
        csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: CSV file not found for rule {rule_dir}")
            continue

        # Determine rule family and add to appropriate list
        family = determine_rule_family(rule_dir)
        rules_by_family[family].append(rule_dir)

    # Create a single plot for all filtered rules
    fig, ax = plt.subplots(figsize=(12, 8))

    # Store rule colors for legend
    rule_colors = {}
    processed_rules = set()

    # Plot each family with different shades of the family color
    for family, rules in rules_by_family.items():
        if not rules:  # Skip empty families
            continue

        # Get base color for this family
        base_color = family_base_colors[family]

        # Create color variants (shades) for each rule in this family
        base_rgb = mcolors.to_rgb(base_color)
        base_hsv = rgb_to_hsv(base_rgb[0], base_rgb[1], base_rgb[2])

        # Generate color variants
        family_colors = []
        num_rules = len(rules)

        for i in range(num_rules):
            # Adjust saturation and value to create variants
            h = base_hsv[0]
            s = max(0.3, min(1.0, base_hsv[1] * (0.7 + 0.6 * i / max(1, num_rules - 1))))
            v = max(0.4, min(0.9, base_hsv[2] * (1.2 - 0.4 * i / max(1, num_rules - 1))))

            # Convert back to RGB
            rgb = hsv_to_rgb(h, s, v)
            family_colors.append(mcolors.rgb2hex(rgb))

        # Plot each rule with its color variant but same line style
        for i, rule_dir in enumerate(rules):
            # Skip if we've already processed this rule
            if rule_dir in processed_rules:
                continue

            processed_rules.add(rule_dir)

            # Read data
            csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
            df = pd.read_csv(csv_path)

            # Get stability and satisfaction
            x, y = df["Stability"], df["Sat_Sum_Normalized"]

            # Use rule-specific color shade
            rule_color = family_colors[i]

            # Store color for legend
            label = clean_label(rule_dir)
            rule_colors[label] = rule_color

            # Plot the snake path
            line = ax.plot(
                x, y,
                color=rule_color,
                linestyle=line_style,
                linewidth=1.5,
                label=label
            )

            # Mark the start point
            ax.scatter(
                x.iloc[0], y.iloc[0],
                marker="o", s=80, color=rule_color,
                edgecolor="black", zorder=5
            )

            # Add arrows every 10 events
            for j in range(0, len(df) - 1, 10):
                if j + 1 < len(df):
                    ax.annotate(
                        "",
                        xy=(x.iloc[j + 1], y.iloc[j + 1]),
                        xytext=(x.iloc[j], y.iloc[j]),
                        arrowprops=dict(arrowstyle="->", color=rule_color, lw=1.2),
                        zorder=4
                    )

            # Mark the end point with a star
            if len(x) > 1:
                ax.scatter(
                    x.iloc[-1], y.iloc[-1],
                    marker="*", s=120, color=rule_color,
                    edgecolor="black", zorder=5
                )

    # Set plot attributes
    ax.set_xlabel("Stability", fontsize=14)
    ax.set_ylabel("Satisfaction", fontsize=14)

    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Create the legend - use custom Line2D objects for complete control
    legend_handles = []
    legend_labels = []

    # Add family section headers and rule entries
    for family in rules_by_family.keys():
        if not rules_by_family[family]:
            continue

        # Add family title with its color
        legend_handles.append(Line2D([], [], color='none'))
        legend_labels.append(f"{family.capitalize()} Rules:")

        # Add each rule with a line sample
        for rule_dir in rules_by_family[family]:
            label = clean_label(rule_dir)
            if label in rule_colors:
                color = rule_colors[label]
                # Create a line with the exact appearance we want
                line = Line2D([], [], color=color, linestyle=line_style, linewidth=1.5)
                legend_handles.append(line)
                legend_labels.append(label)

    # Add the legend with a box - inside the plot area
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fontsize=9,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.4
    )

    # Set color for family headers
    for i, text in enumerate(legend.get_texts()):
        text_content = text.get_text()
        if text_content.endswith("Rules:"):
            family = text_content.split()[0].lower()
            if family in family_base_colors:
                text.set_color(family_base_colors[family])
                text.set_weight('bold')

    # Adjust layout and save
    fig.savefig(
        os.path.join(out_path, f"snake_for_{agent_type}_agents_filtered.png"),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close(fig)

    print(f"Snake plot for {agent_type} agents saved.")


def plot_filtered_snake2(agent_type="unstructured", rule_filter_list=None, results_dir="results/all_rules",
                         figures_dir="results/figures", prefix=""):
    """
    Create a snake plot showing the evolution of stability and satisfaction for filtered rules.
    """
    # Set publication-quality settings
    # plt.rcParams['mathtext.fontset'] = 'custom'  # Add this
    # plt.rcParams['mathtext.rm'] = 'Times New Roman'
    # plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    # plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8

    # Create output directory if it doesn't exist
    agent_path = results_dir
    out_path = figures_dir
    os.makedirs(out_path, exist_ok=True)

    # Define family base colors with improved orange
    family_base_colors = {
        "smooth": "#4472C4",  # Blue
        "harsh": "#E67E22",  # Improved orange
        "static": "#70AD47"  # Green
    }

    # Define custom color palettes for each family to avoid unpleasant shades
    custom_colors = {
        "smooth": ["#4472C4", "#6A8CC8", "#8FA6CD", "#B4C0D2"],  # Blues
        "harsh": ["#E67E22", "#EB9950", "#F0B47D", "#F5CFAA"],  # Oranges
        "static": ["#70AD47", "#8ABD68", "#A5CD89", "#C0DDAA"]  # Greens
    }

    # Use dashed line style for all rules
    line_style = "--"

    # Get all rule directories
    all_rules = sorted(os.listdir(agent_path))

    # Filter rules if a filter list is provided
    if rule_filter_list is not None:
        seen = set()
        rule_dirs = [x for x in rule_filter_list if not (x in seen or seen.add(x))]
    else:
        rule_dirs = all_rules

    # Categorize rules by family
    rules_by_family = {
        "smooth": [],
        "harsh": [],
        "static": []
    }

    for rule_dir in rule_dirs:
        # Check if CSV exists
        file_name = f"{prefix}{rule_dir}.csv"
        csv_path = os.path.join(agent_path, rule_dir, file_name)
        if not os.path.isfile(csv_path):
            print(f"Warning: CSV file not found for rule {rule_dir}")
            continue

        # Determine rule family and add to appropriate list
        family = determine_rule_family(rule_dir)
        rules_by_family[family].append(rule_dir)

    # Create a single plot for all filtered rules with high-quality settings
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    # Store rule colors for legend
    rule_colors = {}
    processed_rules = set()

    # Plot each family with custom colors
    for family, rules in rules_by_family.items():
        if not rules:  # Skip empty families
            continue

        # Use custom color palette for this family
        family_palette = custom_colors[family]

        # If we need more colors than in the palette, extend it
        if len(rules) > len(family_palette):
            # Get base color and create more variants
            base_color = family_base_colors[family]
            base_rgb = mcolors.to_rgb(base_color)
            base_hsv = rgb_to_hsv(base_rgb[0], base_rgb[1], base_rgb[2])

            # Generate additional colors
            extra_colors = []
            for i in range(len(rules) - len(family_palette)):
                # Adjust saturation and value to create variants
                h = base_hsv[0]
                s = max(0.3, min(0.9, base_hsv[1] * (0.8 + 0.4 * i / max(1, len(rules) - len(family_palette) - 1))))
                v = max(0.4, min(0.9, base_hsv[2] * (1.1 - 0.3 * i / max(1, len(rules) - len(family_palette) - 1))))

                # Convert back to RGB
                rgb = hsv_to_rgb(h, s, v)
                extra_colors.append(mcolors.rgb2hex(rgb))

            # Combine predefined palette with generated colors
            family_palette = family_palette + extra_colors

        # Plot each rule with its color variant but same line style
        for i, rule_dir in enumerate(rules):
            # Skip if we've already processed this rule
            if rule_dir in processed_rules:
                continue

            processed_rules.add(rule_dir)

            # Read data
            file_name = f"{prefix}{rule_dir}.csv"
            csv_path = os.path.join(agent_path, rule_dir, file_name)
            df = pd.read_csv(csv_path)

            # Get stability and satisfaction
            x, y = df["Stability"], df["Sat_Sum_Normalized"]

            # Use the color from the palette
            rule_color = family_palette[i % len(family_palette)]

            # Store color for legend
            label = clean_label(rule_dir)
            rule_colors[label] = rule_color

            # Plot the snake path with enhanced line width
            line = ax.plot(
                x, y,
                color=rule_color,
                linestyle=line_style,
                linewidth=2.0,
                label=label
            )

            # Mark the start point with enhanced size
            ax.scatter(
                x.iloc[0], y.iloc[0],
                marker="o", s=100, color=rule_color,
                edgecolor="black", linewidth=1.0, zorder=5
            )

            # Add arrows every 10 events
            for j in range(0, len(df) - 1, 10):
                if j + 1 < len(df):
                    ax.annotate(
                        "",
                        xy=(x.iloc[j + 1], y.iloc[j + 1]),
                        xytext=(x.iloc[j], y.iloc[j]),
                        arrowprops=dict(arrowstyle="->", color=rule_color, lw=2),
                        zorder=4
                    )

            # Mark the end point with a star - enhanced size
            if len(x) > 1:
                ax.scatter(
                    x.iloc[-1], y.iloc[-1],
                    marker="*", s=150, color=rule_color,
                    edgecolor="black", linewidth=1.0, zorder=5
                )

    # Set plot attributes with enhanced fonts
    # ax.set_title("Evolution of Stability and Satisfaction Under Different Rules (Unstructuredd)",
    #              fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Stability", fontsize=20, fontweight='bold')
    ax.set_ylabel("Satisfaction", fontsize=20, fontweight='bold')

    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Make tick labels more visible
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)

    # Add a box around the plot with enhanced line width
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Set grid for better readability
    ax.grid(True, linestyle=':', alpha=0.3)

    # Create the legend - use custom Line2D objects for complete control
    legend_handles = []
    legend_labels = []

    # Add family section headers and rule entries
    for family in rules_by_family.keys():
        if not rules_by_family[family]:
            continue

        # Add family title with its color
        legend_handles.append(Line2D([], [], color='none'))
        legend_labels.append(f"{family.capitalize()} Rules:")

        # Add each rule with a line sample
        for rule_dir in rules_by_family[family]:
            label = clean_label(rule_dir)
            if label in rule_colors:
                color = rule_colors[label]
                # Create a line with the exact appearance we want
                line = Line2D([], [], color=color, linestyle=line_style, linewidth=2.0)
                legend_handles.append(line)
                label = str(label)
                legend_labels.append(label)

    # Add the legend with a box - inside or outside the plot
    if len(rule_filter_list) <= 6:
        loc = 'upper right'
        bbox_to_anchor = (0.99, 0.99)
        prop = {'size': 13, 'weight': 'bold'}

    else:
        loc = 'upper left'
        bbox_to_anchor = (1.02, 1.00)
        prop = {'family': 'Times New Roman', 'size': 10, 'weight': 'bold'}

    legend = ax.legend(
        legend_handles,
        legend_labels,
        fontsize=13,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        labelspacing=0.4,
        handlelength=2.0,
        handletextpad=0.7,
        borderpad=0.5,
        title_fontsize=14,
        prop=prop
    )

    # Enhance legend border width
    # legend.get_frame().set_linewidth(1.5)

    # Set color for family headers and make them bold
    for i, text in enumerate(legend.get_texts()):
        text_content = text.get_text()
        if text_content.endswith("Rules:"):
            family = text_content.split()[0].lower()
            if family in family_base_colors:
                text.set_color(family_base_colors[family])
                text.set_weight('bold')
                # text.set_size(12)  # Slightly larger for headers

    # Save with high resolution
    fig.savefig(
        os.path.join(out_path, f"snake_for_{agent_type}_agents_filtered_new.png"),
        dpi=600,  # Higher DPI for print quality
        bbox_inches='tight',
        pad_inches=0.2,
        format='png'
    )

    plt.close(fig)

    print(f"Snake plot for {agent_type} agents saved")


def plot_filtered_snake_all_rules(agent_type, rule_filter_list=None,
                                  results_dir="results/all_rules",
                                  figures_dir="results/figures", prefix=""):
    """
    Create a snake plot with each rule having a unique color and family using different line styles.
    """
    # Set publication-quality settings
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8

    # Create output directory if it doesn't exist
    agent_path = results_dir
    out_path = figures_dir
    os.makedirs(out_path, exist_ok=True)

    # Define different line styles for each family
    family_line_styles = {
        "smooth": "--",  # Dashed for smooth rules
        "harsh": "-.",  # Dash-dot for harsh rules
        "static": "-"  # Solid for static rules
    }

    # Generate a large color palette using a colormap for individual rules
    # You can change 'tab20' to other colormaps like 'rainbow', 'viridis', etc.
    colormap = plt.cm.get_cmap('tab20', 20)
    color_palette = [colormap(i) for i in range(20)]

    # If we need more colors, we can generate more using HSV
    if rule_filter_list and len(rule_filter_list) > 20:
        additional_colors = []
        for i in range(20, len(rule_filter_list)):
            hue = (i * 7) % 360  # Use a prime number multiplier for good distribution
            color = plt.cm.hsv(hue / 360.0)
            additional_colors.append(color)
        color_palette.extend(additional_colors)

    # Get all rule directories
    all_rules = sorted(os.listdir(agent_path))

    # Filter rules if a filter list is provided
    if rule_filter_list is not None:
        seen = set()
        rule_dirs = [x for x in rule_filter_list if not (x in seen or seen.add(x))]
    else:
        rule_dirs = all_rules

    # Categorize rules by family
    rules_by_family = {
        "smooth": [],
        "harsh": [],
        "static": []
    }

    for rule_dir in rule_dirs:
        # Check if CSV exists
        file_name = f"{prefix}{rule_dir}.csv"
        csv_path = os.path.join(agent_path, rule_dir, file_name)
        if not os.path.isfile(csv_path):
            print(f"Warning: CSV file not found for rule {rule_dir}")
            continue

        # Determine rule family and add to appropriate list
        family = determine_rule_family(rule_dir)
        rules_by_family[family].append(rule_dir)

    # Create a single plot for all filtered rules with high-quality settings
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Store rule colors for legend
    rule_colors = {}
    processed_rules = set()

    # Assign colors to all rules in order
    color_index = 0
    for family in ["smooth", "harsh", "static"]:
        for rule_dir in rules_by_family.get(family, []):
            rule_colors[rule_dir] = color_palette[color_index % len(color_palette)]
            color_index += 1

    # Plot each family with individual colors but same line style per family
    for family, rules in rules_by_family.items():
        if not rules:  # Skip empty families
            continue

        # Get line style for this family
        line_style = family_line_styles[family]

        # Plot each rule with its unique color
        for rule_dir in rules:
            # Skip if we've already processed this rule
            if rule_dir in processed_rules:
                continue

            processed_rules.add(rule_dir)

            # Read data
            file_name = f"{prefix}{rule_dir}.csv"
            csv_path = os.path.join(agent_path, rule_dir, file_name)
            df = pd.read_csv(csv_path)

            # Get stability and satisfaction
            x, y = df["Stability"], df["Sat_Sum_Normalized"]

            # Use the color assigned to this rule
            rule_color = rule_colors[rule_dir]

            # Get label
            label = clean_label(rule_dir)

            # Plot the snake path
            line = ax.plot(
                x, y,
                color=rule_color,
                linestyle=line_style,
                linewidth=2.0,
                label=label
            )

            # Mark the start point
            ax.scatter(
                x.iloc[0], y.iloc[0],
                marker="o", s=100, color=rule_color,
                edgecolor="black", linewidth=1.0, zorder=5
            )

            # Add arrows every 10 events
            for j in range(0, len(df) - 1, 10):
                if j + 1 < len(df):
                    ax.annotate(
                        "",
                        xy=(x.iloc[j + 1], y.iloc[j + 1]),
                        xytext=(x.iloc[j], y.iloc[j]),
                        arrowprops=dict(arrowstyle="->", color=rule_color, lw=2),
                        zorder=4
                    )

            # Mark the end point with a star
            if len(x) > 1:
                ax.scatter(
                    x.iloc[-1], y.iloc[-1],
                    marker="*", s=150, color=rule_color,
                    edgecolor="black", linewidth=1.0, zorder=5
                )

    # Set plot attributes
    ax.set_xlabel("Stability", fontsize=20, fontweight='bold')
    ax.set_ylabel("Satisfaction", fontsize=20, fontweight='bold')

    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Make tick labels more visible
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Set grid for better readability
    ax.grid(True, linestyle=':', alpha=0.3)

    # Create legend with family headers
    legend_handles = []
    legend_labels = []

    # Add family section headers and rule entries
    for family in ["smooth", "harsh", "static"]:
        if not rules_by_family[family]:
            continue

        # Add family title with appropriate line style
        family_title_line = Line2D([], [], color='gray', linestyle=family_line_styles[family], linewidth=2)
        legend_handles.append(family_title_line)
        legend_labels.append(f"{family.capitalize()} Rules (line: {family_line_styles[family]}):")

        # Add each rule with its unique color
        for rule_dir in rules_by_family[family]:
            label = clean_label(rule_dir)
            color = rule_colors[rule_dir]
            # Create a line with its unique color and family line style
            line = Line2D([], [], color=color, linestyle=family_line_styles[family], linewidth=2.0)
            legend_handles.append(line)
            legend_labels.append(str(label))

    # Position legend based on number of rules
    if rule_filter_list and len(rule_filter_list) < 6:
        loc = 'upper right'
        bbox_to_anchor = (0.99, 0.99)
        prop = {'family': 'Times New Roman', 'size': 13, 'weight': 'bold'}
    else:
        loc = 'upper left'
        bbox_to_anchor = (1.02, 1.00)
        prop = {'family': 'Times New Roman', 'size': 10, 'weight': 'bold'}

    legend = ax.legend(
        legend_handles,
        legend_labels,
        fontsize=13,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        labelspacing=0.4,
        handlelength=2.0,
        handletextpad=0.7,
        borderpad=0.5,
        title_fontsize=14,
        prop=prop
    )

    # Make family headers stand out
    for i, text in enumerate(legend.get_texts()):
        text_content = text.get_text()
        if "Rules (line:" in text_content:
            text.set_color('black')
            text.set_weight('bold')

    # Save the figure
    fig.savefig(
        os.path.join(out_path, f"snake_for_{agent_type}_agents_all_rules.png"),
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.2,
        format='png'
    )

    plt.close(fig)

    print(f"Snake plot for {agent_type} agents (all rules) saved")


def plot_snake_for_agent_by_family(agent_type: str,
                                   results_dir: str = "results/all_rules",
                                   figures_dir: str = "results/figures"):
    agent_path = os.path.join(results_dir, agent_type)
    out_path = os.path.join(figures_dir, agent_type)
    os.makedirs(out_path, exist_ok=True)

    all_rules = sorted(os.listdir(agent_path))

    # Categorize rules by family
    rules_by_family = {
        "smooth": [],
        "harsh": [],
        "static": []
    }

    for rule_dir in all_rules:
        # Check if CSV exists
        csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
        if not os.path.isfile(csv_path):
            continue

        # Determine rule family and add to appropriate list
        family = determine_rule_family(rule_dir)
        rules_by_family[family].append(rule_dir)

    # Create a plot for each family
    family_titles = {
        "smooth": "Smoothing Rules",
        "harsh": "Harsh Rules",
        "static": "Static Rules"
    }

    for family, rules in rules_by_family.items():
        if not rules:  # Skip empty families
            continue

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        color_cycle = cycle(colors)
        linestyle_cycle = cycle(linestyles)
        marker_cycle = cycle(markers)

        fig, ax = plt.subplots(figsize=(12, 6))

        for rule_dir in rules:
            csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
            df = pd.read_csv(csv_path)
            x, y = df["Stability"], df["Sat_Sum_Normalized"]

            color = next(color_cycle)
            ls = next(linestyle_cycle)
            m = next(marker_cycle)

            ax.plot(
                x, y,
                color=color,
                linestyle=ls,
                marker=m,
                markevery=10,
                linewidth=1.5,
                markersize=5,
                label=clean_label(rule_dir)
            )

            # start marker
            ax.scatter(
                x.iloc[0], y.iloc[0],
                marker="o", s=80, color=color,
                edgecolor="black", zorder=5
            )
            # end arrow + star
            if len(x) > 1:
                ax.annotate(
                    "",
                    xy=(x.iloc[-1], y.iloc[-1]),
                    xytext=(x.iloc[-2], y.iloc[-2]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    zorder=5
                )
                ax.scatter(
                    x.iloc[-1], y.iloc[-1],
                    marker="*", s=120, color=color,
                    edgecolor="black", zorder=5
                )

        # Set plot attributes
        ax.set_xlabel("Stability", fontsize=14)
        ax.set_ylabel("Satisfaction", fontsize=14)
        ax.set_title(f"{agent_type.title()} - {family_titles[family]}", fontsize=16, pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize="small"
        )

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_xticks(np.arange(0, 1.1, 0.1))
        # ax.set_yticks(np.arange(0, 1.1, 0.1))

        fig.subplots_adjust(right=0.75)
        fig.savefig(
            os.path.join(out_path, f"{agent_type}_{family}_snake_plot.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1
        )
        plt.close(fig)

    # Create a combined plot with all rule families
    generate_combined_plot(agent_type, rules_by_family, agent_path, out_path)

    # Create the grid plot from all generated plots
    generate_grid_plot(agent_type, results_dir, figures_dir)


# def plot_filtered_snake2(agent_type, rule_filter_list=None, results_dir="results/all_rules",
#                         figures_dir="results/figures"):
#     """
#     Create a snake plot showing the evolution of stability and satisfaction for filtered rules.
#
#     Parameters:
#     -----------
#     agent_type : str
#         The type of agent (e.g., 'selfish', 'cooperative')
#     rule_filter_list : list, optional
#         List of rule directory names to include. If None, all rules are included.
#     results_dir : str
#         Directory containing the results for all rules
#     figures_dir : str
#         Directory to save the figure
#     """
#     import os
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import matplotlib.colors as mcolors
#     from itertools import cycle
#     from colorsys import rgb_to_hsv, hsv_to_rgb  # Use colorsys instead of mcolors for rgb_to_hsv
#
#     # Create output directory if it doesn't exist
#     agent_path = os.path.join(results_dir, agent_type)
#     out_path = os.path.join(figures_dir, agent_type)
#     os.makedirs(out_path, exist_ok=True)
#
#     # Define family base colors
#     family_base_colors = {
#         "smooth": "#4472C4",  # Blue
#         "harsh": "#ED7D31",  # Orange
#         "static": "#70AD47"  # Green
#     }
#
#     # Use the same line style for all rules - changed to dashed as requested
#     line_style = "--"  # Dashed line for all rules
#
#     # Get all rule directories
#     all_rules = sorted(os.listdir(agent_path))
#
#     # Filter rules if a filter list is provided
#     if rule_filter_list is not None:
#         rule_dirs = [rule for rule in all_rules if rule in rule_filter_list]
#     else:
#         rule_dirs = all_rules
#
#     # Categorize rules by family
#     rules_by_family = {
#         "smooth": [],
#         "harsh": [],
#         "static": []
#     }
#
#     for rule_dir in rule_dirs:
#         # Check if CSV exists
#         csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
#         if not os.path.isfile(csv_path):
#             continue
#
#         # Determine rule family and add to appropriate list
#         family = determine_rule_family(rule_dir)
#         rules_by_family[family].append(rule_dir)
#
#     # Create a single plot for all filtered rules
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Track rules for the legend
#     legend_elements = []
#
#     # Plot each family with different shades of the family color
#     for family, rules in rules_by_family.items():
#         if not rules:  # Skip empty families
#             continue
#
#         # Get base color for this family
#         base_color = family_base_colors[family]
#
#         # Create color variants (shades) for each rule in this family
#         # Convert hex to RGB, then to HSV for easier manipulation
#         base_rgb = mcolors.to_rgb(base_color)  # Returns tuple (r, g, b)
#         base_hsv = rgb_to_hsv(base_rgb[0], base_rgb[1], base_rgb[2])  # Use colorsys directly
#
#         # Generate color variants
#         rule_colors = []
#         num_rules = len(rules)
#
#         for i in range(num_rules):
#             # Adjust saturation and value to create variants
#             # Keep hue the same to maintain family color identity
#             h = base_hsv[0]
#             s = max(0.3, min(1.0, base_hsv[1] * (0.7 + 0.6 * i / max(1, num_rules - 1))))
#             v = max(0.4, min(0.9, base_hsv[2] * (1.2 - 0.4 * i / max(1, num_rules - 1))))
#
#             # Convert back to RGB
#             rgb = hsv_to_rgb(h, s, v)  # Use colorsys
#             rule_colors.append(mcolors.rgb2hex(rgb))
#
#         # Plot each rule with its color variant but same line style
#         for i, rule_dir in enumerate(rules):
#             # Read data
#             csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
#             df = pd.read_csv(csv_path)
#
#             # Get stability and satisfaction
#             x, y = df["Stability"], df["Sat_Sum_Normalized"]
#
#             # Use rule-specific color shade
#             rule_color = rule_colors[i]
#
#             # Plot the snake path
#             line = ax.plot(
#                 x, y,
#                 color=rule_color,
#                 linestyle=line_style,  # Same dashed line style for all
#                 linewidth=1.5,
#                 label=f"{clean_label(rule_dir)}"
#             )
#
#             # Add to legend elements
#             legend_elements.append((line[0], rule_color))
#
#             # Mark the start point
#             ax.scatter(
#                 x.iloc[0], y.iloc[0],
#                 marker="o", s=80, color=rule_color,
#                 edgecolor="black", zorder=5
#             )
#
#             # Add arrows every 10 events
#             for j in range(0, len(df) - 1, 10):  # Changed i to j to avoid variable collision
#                 if j + 1 < len(df):
#                     ax.annotate(
#                         "",
#                         xy=(x.iloc[j + 1], y.iloc[j + 1]),
#                         xytext=(x.iloc[j], y.iloc[j]),
#                         arrowprops=dict(arrowstyle="->", color=rule_color, lw=1.2),
#                         zorder=4
#                     )
#
#             # Mark the end point with a star
#             if len(x) > 1:
#                 ax.scatter(
#                     x.iloc[-1], y.iloc[-1],
#                     marker="*", s=120, color=rule_color,
#                     edgecolor="black", zorder=5
#                 )
#
#     # Set plot attributes
#     ax.set_xlabel("Stability", fontsize=14)
#     ax.set_ylabel("Satisfaction", fontsize=14)
#
#     # Set axis limits and ticks
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_xticks(np.arange(0, 1.1, 0.1))
#     ax.set_yticks(np.arange(0, 1.1, 0.1))
#
#     # Add a box around the plot
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#
#     # Group legend elements by family
#     legend_by_family = {family: [] for family in rules_by_family.keys()}
#
#     for line, color in legend_elements:
#         # Get rule label
#         label = line.get_label()
#
#         # Determine which family this rule belongs to
#         for family in rules_by_family.keys():
#             if any(clean_label(rule_dir) == label for rule_dir in rules_by_family[family]):
#                 legend_by_family[family].append((label, color))
#                 break
#
#     # Create a structured legend with colored squares instead of dashes
#     legend_text = ""
#     y_pos = 0.99  # Start at the top
#
#     # Create the outer box for the entire legend
#     total_items = sum(len(items) for items in legend_by_family.values()) + len(
#         [f for f in legend_by_family if legend_by_family[f]])
#     box_height = 0.03 * total_items + 0.1  # Height for items plus some padding
#
#     legend_box = plt.Rectangle((1.01, 0.99 - box_height), 0.3, box_height,
#                                transform=ax.transAxes, facecolor='white',
#                                edgecolor='gray', alpha=0.9, zorder=-1)
#     ax.add_patch(legend_box)
#
#     # Add each family and its rules to the legend
#     for family in legend_by_family.keys():
#         if legend_by_family[family]:
#             # Add family title in the family's base color
#             family_title = f"{family.capitalize()} Rules:"
#             plt.text(1.02, y_pos, family_title, transform=ax.transAxes,
#                      fontsize=10, va='top', ha='left', fontweight='bold',
#                      color=family_base_colors[family])
#
#             # Calculate vertical position for rules list
#             y_pos -= 0.03  # Move down slightly for the rules
#
#             # Add each rule with a colored square (no dash)
#             for label, color in legend_by_family[family]:
#                 # Add colored square
#                 square_size = 0.01
#                 rect = plt.Rectangle((1.04, y_pos - 0.005), square_size, square_size,
#                                      transform=ax.transAxes, facecolor=color,
#                                      edgecolor='black', alpha=1.0)
#                 ax.add_patch(rect)
#
#                 # Add rule label next to the square
#                 plt.text(1.04 + square_size + 0.01, y_pos - 0.01, label, transform=ax.transAxes,
#                          fontsize=9, va='center', ha='left')
#
#                 # Move down for next rule
#                 y_pos -= 0.03
#
#             # Add spacing between families
#             y_pos -= 0.02
#
#     # Adjust layout and save
#     fig.subplots_adjust(right=0.75)
#     fig.savefig(
#         os.path.join(out_path, f"snake_for_{agent_type}_agents_filtered.png"),
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0.1
#     )
#     plt.close(fig)
#
#     print(f"Snake plot for {agent_type} agents saved.")


def generate_combined_plot(agent_type, rules_by_family, agent_path, out_path):
    """Generate a combined plot with all rule families"""
    combined_rules = []
    for family in rules_by_family:
        combined_rules.extend(rules_by_family[family])

    if not combined_rules:  # Skip if no rules
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    color_cycle = cycle(colors)
    linestyle_cycle = cycle(linestyles)
    marker_cycle = cycle(markers)

    fig, ax = plt.subplots(figsize=(12, 6))

    for rule_dir in combined_rules:
        csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
        df = pd.read_csv(csv_path)
        x, y = df["Stability"], df["Sat_Sum_Normalized"]

        color = next(color_cycle)
        ls = next(linestyle_cycle)
        m = next(marker_cycle)

        ax.plot(
            x, y,
            color=color,
            linestyle=ls,
            marker=m,
            markevery=10,
            linewidth=1.5,
            markersize=5,
            label=clean_label(rule_dir)
        )

        # start marker
        ax.scatter(
            x.iloc[0], y.iloc[0],
            marker="o", s=80, color=color,
            edgecolor="black", zorder=5
        )
        # end arrow + star
        if len(x) > 1:
            ax.annotate(
                "",
                xy=(x.iloc[-1], y.iloc[-1]),
                xytext=(x.iloc[-2], y.iloc[-2]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=5
            )
            ax.scatter(
                x.iloc[-1], y.iloc[-1],
                marker="*", s=120, color=color,
                edgecolor="black", zorder=5
            )

    # Set plot attributes
    ax.set_xlabel("Stability", fontsize=14)
    ax.set_ylabel("Satisfaction", fontsize=14)
    ax.set_title(agent_type.title(), fontsize=16, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize="small"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    fig.subplots_adjust(right=0.75)
    fig.savefig(
        os.path.join(out_path, f"{agent_type}_all_snake_plot.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig)


def generate_grid_plot(agent_type: str,
                       results_dir: str = "results/all_rules",
                       figures_dir: str = "results/figures"):
    """
    Create a grid layout of the 4 plots for a given agent (3 family plots + combined plot).
    This function loads the previously generated plot images and arranges them in a grid.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.image import imread

    # Define paths
    agent_path = os.path.join(results_dir, agent_type)
    out_path = os.path.join(figures_dir, agent_type)

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])

    # Family titles for labels
    family_titles = {
        "smooth": "Smooth Rules",
        "harsh": "Harsh Rules",
        "static": "Static Rules",
        "all": "All Rules"
    }

    # Load the 4 plots (3 family plots + combined plot)
    plots = []
    for i, family in enumerate(["smooth", "harsh", "static"]):
        plot_path = os.path.join(out_path, f"{agent_type}_{family}_snake_plot.png")
        if os.path.exists(plot_path):
            plots.append((i, family, plot_path))

    # Add the combined plot
    combined_path = os.path.join(out_path, f"{agent_type}_all_snake_plot.png")
    if os.path.exists(combined_path):
        plots.append((3, "all", combined_path))

    # Add each plot to the grid
    for idx, family, plot_path in plots:
        ax = fig.add_subplot(gs[0, idx])

        # Load and display the image
        try:
            img = imread(plot_path)
            ax.imshow(img)
            ax.set_title(family_titles[family], fontsize=16)
            ax.axis('off')  # Hide axes
        except Exception as e:
            print(f"Error loading {plot_path}: {e}")
            ax.text(0.5, 0.5, f"No {family} plot available",
                    ha='center', va='center', fontsize=14)

    # Add overall title
    fig.suptitle(f"{agent_type.title()} - Rule Comparison", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the suptitle

    # Save grid figure
    grid_path = os.path.join(out_path, f"{agent_type}_grid_plot.png")
    fig.savefig(grid_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Created grid plot for {agent_type} at {grid_path}")
    return grid_path


# Collecting results data

def collect_llm_data(config, results_dir="results/all_rules/LLM"):
    """Collect data from all LLM agent CSV files and organize by rule family."""
    agent_path = results_dir

    if not os.path.exists(agent_path):
        print(f"Error: Directory not found: {agent_path}")
        return None

    # Get all subdirectories (one for each rule)
    rule_dirs = [d for d in os.listdir(agent_path)
                 if os.path.isdir(os.path.join(agent_path, d))]

    print(f"Found {len(rule_dirs)} rule directories")

    if not rule_dirs:
        print("No rule directories found!")
        return None

    # Categorize files by rule family
    data_by_family = {
        "smooth": [],
        "harsh": [],
        "static": [],
        "all": []  # Combined dataset
    }

    for rule_dir in rule_dirs:
        rule_path = os.path.join(agent_path, rule_dir)
        csv_file = os.path.join(rule_path, f"{config}_llm_{rule_dir}.csv")
        print(csv_file)

        if not os.path.exists(csv_file):
            print(f"Warning: Missing CSV in {rule_dir}")
            continue

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check for required columns
            if "Stability" not in df.columns or "Sat_Sum_Normalized" not in df.columns:
                print(f"Warning: Required columns missing in {csv_file}")
                print(f"Available columns: {df.columns.tolist()}")
                continue

            # Add a new column with the rule name (directory name)
            df["Rule"] = rule_dir

            # Add an Event column if it doesn't exist (use index)
            if "Event" not in df.columns:
                df["Event"] = df.index

            # Determine rule family
            family = determine_rule_family(rule_dir)

            # Append to the appropriate family and to 'all'
            data_by_family[family].append(df)
            data_by_family["all"].append(df)

            print(f"Processed {rule_dir} as {family} rule")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    print(
        f"Found {len(rule_dirs)} rules. Smooth: {len(data_by_family['smooth'])}, Harsh: {len(data_by_family['harsh'])}, Static: {len(data_by_family['static'])}")

    # Combine dataframes within each family
    for family in data_by_family:
        if data_by_family[family]:
            data_by_family[family] = pd.concat(data_by_family[family], ignore_index=True)
            print(f"{family} family: {len(data_by_family[family])} rows")
        else:
            data_by_family[family] = None
            print(f"{family} family: No data")

    return data_by_family


def collect_non_llm_data(family, results_dir="results/all_rules"):
    """Collect data from all LLM agent CSV files and organize by rule family."""
    agent_path = os.path.join(results_dir, family)

    if not os.path.exists(agent_path):
        print(f"Error: Directory not found: {agent_path}")
        return None

    # Get all subdirectories (one for each rule)
    rule_dirs = [d for d in os.listdir(agent_path)
                 if os.path.isdir(os.path.join(agent_path, d))]

    if not rule_dirs:
        print("No rule directories found!")
        return None

    # Categorize files by rule family
    data_by_family = {
        "smooth": [],
        "harsh": [],
        "static": [],
        "all": []  # Combined dataset
    }

    for rule_dir in rule_dirs:
        rule_path = os.path.join(agent_path, rule_dir)
        csv_file = os.path.join(rule_path, f"{rule_dir}.csv")
        print(csv_file)

        if not os.path.exists(csv_file):
            print(f"Warning: Missing CSV in {rule_dir}")
            continue

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check for required columns
            if "Stability" not in df.columns or "Sat_Sum_Normalized" not in df.columns:
                print(f"Warning: Required columns missing in {csv_file}")
                print(f"Available columns: {df.columns.tolist()}")
                continue

            # Add a new column with the rule name (directory name)
            df["Rule"] = rule_dir

            # Add an Event column if it doesn't exist (use index)
            if "Event" not in df.columns:
                df["Event"] = df.index

            # Determine rule family
            family = determine_rule_family(rule_dir)

            # Append to the appropriate family and to 'all'
            data_by_family[family].append(df)
            data_by_family["all"].append(df)

            print(f"Processed {rule_dir} as {family} rule")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    print(
        f"Found {len(rule_dirs)} rules. Smooth: {len(data_by_family['smooth'])}, Harsh: {len(data_by_family['harsh'])}, Static: {len(data_by_family['static'])}")

    # Combine dataframes within each family
    for family in data_by_family:
        if data_by_family[family]:
            data_by_family[family] = pd.concat(data_by_family[family], ignore_index=True)
            print(f"{family} family: {len(data_by_family[family])} rows")
        else:
            data_by_family[family] = None
            print(f"{family} family: No data")

    return data_by_family


def plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/LLM"):
    """
    Generate Pareto frontier plots for each rule family and a combined plot.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    family_titles = {
        "smooth": "Smoothing Rules - Pareto Frontier",
        "harsh": "Harsh Rules - Pareto Frontier",
        "static": "Static Rules - Pareto Frontier",
        "all": "All Rules - Pareto Frontier"
    }

    for family, data in data_by_family.items():
        if data is None or len(data) == 0:
            print(f"Skipping {family} family - no data")
            continue

        print(f"Creating Pareto frontier for {family} family")

        # Create the plot
        plot_pareto_frontier_for_family_pareto_labels(data, family_titles[family], family, output_dir)


# def plot_pareto_frontier_for_family(data, title, family, output_dir):
#     """
#     Create a Pareto frontier plot for a specific rule family.
#     """
#     plt.figure(figsize=(14, 10))
#     unique_rules = data["Rule"].unique()
#     print(f"Processing {len(unique_rules)} unique rules for {family}")
#
#     point_dict = defaultdict(list)
#     for i, rule in enumerate(unique_rules, 1):
#         rule_data = data[data["Rule"] == rule]
#
#         # Find the last event for each rule
#         max_event = rule_data["Event"].max()
#
#         # Get the final point (last event)
#         final_point = rule_data[rule_data["Event"] == max_event].iloc[0]
#
#         # Round to avoid floating point issues
#         key = (round(final_point["Stability"], 3), round(final_point["Sat_Sum_Normalized"], 3))
#
#         # Store information about this point
#         point_dict[key].append({
#             "Rule": f"R{i}",
#             "Original_Rule": final_point["Rule"],
#             "Stability": key[0],
#             "Satisfaction": key[1],
#             "Clean_Label": clean_label(final_point["Rule"])
#         })
#
#     # Process points with the same coordinates
#     final_points = []
#     for key, points in point_dict.items():
#         if len(points) > 1:
#             # Multiple rules ended at the same point
#             sorted_rules = sorted([p["Rule"] for p in points], key=lambda x: int(x[1:]))
#             combined_rule = ",".join(sorted_rules)
#             clean_labels = [p["Clean_Label"] for p in sorted(points, key=lambda x: x["Rule"])]
#
#             final_points.append({
#                 "Rule": combined_rule,
#                 "Original_Rules": [p["Original_Rule"] for p in sorted(points, key=lambda x: x["Rule"])],
#                 "Clean_Labels": clean_labels,
#                 "Stability": key[0],
#                 "Satisfaction": key[1]
#             })
#         else:
#             # Single rule at this point
#             final_points.append({
#                 "Rule": points[0]["Rule"],
#                 "Original_Rules": [points[0]["Original_Rule"]],
#                 "Clean_Labels": [points[0]["Clean_Label"]],
#                 "Stability": key[0],
#                 "Satisfaction": key[1]
#             })
#
#     # Find Pareto-optimal points
#     pareto_points = []
#     for p1 in final_points:
#         is_pareto = True
#         for p2 in final_points:
#             # Check if p2 dominates p1
#             if (p2["Stability"] >= p1["Stability"] and
#                     p2["Satisfaction"] >= p1["Satisfaction"] and
#                     (p2["Stability"] > p1["Stability"] or
#                      p2["Satisfaction"] > p1["Satisfaction"])):
#                 is_pareto = False
#                 break
#         if is_pareto:
#             pareto_points.append(p1)
#
#     # Sort Pareto points by stability
#     pareto_points.sort(key=lambda x: x["Stability"])
#
#     # Draw the dominated area for each Pareto point
#     for point in pareto_points:
#         rect = patches.Rectangle((0, 0), point["Stability"], point["Satisfaction"],
#                                  facecolor='blue', alpha=0.1)
#         plt.gca().add_patch(rect)
#         plt.vlines(point["Stability"], 0, point["Satisfaction"],
#                    colors='blue', alpha=0.3, linestyles='--')
#         plt.hlines(point["Satisfaction"], 0, point["Stability"],
#                    colors='blue', alpha=0.3, linestyles='--')
#
#     # Plot all final points
#     for i, point in enumerate(final_points):
#         is_pareto = point in pareto_points
#         marker = '*'
#         color = 'blue' if is_pareto else 'gray'
#         size = 150 if is_pareto else 50
#
#         plt.scatter(point["Stability"], point["Satisfaction"],
#                     s=size, color=color, marker=marker)
#
#         # Label each point
#         plt.annotate(point["Rule"],
#                      (point["Stability"], point["Satisfaction"]),
#                      xytext=(10, 10 if is_pareto else -10), textcoords='offset points',
#                      fontsize=9, weight='bold' if is_pareto else 'normal', color='blue' if is_pareto else 'black')
#
#     # Draw the Pareto frontier line
#     if len(pareto_points) > 1:
#         pareto_x = [p["Stability"] for p in pareto_points]
#         pareto_y = [p["Satisfaction"] for p in pareto_points]
#         plt.plot(pareto_x, pareto_y, 'b--', linewidth=2, label='Pareto Frontier')
#
#     # Prepare rule mapping for the legend
#     all_rules = []
#     for point in final_points:
#         # Handle multiple rules at the same point
#         rules = point["Rule"].split(",") if "," in point["Rule"] else [point["Rule"]]
#         for i, rule in enumerate(point["Original_Rules"]):
#             # Use clean_label for the rule text
#             clean_lbl = point["Clean_Labels"][i] if i < len(point["Clean_Labels"]) else clean_label(rule)
#             all_rules.append((rules[i], clean_lbl))
#
#     # Sort rules by number and create the legend text
#     all_rules.sort(key=lambda x: int(x[0][1:]))
#     rule_mapping = "\n".join([f'{rule[0]} = {rule[1]}' for rule in all_rules])
#
#     # Add the rule mapping legend on the right side
#     props = dict(boxstyle='round', facecolor='white', alpha=0.9)
#     plt.figtext(0.87, 0.5, "Rule mapping:\n" + rule_mapping,
#                 fontsize=8, va='center', bbox=props)
#
#     # Set plot attributes
#     plt.xlim(0.4, 1)
#     plt.ylim(0, 1)
#     # plt.title(title, fontsize=14, weight='bold')
#     plt.xlabel("Stability", fontsize=12)
#     plt.ylabel("Satisfaction", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#
#     # Place the legend outside the plot to avoid overlapping
#     if len(pareto_points) > 1:
#         plt.legend(bbox_to_anchor=(1.01, 0.95), loc='upper left', fontsize=10)
#
#     # Adjust layout and save
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the rule mapping
#
#     output_path = os.path.join(output_dir, f"pareto_{family}_rules.png")
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     print(f"Saved plot to {output_path}")
#     plt.close()


# Only pareto points labels
def plot_pareto_frontier_for_family_pareto_labels(data, title, family, output_dir):
    """
    Create a Pareto frontier plot for a specific rule family,
    with the following enhanced features:
    1. Clean color scheme with #4472C4 blue
    2. Labels only on Pareto points
    5. Blue shaded areas for dominated regions
    """
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'Times New Roman'
    # plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(14, 10))
    unique_rules = data["Rule"].unique()
    print(f"Processing {len(unique_rules)} unique rules for {family}")

    point_dict = defaultdict(list)
    for i, rule in enumerate(unique_rules, 1):
        rule_data = data[data["Rule"] == rule]

        # Find the last event for each rule
        max_event = rule_data["Event"].max()

        # Get the final point (last event)
        final_point = rule_data[rule_data["Event"] == max_event].iloc[0]

        # Round to avoid floating point issues
        key = (round(final_point["Stability"], 3), round(final_point["Sat_Sum_Normalized"], 3))

        # Store information about this point
        point_dict[key].append({
            "Rule": f"R{i}",
            "Original_Rule": final_point["Rule"],
            "Stability": key[0],
            "Satisfaction": key[1],
            "Clean_Label": clean_label(final_point["Rule"]) if 'clean_label' in globals() else final_point["Rule"]
        })

    # Process points with the same coordinates
    final_points = []
    for key, points in point_dict.items():
        if len(points) > 1:
            # Multiple rules ended at the same point
            sorted_rules = sorted([p["Rule"] for p in points], key=lambda x: int(x[1:]))
            combined_rule = ",".join(sorted_rules)
            clean_labels = [p["Clean_Label"] for p in sorted(points, key=lambda x: x["Rule"])]

            final_points.append({
                "Rule": combined_rule,
                "Original_Rules": [p["Original_Rule"] for p in sorted(points, key=lambda x: x["Rule"])],
                "Clean_Labels": clean_labels,
                "Stability": key[0],
                "Satisfaction": key[1]
            })
        else:
            # Single rule at this point
            final_points.append({
                "Rule": points[0]["Rule"],
                "Original_Rules": [points[0]["Original_Rule"]],
                "Clean_Labels": [points[0]["Clean_Label"]],
                "Stability": key[0],
                "Satisfaction": key[1]
            })

    # Find Pareto-optimal points
    pareto_points = []
    for p1 in final_points:
        is_pareto = True
        for p2 in final_points:
            # Check if p2 dominates p1
            if (p2["Stability"] >= p1["Stability"] and
                    p2["Satisfaction"] >= p1["Satisfaction"] and
                    (p2["Stability"] > p1["Stability"] or
                     p2["Satisfaction"] > p1["Satisfaction"])):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(p1)

    # Sort Pareto points by stability
    pareto_points.sort(key=lambda x: x["Stability"])

    # Blue shaded areas for dominated regions
    for point in pareto_points:
        rect = patches.Rectangle((0, 0), point["Stability"], point["Satisfaction"],
                                 facecolor='#4472C4', alpha=0.1)
        plt.gca().add_patch(rect)
        plt.vlines(point["Stability"], 0, point["Satisfaction"],
                   colors='#4472C4', alpha=0.3, linestyles='--')
        plt.hlines(point["Satisfaction"], 0, point["Stability"],
                   colors='#4472C4', alpha=0.3, linestyles='--')

    # Plot all final points
    for i, point in enumerate(final_points):
        is_pareto = point in pareto_points
        marker = '*'

        # FEATURE 1: Clean color scheme with #4472C4 blue
        color = '#4472C4' if is_pareto else 'lightgray'
        size = 150 if is_pareto else 50

        plt.scatter(point["Stability"], point["Satisfaction"],
                    s=size, color=color, marker=marker)

        # FEATURE 2: Labels only on Pareto points
        if is_pareto:
            plt.annotate(point["Rule"],
                         (point["Stability"], point["Satisfaction"]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=9, weight='bold', color='#4472C4')

    # Draw the Pareto frontier line
    if len(pareto_points) > 1:
        pareto_x = [p["Stability"] for p in pareto_points]
        pareto_y = [p["Satisfaction"] for p in pareto_points]
        plt.plot(pareto_x, pareto_y, '--', linewidth=2, color='#4472C4', label='Pareto Frontier')

    # Create a set of rule IDs that are Pareto-optimal for easy lookup
    pareto_rule_ids = set()
    for point in pareto_points:
        if "," in point["Rule"]:
            # Handle combined rules
            rule_ids = point["Rule"].split(",")
            for rule_id in rule_ids:
                pareto_rule_ids.add(rule_id)
        else:
            pareto_rule_ids.add(point["Rule"])

    # Prepare rule mapping for the legend - split into two separate lists
    pareto_rules = []
    non_pareto_rules = []

    for point in final_points:
        # Handle multiple rules at the same point
        rules = point["Rule"].split(",") if "," in point["Rule"] else [point["Rule"]]
        for i, rule in enumerate(point["Original_Rules"]):
            rule_id = rules[i] if i < len(rules) else rules[0]
            # Use clean_label for the rule text if available
            clean_lbl = point["Clean_Labels"][i] if i < len(point["Clean_Labels"]) else rule

            # Add to appropriate list based on if it's a Pareto point
            if rule_id in pareto_rule_ids:
                pareto_rules.append((rule_id, clean_lbl))
            else:
                non_pareto_rules.append((rule_id, clean_lbl))

    # Sort both lists by rule number
    pareto_rules.sort(key=lambda x: int(x[0][1:]))
    non_pareto_rules.sort(key=lambda x: int(x[0][1:]))

    # Create legend items for the Pareto elements
    legend_elements = []

    # Add Pareto point star to legend
    legend_elements.append(
        mlines.Line2D([], [], color='#4472C4', marker='*', linestyle='None',
                      markersize=10, label='Pareto Optimal Rules')
    )

    # Add Pareto frontier line to legend
    legend_elements.append(
        mlines.Line2D([], [], color='#4472C4', linestyle='--',
                      linewidth=2, label='Pareto Frontier')
    )

    # Place legend elements at the bottom of the plot
    plt.legend(handles=legend_elements, loc='lower left', fontsize=10)

    # Place Pareto rules inside the plot at the true top left corner
    pareto_text = "Pareto Rules:\n" + "\n".join([f'{rid} = {rtxt}' for rid, rtxt in pareto_rules])
    pareto_box = dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5)
    plt.text(1.02, 0.99, pareto_text, transform=plt.gca().transAxes,
             fontsize=12, color='#4472C4', weight='bold', va='top', ha='left',
             bbox=pareto_box)

    # Non-Pareto rules on the right side
    if non_pareto_rules:
        non_pareto_text = "Other Rules:\n" + "\n".join([f'{rid} = {rtxt}' for rid, rtxt in non_pareto_rules])
        non_pareto_box = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.figtext(0.87, 0.5, non_pareto_text,
                    fontsize=8, color='gray', va='center',
                    bbox=non_pareto_box)

    # Set plot attributes
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    # plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("Stability", fontsize=20, fontweight='bold')
    plt.ylabel("Satisfaction", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path = os.path.join(output_dir, f"pareto_{family}_rules.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_unified_analysis_by_family(data_by_family, output_dir="results/figures/coverage_analysis"):
    """
    Generate unified Pareto and dominance area plots for each rule family and calculate metrics.

    Parameters:
    -----------
    data_by_family : dict
        Dictionary where keys are family names and values are pandas DataFrames
        containing the rule data for that family.
    output_dir : str, optional
        Directory to save the output plots.

    Returns:
    --------
    dict
        Dictionary of dominance areas for each family.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    family_titles = {
        "smooth": "Smoothing Rules - Analysis",
        "harsh": "Harsh Rules - Analysis",
        "static": "Static Rules - Analysis",
        "all": "All Rules - Analysis"
    }

    # Dictionary to store the dominance areas
    dominance_results = {}

    for family, data in data_by_family.items():
        if data is None or len(data) == 0:
            print(f"Skipping {family} family - no data")
            continue

        print(f"Creating unified analysis for {family} family")

        # Create the unified plot and get the dominance area
        title = family_titles.get(family, f"{family.capitalize()} Rules - Analysis")
        dominance_area = plot_unified_pareto_and_dominance(data, title, family, output_dir)

        # Store the result
        dominance_results[family] = dominance_area

    return dominance_results


# Compare normal and two-peaks

def get_pareto_points(data, dataset_name):
    """Extract Pareto-optimal points while grouping rules with identical coordinates."""
    from collections import defaultdict

    unique_rules = data["Rule"].unique()
    point_dict = defaultdict(list)

    for i, rule in enumerate(unique_rules, 1):
        rule_data = data[data["Rule"] == rule]
        max_event = rule_data["Event"].max()
        final_point = rule_data[rule_data["Event"] == max_event].iloc[0]

        key = (round(final_point["Stability"], 3), round(final_point["Sat_Sum_Normalized"], 3))
        point_dict[key].append({
            "Rule": f"R{i}",
            "Original_Rule": rule,
            "Stability": key[0],
            "Satisfaction": key[1]
        })

    # Prepare all final points (some may be overlapping)
    final_points = []
    for key, points in point_dict.items():
        if len(points) > 1:
            final_points.append({
                "Rule": None,  # to be filled with group label later
                "Original_Rule": [p["Original_Rule"] for p in points],
                "Rules": [p["Rule"] for p in points],
                "Stability": key[0],
                "Satisfaction": key[1]
            })
        else:
            point = points[0]
            point["Rules"] = [point["Rule"]]
            final_points.append(point)

    # Identify Pareto-optimal points
    pareto_points = []
    for p1 in final_points:
        is_pareto = True
        for p2 in final_points:
            if (p2["Stability"] >= p1["Stability"] and
                    p2["Satisfaction"] >= p1["Satisfaction"] and
                    (p2["Stability"] > p1["Stability"] or p2["Satisfaction"] > p1["Satisfaction"])):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(p1)

    # Only create labels for Pareto points
    pareto_keys = {(p["Stability"], p["Satisfaction"]) for p in pareto_points}
    filtered_point_dict = {k: v for k, v in point_dict.items() if k in pareto_keys}
    group_labels, label_mappings = create_group_labels(filtered_point_dict)

    # Assign correct group labels back to Pareto points
    for p in pareto_points:
        key = (p["Stability"], p["Satisfaction"])
        p["Rule"] = group_labels[key]

    return sorted(pareto_points, key=lambda x: x["Stability"]), label_mappings


def create_group_labels(point_dict):
    """Assign unique group labels (G{i}) to overlapping Pareto points."""
    group_labels = {}
    label_mappings = []
    group_index = 1

    for key, points in point_dict.items():
        if len(points) > 1:
            group_label = f"G{group_index}"
            group_labels[key] = group_label
            group_index += 1

            all_rules = [p["Rule"] for p in points]
            # Map R{i} → clean_label(Original_Rule)
            all_descriptions = {p["Rule"]: clean_label(p["Original_Rule"]) for p in points}
            label_mappings.append((group_label, all_rules, all_descriptions))
        else:
            point = points[0]
            group_labels[key] = point["Rule"]
            label_mappings.append(
                (point["Rule"], [point["Rule"]],
                 {point["Rule"]: clean_label(point["Original_Rule"])}))

    return group_labels, label_mappings


def plot_combined_pareto_distributions(data_normal, data_bimodal, output_path):
    """Create visually improved plot with dynamic label placement to avoid overlap."""
    def is_display_overlap(ax, x, y, offset_xy, used_positions, threshold=10):
        """Check if placing a label at (x, y) + offset in display space would overlap others."""
        x_data, y_data = ax.transData.transform((x, y))
        label_pos = np.array([x_data + offset_xy[0], y_data + offset_xy[1]])
        return any(np.linalg.norm(label_pos - np.array(p)) < threshold for p in used_positions)

    def get_safe_offset(ax, x, y, used_positions):
        """Try multiple offsets and return the first that avoids overlap."""
        for offset in [(10, 10), (10, -20), (10, 25), (10, -35), (10, 40)]:
            if not is_display_overlap(ax, x, y, offset, used_positions):
                return offset
        return (10, 10)  # fallback

    # Visual settings
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    pareto_normal, normal_mappings = get_pareto_points(data_normal, "normal")
    pareto_bimodal, bimodal_mappings = get_pareto_points(data_bimodal, "bimodal")

    used_label_positions_normal = []
    used_label_positions_bimodal = []

    # --- Normal ---
    for point in pareto_normal:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#4472C4', marker='*', edgecolor='black', linewidth=0.5)

        offset = get_safe_offset(ax, x, y, used_label_positions_normal)
        label_disp = ax.transData.transform((x, y))
        label_disp_with_offset = (label_disp[0] + offset[0], label_disp[1] + offset[1])
        used_label_positions_normal.append(label_disp_with_offset)

        plt.annotate(point["Rule"], (x, y),
                     xytext=offset, textcoords='offset points',
                     fontsize=10, weight='bold', color='#4472C4')

    if len(pareto_normal) > 1:
        x_n = [p["Stability"] for p in pareto_normal]
        y_n = [p["Satisfaction"] for p in pareto_normal]
        plt.plot(x_n, y_n, '--', linewidth=2.5, color='#4472C4', alpha=0.8)

    # --- Bimodal ---
    for point in pareto_bimodal:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#2ca02c', marker='*', edgecolor='black', linewidth=0.5)

        offset = get_safe_offset(ax, x, y, used_label_positions_bimodal)
        label_disp = ax.transData.transform((x, y))
        label_disp_with_offset = (label_disp[0] + offset[0], label_disp[1] + offset[1])
        used_label_positions_bimodal.append(label_disp_with_offset)

        plt.annotate(point["Rule"], (x, y),
                     xytext=offset, textcoords='offset points',
                     fontsize=10, weight='bold', color='#2ca02c')

    if len(pareto_bimodal) > 1:
        x_b = [p["Stability"] for p in pareto_bimodal]
        y_b = [p["Satisfaction"] for p in pareto_bimodal]
        plt.plot(x_b, y_b, '--', linewidth=2.5, color='#2ca02c', alpha=0.8)

    # Legend text blocks
    def create_legend_text(mappings):
        legend_lines = []
        for label, rules, descriptions in mappings:
            if "G" in label:
                rule_list = ", ".join(sorted(rules, key=lambda x: int(x[1:])))
                legend_lines.append(f"{label} = {rule_list}")
                for rule in sorted(rules, key=lambda x: int(x[1:])):
                    desc = descriptions.get(rule, "")
                    legend_lines.append(f"    {rule} = {desc}")
            else:
                rule = rules[0]
                desc = descriptions[rule]
                legend_lines.append(f"{label} = {desc}")
        return "\n".join(legend_lines)

    normal_text = "Normal Pareto Rules:\n" + create_legend_text(normal_mappings)
    plt.text(0.02, 0.98, normal_text, transform=ax.transAxes,
             fontsize=11, color='#4472C4', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#4472C4', linewidth=1))

    bimodal_text = "Bimodal Pareto Rules:\n" + create_legend_text(bimodal_mappings)
    plt.text(0.22, 0.98, bimodal_text, transform=ax.transAxes,
             fontsize=11, color='#2ca02c', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#2ca02c', linewidth=1))

    # Legend for the curves
    legend_elements = [
        mlines.Line2D([], [], color='#4472C4', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label='Normal Population'),
        mlines.Line2D([], [], color='#2ca02c', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label='Two-Peaks Population')
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=14,
               frameon=True, framealpha=0.95, edgecolor='black')

    # Axes
    plt.xlim(0.5, 1.03)
    plt.ylim(0.3, 1.03)
    plt.xticks(np.arange(0.5, 1.03, 0.1), fontsize=16)
    plt.yticks(np.arange(0.3, 1.03, 0.1), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    plt.xlabel("Stability", fontsize=22, fontweight='bold')
    plt.ylabel("Satisfaction", fontsize=22, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Set grid for better readability
    ax.grid(True, linestyle=':', alpha=0.3)

    # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.02))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))
    # plt.grid(which='minor', linestyle=':', alpha=0.2, color='gray')

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.2,
        format='png'
    )
    print(f"Saved plot to {output_path}")
    plt.close()

# Compare two general agents


def plot_combined_pareto_two_agents(data_config001, data_config002, output_path, config001_name, config002_name):
    """Create Pareto plot for two LLM configurations using adaptive label positioning."""
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    pareto_1, mapping_1 = get_pareto_points(data_config001, config001_name)
    pareto_2, mapping_2 = get_pareto_points(data_config002, config002_name)

    text_labels = []

    # --- Config001 ---
    for point in pareto_1:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#4472C4', marker='*', edgecolor='black', linewidth=0.5)
        text_labels.append(
            plt.text(x + 0.005, y + 0.005, point["Rule"], fontsize=11, weight='bold', color='#4472C4')
        )

    if len(pareto_1) > 1:
        x1 = [p["Stability"] for p in pareto_1]
        y1 = [p["Satisfaction"] for p in pareto_1]
        plt.plot(x1, y1, '--', linewidth=2.5, color='#4472C4', alpha=0.8)

    # --- Config002 ---
    for point in pareto_2:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#2ca02c', marker='*', edgecolor='black', linewidth=0.5)
        text_labels.append(
            plt.text(x + 0.01, y + 0.01, point["Rule"], fontsize=11, weight='bold', color='#2ca02c')
        )

    if len(pareto_2) > 1:
        x2 = [p["Stability"] for p in pareto_2]
        y2 = [p["Satisfaction"] for p in pareto_2]
        plt.plot(x2, y2, '--', linewidth=2.5, color='#2ca02c', alpha=0.8)

    adjust_text(text_labels, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    def create_legend_text(mappings):
        legend_lines = []
        for label, rules, descriptions in mappings:
            if "G" in label:
                rule_list = ", ".join(sorted(rules, key=lambda x: int(x[1:])))
                legend_lines.append(f"{label} = {rule_list}")
                for rule in sorted(rules, key=lambda x: int(x[1:])):
                    desc = descriptions.get(rule, "")
                    legend_lines.append(f"    {rule} = {desc}")
            else:
                rule = rules[0]
                desc = descriptions[rule]
                legend_lines.append(f"{label} = {desc}")
        return "\n".join(legend_lines)

    # Text blocks
    text_1 = f"{config001_name} Pareto Rules:\n" + create_legend_text(mapping_1)
    text_2 = f"{config002_name} Pareto Rules:\n" + create_legend_text(mapping_2)

    plt.text(1.03, 0.99, text_1, transform=ax.transAxes,
             fontsize=10, color='#4472C4', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#4472C4', linewidth=1))

    plt.text(1.03, 0.49, text_2, transform=ax.transAxes,
             fontsize=10, color='#2ca02c', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#2ca02c', linewidth=1))

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color='#4472C4', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config001_name),
        mlines.Line2D([], [], color='#2ca02c', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config002_name)
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=14,
               frameon=True, framealpha=0.95, edgecolor='black')

    plt.xlim(0.3, 1.03)
    plt.ylim(0.2, 1.03)
    plt.xticks(np.arange(0.3, 1.03, 0.1), fontsize=16)
    plt.yticks(np.arange(0.2, 1.03, 0.1), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    plt.xlabel("Stability", fontsize=22, fontweight='bold')
    plt.ylabel("Satisfaction", fontsize=22, fontweight='bold')
    plt.title("Pareto Comparison - Unstructured vs Uniformal", fontsize=24, fontweight='bold', pad=20)
    plt.grid(True, linestyle=':', alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2, format='png')
    print(f"Saved plot to {output_path}")
    plt.close()


# Combine configurations - (LLM configuration)

def plot_combined_pareto_configuration_llm(data_config001, data_config002, output_path, config001_name, config002_name):
    """Create Pareto plot for two LLM configurations using adaptive label positioning."""
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    pareto_1, mapping_1 = get_pareto_points(data_config001, config001_name)
    pareto_2, mapping_2 = get_pareto_points(data_config002, config002_name)

    text_labels = []

    # --- Config001 ---
    for point in pareto_1:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#4472C4', marker='*', edgecolor='black', linewidth=0.5)
        text_labels.append(
            plt.text(x + 0.003, y + 0.003, point["Rule"], fontsize=11, weight='bold', color='#4472C4')
        )

    if len(pareto_1) > 1:
        x1 = [p["Stability"] for p in pareto_1]
        y1 = [p["Satisfaction"] for p in pareto_1]
        plt.plot(x1, y1, '--', linewidth=2.5, color='#4472C4', alpha=0.8)

    # --- Config002 ---
    for point in pareto_2:
        x, y = point["Stability"], point["Satisfaction"]
        plt.scatter(x, y, s=200, color='#2ca02c', marker='*', edgecolor='black', linewidth=0.5)
        text_labels.append(
            plt.text(x - 0.01, y - 0.01, point["Rule"], fontsize=11, weight='bold', color='#2ca02c')
        )

    if len(pareto_2) > 1:
        x2 = [p["Stability"] for p in pareto_2]
        y2 = [p["Satisfaction"] for p in pareto_2]
        plt.plot(x2, y2, '--', linewidth=2.5, color='#2ca02c', alpha=0.8)

    adjust_text(text_labels, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    def create_legend_text(mappings):
        legend_lines = []
        for label, rules, descriptions in mappings:
            if "G" in label:
                rule_list = ", ".join(sorted(rules, key=lambda x: int(x[1:])))
                legend_lines.append(f"{label} = {rule_list}")
                for rule in sorted(rules, key=lambda x: int(x[1:])):
                    desc = descriptions.get(rule, "")
                    legend_lines.append(f"    {rule} = {desc}")
            else:
                rule = rules[0]
                desc = descriptions[rule]
                legend_lines.append(f"{label} = {desc}")
        return "\n".join(legend_lines)

    # Text blocks
    text_1 = f"{config001_name} Pareto Rules:\n" + create_legend_text(mapping_1)
    text_2 = f"{config002_name} Pareto Rules:\n" + create_legend_text(mapping_2)

    plt.text(0.02, 0.98, text_1, transform=ax.transAxes,
             fontsize=13, color='#4472C4', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#4472C4', linewidth=1))

    plt.text(0.42, 0.98, text_2, transform=ax.transAxes,
             fontsize=13, color='#2ca02c', weight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       alpha=0.95, edgecolor='#2ca02c', linewidth=1))

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color='#4472C4', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config001_name),
        mlines.Line2D([], [], color='#2ca02c', marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config002_name)
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=14,
               frameon=True, framealpha=0.95, edgecolor='black')

    plt.xlim(0.7, 1.03)
    plt.ylim(0.4, 1.03)
    plt.xticks(np.arange(0.7, 1.03, 0.1), fontsize=16)
    plt.yticks(np.arange(0.4, 1.03, 0.1), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    plt.xlabel("Stability", fontsize=22, fontweight='bold')
    plt.ylabel("Satisfaction", fontsize=22, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2, format='png')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_combined_pareto_configuration_llm_three(data_config001, data_config002, data_config003,
                                                  config001_name, config002_name, config003_name,
                                                  output_path):

    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    pareto_1, mapping_1 = get_pareto_points(data_config001, config001_name)
    pareto_2, mapping_2 = get_pareto_points(data_config002, config002_name)
    pareto_3, mapping_3 = get_pareto_points(data_config003, config003_name)

    text_labels = []

    colors = ['#4472C4', '#2ca02c', '#D62728']
    pareto_sets = [pareto_1, pareto_2, pareto_3]
    mappings = [mapping_1, mapping_2, mapping_3]
    config_names = [config001_name, config002_name, config003_name]

    for idx, (pareto, color) in enumerate(zip(pareto_sets, colors)):
        for point in pareto:
            x, y = point["Stability"], point["Satisfaction"]
            plt.scatter(x, y, s=200, color=color, marker='*', edgecolor='black', linewidth=0.5)
            text_labels.append(
                plt.text(x + 0.003, y + 0.003, point["Rule"], fontsize=11, weight='bold', color=color)
            )
        if len(pareto) > 1:
            x_vals = [p["Stability"] for p in pareto]
            y_vals = [p["Satisfaction"] for p in pareto]
            plt.plot(x_vals, y_vals, '--', linewidth=2.5, color=color, alpha=0.8)

    adjust_text(text_labels, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    def create_legend_text(mapping):
        lines = []
        for label, rules, descriptions in mapping:
            if "G" in label:
                rule_list = ", ".join(sorted(rules, key=lambda x: int(x[1:])))
                lines.append(f"{label} = {rule_list}")
                for rule in sorted(rules, key=lambda x: int(x[1:])):
                    desc = descriptions.get(rule, "")
                    lines.append(f"    {rule} = {desc}")
            else:
                rule = rules[0]
                desc = descriptions[rule]
                lines.append(f"{label} = {desc}")
        return "\n".join(lines)

    legend_texts = [create_legend_text(m) for m in mappings]
    colors_rgb = ['#4472C4', '#2ca02c', '#D62728']
    text_coords = [(0.02, 0.98), (0.37, 0.98), (0.69, 0.98)]

    for i in range(3):
        plt.text(*text_coords[i], f"{config_names[i]} Pareto Rules:\n{legend_texts[i]}",
                 transform=ax.transAxes, fontsize=13, color=colors_rgb[i], weight='bold',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           alpha=0.95, edgecolor=colors_rgb[i], linewidth=1))

    legend_elements = [
        mlines.Line2D([], [], color=colors[0], marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config001_name),
        mlines.Line2D([], [], color=colors[1], marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config002_name),
        mlines.Line2D([], [], color=colors[2], marker='*', linestyle='--',
                      markersize=12, linewidth=2.5, label=config003_name)
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=14,
               frameon=True, framealpha=0.95, edgecolor='black')

    plt.xlim(0.7, 1.03)
    plt.ylim(0.4, 1.03)
    plt.xticks(np.arange(0.7, 1.03, 0.1), fontsize=16)
    plt.yticks(np.arange(0.4, 1.03, 0.1), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    plt.xlabel("Stability", fontsize=22, fontweight='bold')
    plt.ylabel("Satisfaction", fontsize=22, fontweight='bold')
    plt.title("Pareto Comparison Across LLM Configurations", fontsize=24, fontweight='bold', pad=20)
    plt.grid(True, linestyle=':', alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2, format='png')
    print(f"Saved plot to {output_path}")
    plt.close()


# Size analysis

def collect_solution_sizes(base_dir="results/all_rules"):
    """
    Create a dataset of final solution sizes for each (agent_type, rule).

    Returns:
    --------
    pd.DataFrame
        Columns: ['AgentType', 'Rule', 'RuleFamily', 'FinalSize']
    """
    agent_types = ["Unstructured", "Uniform", "Normal", "Two-Peaks"]
    records = []

    for agent_type in agent_types:
        if agent_type == "Uniform":
            agent_dir = "uniformal interval"
        elif agent_type == "Normal":
            agent_dir = "normal interval"
        elif agent_type == "Two-Peaks":
            agent_dir = "two peaks interval"
        else:
            agent_dir = agent_type.lower()

        agent_path = os.path.join(base_dir, agent_dir)
        if not os.path.exists(agent_path):
            continue

        for rule in os.listdir(agent_path):
            rule_path = os.path.join(agent_path, rule, f"{rule}.csv")
            if not os.path.isfile(rule_path):
                print(f"skipe for {agent_type} {rule_path}")
                continue

            try:
                df = pd.read_csv(rule_path)
                if "Size" in df.columns and len(df) > 0:
                    final_size = df["Size"].iloc[-1]
                    rule_family = determine_rule_family(rule)
                    records.append({
                        "AgentType": agent_type,
                        "Rule": rule,
                        "RuleFamily": rule_family,
                        "FinalSize": final_size
                    })
            except Exception as e:
                print(f"Error reading {rule_path}: {e}")

    return pd.DataFrame(records)


def summarize_solution_sizes(df):
    """
    Compute mean and std of solution size by agent type and rule family.

    Parameters:
    -----------
    df : pd.DataFrame
        Output of collect_solution_sizes

    Returns:
    --------
    pd.DataFrame
        Grouped summary statistics
    """
    return df.groupby(["AgentType", "RuleFamily"])["FinalSize"].agg(["mean", "std", "count"]).reset_index()


def analyze_solution_size(base_dir="results/all_rules"):
    """
    Analyze the solution size for different agent models and rule families.

    Parameters:
    -----------
    base_dir : str
        Base directory containing all agent results

    Returns:
    --------
    dict
        Nested dictionary with statistics for each agent type and rule family
    """
    agent_types = ["Unstructured", "Uniform", "Normal", "Two-Peaks"]
    results = {}

    for agent_type in agent_types:
        if agent_type == "Uniform":
            agent_dir = "uniformal interval"
        elif agent_type == "Normal":
            agent_dir = "normal interval"
        elif agent_type == "Two-Peaks":
            agent_dir = "two peaks interval"
        else:
            agent_dir = agent_type.lower()

        agent_path = os.path.join(base_dir, agent_dir)
        print(f"\n{agent_type}")

        if not os.path.exists(agent_path):
            print(f"Directory not found: {agent_path}")
            continue

        rules_by_family = defaultdict(list)
        rule_dirs = [d for d in os.listdir(agent_path)
                     if os.path.isdir(os.path.join(agent_path, d))]

        for rule_dir in rule_dirs:
            csv_path = os.path.join(agent_path, rule_dir, f"{rule_dir}.csv")
            if not os.path.isfile(csv_path):
                print(f"Warning: CSV file not found for {rule_dir}")
                continue

            family = determine_rule_family(rule_dir)
            print(f"rule: {rule_dir} , family: {family}")

            try:
                df = pd.read_csv(csv_path)
                if "Size" not in df.columns:
                    print(f"Warning: Size column missing in {csv_path}")
                    continue

                if len(df) > 0:
                    last_size = df["Size"].iloc[-1]
                    if last_size == 0:
                        print(f"Zero size in rule {rule_dir}")
                    rules_by_family[family].append(last_size)
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")

        print(rules_by_family)
        results[agent_type] = dict(rules_by_family)

    return results


if __name__ == "__main__":
    # Snake plots
    for agent in os.listdir("results/all_rules"):
        plot_snake_for_agent_by_family(agent)

    # Unstructured

    data_by_family = collect_non_llm_data(family="unstructured")
    plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/unstructured")
    rules = ["(CSF=0_events,_APS,_threshold=0.5)", "(CSF=50_events,_APS,_threshold=0.5)",
             "(CSF=APS_r,_threshold=0.5)", "(CSF=APS,_threshold=0.5)",
             "(CSF=AM_beta=0.1,_Function=exp_alpha=1_events,_threshold=0.5)",
             "(CSF=AM_beta=0.3,_Function=exp_alpha=0.5_events,_threshold=0.5)"]
    plot_filtered_snake2(rule_filter_list=rules, results_dir="results/all_rules/unstructured",
                         figures_dir="results/figures/unstructured")

    # Uniformal
    data_by_family = collect_non_llm_data(family="uniformal interval")
    plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/uniformal interval")

    # Normal interval
    data_by_family = collect_non_llm_data(family="normal interval")
    plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/normal interval")

    # Two peaks interval
    data_by_family = collect_non_llm_data(family="two peaks interval")
    plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/two peaks interval")

    # Normal vs Two peaks
    data_normal = collect_non_llm_data("normal interval")
    data_bimodal = collect_non_llm_data("two peaks interval")
    plot_combined_pareto_distributions(
        data_normal=data_normal["all"],
        data_bimodal=data_bimodal["all"],
        output_path="results/figures/pareto_normal_two_combined.png"
    )

    # Unstructured vs Uniform
    data_unstructured = collect_non_llm_data("unstructured")
    data_uniformal = collect_non_llm_data("uniformal interval")
    plot_combined_pareto_two_agents(data_config001=data_unstructured["all"],
                                           data_config002=data_uniformal["all"],
                                           output_path="results/figures/unstructured_uniformal_combined.png",
                                           config001_name="Unstructured",
                                           config002_name="Uniformal")

    # - LLM

    # Config 1
    data_config1 = collect_llm_data(config="config001", results_dir="results/all_rules/LLM/config001_llm")

    # Config 2
    data_by_config2 = collect_llm_data(config="config002", results_dir="results/all_rules/LLM/config002_llm")
    plot_pareto_frontier_by_family(data_by_family, output_dir="results/figures/LLM/config002")

    # Config 1 vs 2 LLM
    data_config1 = collect_llm_data(config="config001", results_dir="results/all_rules/LLM/config001_llm")
    data_config2 = collect_llm_data(config="config002", results_dir="results/all_rules/LLM/config002_llm")
    plot_combined_pareto_configuration_llm(data_config001=data_config1["all"],
                                           data_config002=data_config2["all"],
                                           output_path="results/figures/llm_config1_vs_2_combined.png",
                                           config001_name="(20 agents, 150 events)",
                                           config002_name="(20 agents, 250 events)")

    # Config 2 vs 3 LLM
    # data_config2 = collect_llm_data(config="config002", results_dir="results/all_rules/LLM/config002_llm")
    # data_config3 = collect_llm_data(config="config003", results_dir="results/all_rules/LLM/config003_llm")
    data_config2 = collect_llm_data(config="config002", results_dir="datasets/metrics/config002_llm")
    data_config3 = collect_llm_data(config="config003", results_dir="datasets/metrics/config003_llm")

    plot_combined_pareto_configuration_llm(data_config001=data_config3["all"],
                                           data_config002=data_config2["all"],
                                           output_path="results/figures/llm_config2_vs_3_combined.png",
                                           config001_name="(40 agents, 250 events)",
                                           config002_name="(20 agents, 250 events)")

    # All configs together
    data_config1 = collect_llm_data(config="config001", results_dir="results/all_rules/LLM/config001_llm")
    data_config2 = collect_llm_data(config="config002", results_dir="results/all_rules/LLM/config002_llm")
    data_config3 = collect_llm_data(config="config003", results_dir="results/all_rules/LLM/config003_llm")
    plot_combined_pareto_configuration_llm_three(data_config001=data_config1["all"],
                                                 data_config002=data_config2["all"],
                                                 data_config003=data_config3["all"],
                                                 config001_name="(20 agents, 150 events)",
                                                 config002_name="(20 agents, 250 events)",
                                                 config003_name="(40 agents, 250 events)",
                                                 output_path="results/figures/llm_configs_combined.png")

    # Size
    df_sizes = collect_solution_sizes()
    summary = summarize_solution_sizes(df_sizes)

 
    # Generate Pareto frontier plots for each family
    plot_pareto_frontier_by_family(data_by_family, output_dir)
    print("All Pareto frontier plots completed")