"""
Create line plots for accuracy-per-layer data (instead of bar plots).
Each component type is a separate line, with layer depth on the x-axis.

Usage:
    python scripts/line_plot_accuracy_per_layer.py
"""

import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    "paraphrased": os.path.join(
        BASE_DIR,
        "results/accuracy_per_layer/paraphrased/amd/AMD-OLMo-1B-SFT/sample_size/full/values.json",
    ),
    "model_generated": os.path.join(
        BASE_DIR,
        "results/accuracy_per_layer/model_generated/amd/AMD-OLMo-1B-SFT/sample_size/full/values.json",
    ),
}


def parse_layer_data(values: dict) -> dict[str, dict[int, float]]:
    """Parse raw values dict into {component_type: {layer_depth: accuracy}}."""
    components: dict[str, dict[int, float]] = {}

    for key, acc in values.items():
        if "embed_tokens" in key:
            comp = "embed_tokens"
            layer = 0
        else:
            m = re.match(
                r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight$", key
            )
            if not m:
                continue
            layer = int(m.group(1)) + 1  # 1-based layer depth
            comp = f"{m.group(2)}.{m.group(3)}"

        components.setdefault(comp, {})[layer] = acc

    return components


def line_plot(values: dict, title: str, output_path: str):
    """Create a line plot with one line per component type."""
    components = parse_layer_data(values)

    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("tab10", n_colors=len(components))

    for (comp, layer_acc), color in zip(sorted(components.items()), palette):
        layers = sorted(layer_acc.keys())
        accs = [layer_acc[l] for l in layers]
        labels = ["Embed" if l == 0 else str(l) for l in layers]
        plt.plot(range(len(layers)), accs, marker="o", markersize=3, label=comp, color=color)

    # Collect all unique layer depths across all components
    all_layers = sorted(set(l for layer_acc in components.values() for l in layer_acc))
    tick_labels = ["Embed" if l == 0 else str(l) for l in all_layers]
    plt.xticks(range(len(all_layers)), tick_labels)

    plt.ylim(bottom=0)
    plt.xlabel("Layer Depth")
    plt.ylabel("Accuracy")
    plt.legend(title="Component Type", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    for setting, path in DATA_PATHS.items():
        with open(path) as f:
            values = json.load(f)

        output_dir = os.path.dirname(path)
        output_path = os.path.join(output_dir, "line_plot.png")
        title = f"AMD-OLMo-1B-SFT - {setting} - accuracy per layer"
        line_plot(values, title, output_path)


if __name__ == "__main__":
    main()
