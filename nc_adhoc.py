import json
import os
import pickle
from functools import partial

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import ks_2samp

from data_models.Label import NCLabel
from models.nearest_centroid.nearest_centroid import AdhocNearestCentroid
from utils.load_data import SpecimenData
from utils.slide_utils import plot_image

DATA_DIR = "/opt/gpudata/skin-cancer/data"
OUTPUT_DIR = "/opt/gpudata/skin-cancer/outputs"
labels_path = os.path.join(DATA_DIR, "labels/labels.csv")
embeddings_path = os.path.join(OUTPUT_DIR, "prism/tile_embeddings_sorted")

sampled_specs = {
    "bowens": ["660524"],
    "bcc": ["660369"],
    "scc": ["660109"],
    "na": ["660375"],
}

slides_per_specimen = {
    spec: [s[:-4] for s in os.listdir(embeddings_path) if s[:6] == spec]
    for v in sampled_specs.values()
    for spec in v
}


def load_config():
    with open("models/nearest_centroid/config.json", "rb") as f:
        config = json.load(f)
    return config


def tensor_to_html_table(tensor):
    html_table = """
<table>
<tr>
    <td></td>
    <td class="rotated">dermis</td>
    <td class="rotated">epidermis_corneum</td>
    <td class="rotated">bowens</td>
    <td class="rotated">bcc_nodular</td>
    <td class="rotated">bcc_superficial</td>
    <td class="rotated">scc</td>
    <td class="rotated">artifact</td>
</tr>
"""
    html_table += "  <tr>\n    <td></td>\n"
    for label in NCLabel._member_names_:
        html_table += f"    <td>{label}</td>\n"
    html_table += "  </tr>\n"

    for i, row in enumerate(tensor):
        html_table += "  <tr>\n"
        html_table += f"    <td>{NCLabel._member_names_[i]}</td>\n"
        for cell in row:
            html_table += f"    <td>{cell:.4f}</td>\n"
        html_table += "  </tr>\n"
    html_table += "</table>"
    return html_table


def print_html_tables(model):
    print(
        """
<style>
table {
    border-collapse: collapse;
}
table, th, td {
    border: 1px solid black;
    text-align: center;
    width: 80px;
}
td.rotated {
    height: 120px;
    vertical-align: middle;
    text-align: center;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
}
</style>
"""
    )
    print("**Centroid cosine similarities**")
    html_output = tensor_to_html_table(
        model._cosine_similarity(model.centroids)
    )
    print(html_output)

    print("**Centroid euclidean distances**")
    html_output = tensor_to_html_table(
        model._euclidean_distance(model.centroids)
    )
    print(html_output)

    print("**Centroid raw dot products**")
    html_output = tensor_to_html_table(model._dot_product(model.centroids))
    print(html_output)


def softmax_with_temp(preds: torch.Tensor, temp=0.5):
    preds = preds.log() / temp
    return preds.softmax(dim=-1)


def extract_relevant_preds(input: torch.Tensor, ignore_class):
    ignore_class = (
        [ignore_class] if isinstance(ignore_class, int) else ignore_class
    )

    new = []
    for row in input:
        if row.argmax() not in ignore_class:
            new.append(row)
    return torch.stack(new)


def probability_normalization(input: torch.Tensor, dim: int):
    return input / input.sum(dim=dim, keepdim=True)


def log_prob_normalization(input: torch.Tensor, dim: int):
    return torch.log(probability_normalization(input, dim))


def get_preds(
    model: AdhocNearestCentroid, slide_id, normalization: str = "probability"
):
    norm = {
        "probability": partial(probability_normalization, dim=-1),
        "softmax": partial(torch.softmax, dim=-1),
        "log_prob": partial(log_prob_normalization, dim=-1),
    }

    with open(os.path.join(embeddings_path, f"{slide_id}.pkl"), "rb") as f:
        slide_data = pickle.load(f)

    preds = model.predict(
        slide_data["tile_embeds"].float(), mode="dot_product"
    )
    if normalization is not None:
        preds = norm[normalization](input=preds)
    return preds


def plot_kdes(
    model, slides, c, label, color, alpha=0.5, return_max_coords: bool = False
):
    coords = {}
    for i, slide in enumerate(slides):
        preds = get_preds(model, slide, normalization="probability")
        preds = extract_relevant_preds(preds, ignore_class=6)
        plot = sns.kdeplot(
            preds[:, c],
            label=f"{label}_{i}",
            alpha=alpha,
            color=color,
        )
        num_lines = len(plot.get_lines())
        y_coords = plot.get_lines()[num_lines - 1].get_ydata(orig=True)
        x_coords = plot.get_lines()[num_lines - 1].get_xdata(orig=True)
        max_y = np.argmax(y_coords)
        max_x = x_coords[max_y]
        coords[slide] = (max_x, max_y)

    if return_max_coords:
        return coords


def plot_histograms(model, slides, c, label, color, alpha=0.5):
    for i, slide in enumerate(slides):
        preds = get_preds(model, slide, normalization="log_prob")
        preds = extract_relevant_preds(preds, ignore_class=6)
        sns.histplot(
            preds[:, c],
            label=f"{label}_{i}",
            alpha=alpha,
            color=color,
            element="step",
            fill=False,
            stat="density",
        )


def generate_heatmap_plots(model):
    modes = ["dot_product", "cosine", "euclidean"]
    slides_per_specimen = {
        spec: [s[:-4] for s in os.listdir(embeddings_path) if s[:6] == spec]
        for v in sampled_specs.values()
        for spec in v
    }

    for mode in modes:
        try:
            os.makedirs(f"outputs/nearest_centroid_adhoc/{mode}")
        except FileExistsError:
            pass

        pred_func = torch.argmin if mode == "euclidean" else torch.argmax
        for label, specs in sampled_specs.items():
            for spec in specs:
                for slide_id in slides_per_specimen[spec]:
                    print(slide_id)
                    with open(
                        os.path.join(embeddings_path, f"{slide_id}.pkl"), "rb"
                    ) as f:
                        slide_data = pickle.load(f)
                        preds = model.predict(
                            slide_data["tile_embeds"].float(), mode=mode
                        )

                    print(pred_func(preds, dim=-1).unique(return_counts=True))
                    fig, ax = plt.subplots(
                        figsize=(10, 10), constrained_layout=True
                    )
                    plot_image(
                        fpath="/opt/gpudata/skin-cancer/data/slides/"
                        + f"{slide_id}.svs",
                        ax=ax,
                        tile_coords=slide_data["coords"],
                        tile_weights=pred_func(preds, dim=-1),
                        weight_labels={
                            label.name: label.value for label in NCLabel
                        },
                    )
                    fig.savefig(
                        f"outputs/nearest_centroid_adhoc/{mode}/"
                        + f"{label}-{slide_id}.png",
                        bbox_inches="tight",
                        dpi=200,
                    )
                    plt.close()


def ks_test(benchmarks, test):
    results = []
    for bm in benchmarks:
        results.append(ks_2samp(bm[:, 0], test).pvalue)
    print(results)


def ks_test_script():
    model = AdhocNearestCentroid(NCLabel)
    model.fit(tile_embed_dir=embeddings_path, roi_dir="nearest_centroid")
    # model.fit(
    # tile_embed_dir=embeddings_path,
    # roi_dir="nearest_centroid_contains"
    # )

    slides_per_class = {
        c: [
            s[:-4]
            for spec in sampled_specs[c]
            for s in os.listdir(embeddings_path)
            if s[:6] == spec
        ]
        for c in sampled_specs
    }

    ####
    benchmarks = []
    for slide in slides_per_class["na"]:
        benchmarks.append(get_preds(model, slide, normalization="probability"))

    test_sets = {}
    for c, slides in slides_per_class.items():
        test_sets[c] = []
        for slide in slides:
            test_sets[c].append(
                get_preds(model, slide, normalization="probability")
            )

    for c, slide_preds in test_sets.items():
        print(c)
        for slide_pred in slide_preds:
            ks_test(benchmarks, slide_pred[:, 0])


def plot_rois():
    data = SpecimenData(labels_path)
    model = AdhocNearestCentroid(NCLabel)
    weights = {c: i for i, c in enumerate(NCLabel._member_names_)}

    with open("models/nearest_centroid/config.json", "rb") as f:
        roi_config = json.load(f)

    roi_tiles = {}
    polygon_dir = roi_config["annotation_directory"]
    slide_dir = roi_config["slide_directory"]
    tiles_dir = roi_config["tiles_directory"]
    for name, pg_type in roi_config["polygon_type"].items():
        root = os.path.join(polygon_dir, name)
        # for classes with subclasses, must use modified process
        if pg_type == "subclass":
            pgs = model._get_subclass_polygons(root)
            for label, polygons in pgs.items():
                roi_tiles[label] = model._roi_tiles_by_slide(
                    polygon_map=polygons,
                    slide_dir=slide_dir,
                    tiles_dir=tiles_dir,
                    classification=label,
                    output_dir=roi_config["roi_tiles_output_directory"]
                    or None,
                )
        else:
            polygons = model._get_class_polygons(root)
            roi_tiles[name] = model._roi_tiles_by_slide(
                polygon_map=polygons,
                slide_dir=slide_dir,
                tiles_dir=tiles_dir,
                classification=name,
                output_dir=roi_config["roi_tiles_output_directory"] or None,
            )

    rois_by_slide = {}
    for label, label_data in roi_tiles.items():
        for slide_id, coords in label_data.items():
            if slide_id not in rois_by_slide:
                rois_by_slide[slide_id] = {
                    "coords": list(coords),
                    "weights": [weights[label] for _ in range(len(coords))],
                }
            else:
                rois_by_slide[slide_id]["coords"].extend(list(coords))
                rois_by_slide[slide_id]["weights"].extend(
                    [weights[label] for _ in range(len(coords))]
                )

    for slide_id, data in rois_by_slide.items():
        print(f"creating plot for slide {slide_id}")
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        plot_image(
            fpath=f"/opt/gpudata/skin-cancer/data/slides/{slide_id}.svs",
            ax=ax,
            tile_coords=data["coords"],
            tile_weights=data["weights"],
            weight_labels={label.name: label.value for label in NCLabel},
        )
        fig.savefig(
            f"{slide_id}-roi.png",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()


if __name__ == "__main__":
    data = SpecimenData(labels_path)
    model = AdhocNearestCentroid(NCLabel)
    model.fit(tile_embed_dir=embeddings_path, roi_dir="nearest_centroid")
    generate_heatmap_plots(model)
    exit()
    # model.fit(
    # tile_embed_dir=embeddings_path,
    # roi_dir="nearest_centroid_contains"
    # )

    slides_per_class = {
        c: [
            s[:-4]
            for spec in sampled_specs[c]
            for s in os.listdir(embeddings_path)
            if s[:6] == spec
        ]
        for c in sampled_specs
    }

    class_to_labels = {
        "bowens": [2],
        "bcc": [3, 4],
        # "bcc_ref": [3, 4],
        "scc": [5],
    }
    colors = {
        "na": "gray",
        "bowens": "blue",
        "bcc": "green",
        "scc": "red",
        # "bcc_nod": "black",
        # "bcc_sf": "maroon",
    }

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    for c in slides_per_class:
        if c in {"na", "artifact"}:
            continue
        for label in class_to_labels[c]:
            for d in colors:
                if d == c:
                    continue
                plot_histograms(
                    model,
                    slides_per_class[d],
                    label,
                    d,
                    color=colors[d],
                    alpha=1.0 if d in {"bcc_nod", "bcc_sf"} else 0.25,
                )
            plot_histograms(
                model,
                slides_per_class[c],
                label,
                c,
                color=colors[c],
                alpha=0.5,
            )
            plt.legend(handles=patches)
            ax = plt.gca()
            # ax.set_ylim([0, 5])
            plt.title(f"Class of Interest: {NCLabel(label).name}")
            plt.savefig(f"hist_{NCLabel(label).name}.png", dpi=200)
            plt.close()
