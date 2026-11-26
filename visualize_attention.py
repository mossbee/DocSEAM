"""Visualize DocSEAM directional salience gates on three consecutive pages."""

import argparse
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText

from infer import load_model


def averaged_center_record(records, curr_page_index: int):
    selected = [record for record in records if record["curr_page_index"] == curr_page_index]
    if not selected:
        raise ValueError(f"No salience records found for center page index {curr_page_index}.")

    n_prev = selected[0]["n_prev"]
    n_curr = selected[0]["n_curr"]
    n_next = selected[0]["n_next"]
    for record in selected:
        if (record["n_prev"], record["n_curr"], record["n_next"]) != (n_prev, n_curr, n_next):
            raise ValueError("Selected salience records have inconsistent triplet sizes.")

    salience = {}
    for key in ("s_send_to_next", "s_recv_from_prev", "s_send_to_prev", "s_recv_from_next"):
        salience[key] = torch.stack([torch.sigmoid(record[key]).float() for record in selected]).mean(dim=0).numpy()

    return selected, salience


def normalize_heatmaps(heatmaps):
    values = np.concatenate([heatmap.reshape(-1) for heatmap in heatmaps])
    vmin, vmax = np.percentile(values, [2, 98])
    if vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def salience_stats(name, values):
    values = values.astype(np.float64)
    return {
        "name": name,
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mossbee/DocSEAM")
    parser.add_argument("--output", type=Path, default=Path("results/directional_heatmap.png"))
    parser.add_argument("--images", nargs=3, type=Path, help="Previous, center, and next page")
    parser.add_argument("--first-page", default=4, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-model", help="Optional original backbone for Stage A reproduction")
    parser.add_argument("--position-feature-mode", choices=("physical", "frontier"))
    args = parser.parse_args()
    model, processor = load_model(args.model, args.device)
    if args.vision_model:
        backbone = AutoModelForImageTextToText.from_pretrained(
            args.vision_model, dtype=torch.bfloat16, device_map={"": args.device}
        )
        model.model.visual = backbone.model.visual
        del backbone
        torch.cuda.empty_cache()
    encoder = model.model.docseam
    if args.position_feature_mode:
        encoder.pos_feature_mode = args.position_feature_mode
    example = Path(__file__).resolve().parent / "examples/attention"
    image_paths = args.images or [
        example / f"1706.03762v7_page-{page:04d}.jpg" for page in (4, 5, 6)
    ]
    page_numbers = list(range(args.first_page, args.first_page + 3))
    content = []
    for page, path in zip(page_numbers, image_paths):
        content.extend([
            {"type": "text", "text": f"Page {page}:\n"},
            {"type": "image", "image": f"file://{path.resolve()}"},
            {"type": "text", "text": "\n"},
        ])
    content.append({"type": "text", "text":
        f"Read the table continuously across the boundary from Page {page_numbers[0]} "
        f"into Page {page_numbers[1]}, then use Page {page_numbers[2]} only as the following context page."})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # Match the reference visualization's image resizing, including its patch-size default.
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    encoder.record_salience = True
    # Gates depend only on visual features, so skip language decoding.
    with torch.inference_mode():
        vision = model.model.visual(
            inputs.pixel_values.to(model.dtype), grid_thw=inputs.image_grid_thw, return_dict=True
        )
        grids = [(int(h) // 2, int(w) // 2) for _, h, w in inputs.image_grid_thw]
        counts = [h * w for h, w in grids]
        tokens = list(vision.pooler_output.split(counts))
        ink_masks = importlib.import_module(model.__class__.__module__)._ink_masks(
            inputs.pixel_values, inputs.image_grid_thw.tolist(), 2
        )
        encoder(tokens, page_grid_dims=grids, pages_per_document=[3], page_ink_masks=ink_masks)
    selected_records, avg_salience = averaged_center_record(encoder._last_salience_records, 1)
    n_layers = len(selected_records)
    page_grid_dims = grids
    print("Generating center-fusion heatmap...")
    n_prev, n_curr, n_next = selected_records[0]["n_prev"], selected_records[0]["n_curr"], selected_records[0]["n_next"]
    h_prev, w_prev = page_grid_dims[0]
    h_curr, w_curr = page_grid_dims[1]
    h_next, w_next = page_grid_dims[2]

    prev_send_to_next = avg_salience["s_send_to_next"][:n_prev].reshape(h_prev, w_prev)
    curr_recv_from_prev = avg_salience["s_recv_from_prev"][n_prev:n_prev + n_curr].reshape(h_curr, w_curr)
    curr_recv_from_next = avg_salience["s_recv_from_next"][n_prev:n_prev + n_curr].reshape(h_curr, w_curr)
    next_send_to_prev = avg_salience["s_send_to_prev"][n_prev + n_curr:n_prev + n_curr + n_next].reshape(h_next, w_next)

    panels = [
        (0, prev_send_to_next, f"Page {page_numbers[0]} send_to_next", "source for center"),
        (1, curr_recv_from_prev, f"Page {page_numbers[1]} recv_from_prev", "center reads previous"),
        (1, curr_recv_from_next, f"Page {page_numbers[1]} recv_from_next", "center reads next"),
        (2, next_send_to_prev, f"Page {page_numbers[2]} send_to_prev", "source for center"),
    ]
    vmin, vmax = normalize_heatmaps([panel[1] for panel in panels])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    overlay = None
    for col, (img_idx, heatmap, title, subtitle) in enumerate(panels):
        img = Image.open(image_paths[img_idx]).convert("RGB")

        axes[0, col].imshow(img)
        axes[0, col].set_title(title)
        axes[0, col].axis("off")

        axes[1, col].imshow(img)
        relative_heatmap = np.clip((heatmap - vmin) / (vmax - vmin), 0.0, 1.0)
        hm_img = Image.fromarray(relative_heatmap.astype(np.float32)).resize(img.size, Image.BILINEAR)
        overlay = axes[1, col].imshow(np.array(hm_img), cmap="turbo", alpha=0.62, vmin=0.0, vmax=1.0)
        axes[1, col].set_title(subtitle)
        axes[1, col].axis("off")

    fig.colorbar(
        overlay,
        ax=axes[1, :].ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label="relative salience gate strength",
    )

    output_png = Path(args.output)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Heatmap saved to {args.output}")

    stats = {
        "images": [str(path) for path in image_paths],
        "source_pages": page_numbers,
        "center_page": 2,
        "checkpoint": args.model,
        "vision_model": args.vision_model,
        "position_feature_mode": encoder.pos_feature_mode,
        "num_layers_averaged": n_layers,
        "relative_vmin": float(vmin),
        "relative_vmax": float(vmax),
        "panels": [
            salience_stats("prev_send_to_next", prev_send_to_next),
            salience_stats("curr_recv_from_prev", curr_recv_from_prev),
            salience_stats("curr_recv_from_next", curr_recv_from_next),
            salience_stats("next_send_to_prev", next_send_to_prev),
        ],
        "mean_cross_gate_multiplier": {
            "center_to_prev": float(curr_recv_from_prev.mean() * prev_send_to_next.mean()),
            "center_to_next": float(curr_recv_from_next.mean() * next_send_to_prev.mean()),
        },
    }

    output_json = output_png.with_suffix(".json")
    with open(output_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {output_json}")
    for panel in stats["panels"]:
        print(
            f"{panel['name']}: mean={panel['mean']:.4f}, std={panel['std']:.4f}, "
            f"p05={panel['p05']:.4f}, p95={panel['p95']:.4f}"
        )
    print(
        "mean_cross_gate_multiplier: "
        f"center_to_prev={stats['mean_cross_gate_multiplier']['center_to_prev']:.4f}, "
        f"center_to_next={stats['mean_cross_gate_multiplier']['center_to_next']:.4f}"
    )


if __name__ == "__main__":
    main()
