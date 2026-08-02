"""Render an aggregate model-comparison figure without row-level data."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "figures" / "model_comparison.png"


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    models = [
        ("Tabular logistic", 0.639, None, "#64748b"),
        ("Hand-selected causal history", 0.648, None, "#2563eb"),
        ("Linear graph residual", 0.647, None, "#0f766e"),
        ("GraphSAGE (5-seed mean)", 0.636, 0.040, "#7c3aed"),
    ]
    width, height = 1180, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top = 390, 1080, 105
    minimum, maximum = 0.55, 0.70
    draw.text((42, 28), "Validation AUROC: stable models vs neural prototype", fill="#172033", font=font(30, True))
    for tick in [0.55, 0.60, 0.65, 0.70]:
        x = left + (right - left) * (tick - minimum) / (maximum - minimum)
        draw.line((x, top - 15, x, height - 85), fill="#e2e8f0", width=1)
        draw.text((x - 20, height - 70), f"{tick:.2f}", fill="#475569", font=font(16))
    for index, (label, value, error, color) in enumerate(models):
        y = top + index * 78
        draw.text((42, y - 12), label, fill="#334155", font=font(19))
        x_value = left + (right - left) * (value - minimum) / (maximum - minimum)
        draw.rectangle((left, y - 13, x_value, y + 15), fill=color)
        draw.text((x_value + 12, y - 11), f"{value:.3f}", fill="#172033", font=font(17, True))
        if error is not None:
            x_low = left + (right - left) * (max(value - error, minimum) - minimum) / (maximum - minimum)
            x_high = left + (right - left) * (min(value + error, maximum) - minimum) / (maximum - minimum)
            draw.line((x_low, y + 27, x_high, y + 27), fill=color, width=3)
            draw.line((x_low, y + 20, x_low, y + 34), fill=color, width=3)
            draw.line((x_high, y + 20, x_high, y + 34), fill=color, width=3)
            draw.text((x_high + 10, y + 17), "±1 SD", fill="#64748b", font=font(14))
    draw.text((650, 485), "AUROC (validation)", fill="#172033", font=font(19))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
