import os
from pathlib import Path

try:
    import cairosvg
    def _svg_to_png(svg_path: str, png_path: str):
        cairosvg.svg2png(url=svg_path, write_to=png_path)
except OSError:
    try:
        from svglib.svglib import svg2rlg
        # Force reportlab to use its built-in C backend, not rlPyCairo (which needs Cairo DLL)
        from reportlab.lib import rl_config
        rl_config.renderPMBackend = '_renderPM'
        from reportlab.graphics import renderPM
        def _svg_to_png(svg_path: str, png_path: str):
            drawing = svg2rlg(svg_path)
            if drawing is None:
                raise ValueError(f"svglib could not parse {svg_path}")
            renderPM.drawToFile(drawing, png_path, fmt="PNG")
    except (ImportError, OSError):
        raise SystemExit(
            "No SVG backend found. Run:  pip install svglib reportlab\n"
            "Or install the GTK3 runtime for Cairo support on Windows:\n"
            "  https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer"
        )


def convert_and_cleanup_svg(base_folder):
    base_path = Path(base_folder)

    if not base_path.exists():
        print(f"Error: The folder '{base_folder}' does not exist.")
        return

    print(f"Starting conversion and cleanup in: {base_path.absolute()}")

    count = 0
    # Walk through all directories and files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".svg"):
                svg_path = Path(root) / file
                png_path = svg_path.with_suffix(".png")

                try:
                    # 1. Perform the conversion
                    _svg_to_png(str(svg_path), str(png_path))

                    # 2. Delete the original SVG only if conversion succeeded
                    svg_path.unlink()

                    print(f"Converted & deleted: {svg_path.name}")
                    count += 1
                except Exception as e:
                    print(f"Failed to process {svg_path.name}: {e}")

    print(f"\nDone. Processed {count} files.")


if __name__ == "__main__":
    target_folder = input("Enter the base folder path (or press Enter for current directory): ").strip() or "."
    convert_and_cleanup_svg(target_folder)