# Blur Generator cover

[cover.svg](cover.svg) is the editable source. [cover.png](cover.png) is the
1280 × 640 PNG used at the top of the README (under 1 MB).

The cover presents the original image next to the project's existing motion,
lens and Gaussian examples. These are existing outputs, not newly run examples:

| Panel | Source |
| --- | --- |
| Original | [sample.jpg](sample.jpg) |
| Motion | [gallery-motion.jpg](gallery-motion.jpg) |
| Lens | [gallery-lens.jpg](gallery-lens.jpg) |
| Gaussian | [gallery-gaussian.jpg](gallery-gaussian.jpg) |

The SVG embeds the original JPEG bytes and scales each image without cropping.
Typography and layout are separate vector elements. All four source files remain
unchanged. Their generation commands are documented in the README and
[make_readme_images.py](../../scripts/make_readme_images.py).

The subtitle mentions depth-aware focus, which is supported by the library;
the four panels show the basic blur types, not depth-map results.

## Export

Open the SVG in a browser or vector editor. Text uses Segoe UI with a sans-serif
fallback, plus Consolas for the install command. Another computer may substitute
fonts; the PNG preserves the reviewed appearance.

With Microsoft Edge installed, render to a scratch directory:

```text
msedge --headless=new --disable-gpu --hide-scrollbars --window-size=1280,640 --screenshot=cover.png file:///absolute/path/to/cover.svg
```

Inspect the full-size image and a 320 × 160 thumbnail before replacing the PNG.
Keep the image below 1 MB and retain the original demo images.
