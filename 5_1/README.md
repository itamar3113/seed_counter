# Test Images Directory - 5_1

This directory is for storing test images that the seed counter should process.

## Usage

Place your test images (white discs with seeds) in this directory, then run:

```bash
python main.py 5_1/your_image.jpg -d
```

The `-d` or `--debug` flag will enable debug mode, which saves intermediate processing images showing:
- Threshold result
- All detected contours
- Individual disc masks
- Full disc images with bounding boxes

## Debug Output

When debug mode is enabled, the following debug images will be saved to the output directory:

### For Contour-based Detection (default):
- `debug_threshold.png` - Binary threshold image showing white disc regions
- `debug_all_contours.png` - Original image with all detected contours outlined
- `debug_disc_N_mask.png` - Individual mask for each disc
- `debug_disc_N_full.png` - Full disc image with bounding box before cropping
- `disc_N.png` - Final cropped disc image

### For Circle-based Detection (`--method circle`):
- `debug_blurred.png` - Blurred image used for circle detection
- `debug_detected_circles.png` - Original image with detected circles
- `debug_circle_N_mask.png` - Individual circular mask for each disc
- `debug_circle_N_full.png` - Full disc image with bounding box before cropping
- `disc_N.png` - Final cropped disc image

## Example

```bash
# Process an image with debug mode enabled
python main.py 5_1/test_image.jpg -d -o 5_1_output

# Process with circle detection and debug mode
python main.py 5_1/test_image.jpg -d -m circle -o 5_1_output
```
