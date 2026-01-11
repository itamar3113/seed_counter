# Seed Counter

A Python program for processing images of white discs with seeds. The program extracts disc images from a larger image (removing background) and then counts the seeds on each disc.

## Features

- **Disc Extraction**: Isolates white discs from background using two methods:
  - Contour-based detection (default)
  - Circle detection using Hough Circle Transform
  
- **Seed Counting**: Counts seeds on extracted discs using two methods:
  - Adaptive threshold-based detection (default)
  - Blob detection

- **Visualization**: Optional annotated images showing detected seeds

- **Configurable Parameters**: Adjustable area thresholds for both discs and seeds

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/itamar3113/seed_counter.git
cd seed_counter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py path/to/your/image.jpg
```

This will:
1. Extract white discs from the image
2. Count seeds on each disc
3. Save extracted disc images to the `output` directory

### Advanced Usage

```bash
python main.py path/to/your/image.jpg \
  --output-dir results \
  --method circle \
  --count-method blob \
  --visualize \
  --debug \
  --min-disc-area 10000 \
  --max-disc-area 300000 \
  --min-seed-area 20 \
  --max-seed-area 800
```

### Debug Mode

Enable debug mode with the `-d` or `--debug` flag to save intermediate processing images that help understand and troubleshoot the disc extraction process:

```bash
python main.py path/to/your/image.jpg --debug
```

Debug mode saves the following images to the output directory:

**For contour-based detection (default):**
- `debug_threshold.png` - Binary threshold image showing white disc regions
- `debug_all_contours.png` - Original image with all detected contours outlined
- `debug_disc_N_mask.png` - Individual mask for each disc
- `debug_disc_N_full.png` - Full disc image with bounding box before cropping

**For circle-based detection (`--method circle`):**
- `debug_blurred.png` - Blurred image used for circle detection
- `debug_detected_circles.png` - Original image with detected circles
- `debug_circle_N_mask.png` - Individual circular mask for each disc
- `debug_circle_N_full.png` - Full disc image with bounding box before cropping

Debug mode is particularly useful when:
- Discs are not being detected correctly
- You need to tune the area parameters
- You want to understand how the extraction process works

### Command-Line Options

- `image_path`: Path to the input image (required)
- `-o, --output-dir`: Directory to save extracted disc images (default: output)
- `-m, --method`: Disc extraction method - `contour` or `circle` (default: contour)
- `-c, --count-method`: Seed counting method - `threshold` or `blob` (default: threshold)
- `-v, --visualize`: Save annotated images showing detected seeds
- `-d, --debug`: Save debug images showing disc extraction process
- `--min-disc-area`: Minimum disc area in pixels (default: 5000)
- `--max-disc-area`: Maximum disc area in pixels (default: 500000)
- `--min-seed-area`: Minimum seed area in pixels (default: 10)
- `--max-seed-area`: Maximum seed area in pixels (default: 1000)

### Output

The program will:
1. Display the number of discs found
2. Show the seed count for each disc
3. Display the total seed count across all discs
4. Save extracted disc images to the output directory
5. If `--visualize` is used, save annotated images showing detected seeds

Example output:
```
Processing image: seeds.jpg
Extraction method: contour
Extracted 3 disc(s)

Counting method: threshold
--------------------------------------------------
Disc 0: 45 seeds
  Annotated image saved to: output/disc_0_annotated.png
Disc 1: 52 seeds
  Annotated image saved to: output/disc_1_annotated.png
Disc 2: 38 seeds
  Annotated image saved to: output/disc_2_annotated.png
--------------------------------------------------

Total seeds across all discs: 135

Extracted disc images saved to: output
```

## How It Works

### Disc Extraction

1. **Contour Method** (default):
   - Converts image to grayscale
   - Applies Gaussian blur to reduce noise
   - Uses binary threshold to isolate white regions
   - Finds contours and filters by area
   - Extracts each disc using contour masks

2. **Circle Method**:
   - Converts image to grayscale
   - Applies Gaussian blur
   - Uses Hough Circle Transform to detect circular discs
   - Extracts each disc using circular masks

### Seed Counting

1. **Threshold Method** (default):
   - Converts disc image to grayscale
   - Applies adaptive thresholding to detect darker seeds
   - Uses morphological operations to clean up detection
   - Finds and counts seed contours filtered by area

2. **Blob Method**:
   - Detects seeds as blob features
   - Filters by area, circularity, convexity, and inertia
   - More robust for certain seed types

## Python API

You can also use the modules programmatically:

```python
from disc_extractor import DiscExtractor
from seed_counter import SeedCounter

# Extract discs
extractor = DiscExtractor(min_disc_area=5000, max_disc_area=500000)
discs = extractor.extract_discs("image.jpg", output_dir="output")

# Count seeds
counter = SeedCounter(min_seed_area=10, max_seed_area=1000)
for i, disc in enumerate(discs):
    count, annotated = counter.count_seeds(disc, visualize=True)
    print(f"Disc {i}: {count} seeds")
```

## Tips for Best Results

1. **Disc Extraction**:
   - Ensure good contrast between white discs and background
   - Adjust `--min-disc-area` and `--max-disc-area` based on your image resolution
   - Try `--method circle` if discs are perfectly circular
   - Use `--method contour` if discs have irregular shapes

2. **Seed Counting**:
   - Adjust `--min-seed-area` and `--max-seed-area` based on seed size
   - Try `--count-method blob` if seeds are round and well-separated
   - Use `--count-method threshold` for irregularly shaped seeds
   - Use `--visualize` to see what's being detected and tune parameters

3. **Image Quality**:
   - Use high-resolution images for better accuracy
   - Ensure good lighting with minimal shadows
   - Keep the camera perpendicular to the discs for best results

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.