#!/usr/bin/env python3
"""
Example script demonstrating how to use the seed counter API programmatically.
"""

import os
import cv2
from disc_extractor import DiscExtractor
from seed_counter import SeedCounter


def example_basic_usage(image_path: str):
    """
    Basic example: Extract discs and count seeds.
    
    Args:
        image_path: Path to the input image
    """
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract discs
    print(f"\nExtracting discs from {image_path}...")
    extractor = DiscExtractor()
    discs = extractor.extract_discs(image_path, output_dir=output_dir)
    print(f"Found {len(discs)} disc(s)")
    
    # Step 2: Count seeds on each disc
    print("\nCounting seeds...")
    counter = SeedCounter()
    
    for i, disc in enumerate(discs):
        count, _ = counter.count_seeds(disc)
        print(f"  Disc {i}: {count} seeds")
    
    print(f"\nResults saved to: {output_dir}/")


def example_with_visualization(image_path: str):
    """
    Example with visualization: Show detected seeds on annotated images.
    
    Args:
        image_path: Path to the input image
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = "example_output_viz"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract discs
    print(f"\nExtracting discs from {image_path}...")
    extractor = DiscExtractor()
    discs = extractor.extract_discs(image_path, output_dir=output_dir)
    print(f"Found {len(discs)} disc(s)")
    
    # Count seeds with visualization
    print("\nCounting seeds with visualization...")
    counter = SeedCounter()
    
    for i, disc in enumerate(discs):
        count, annotated = counter.count_seeds(disc, visualize=True)
        print(f"  Disc {i}: {count} seeds")
        
        # Save annotated image
        if annotated is not None:
            annotated_path = os.path.join(output_dir, f"disc_{i}_annotated.png")
            cv2.imwrite(annotated_path, annotated)
            print(f"    Annotated image: {annotated_path}")
    
    print(f"\nResults saved to: {output_dir}/")


def example_custom_parameters(image_path: str):
    """
    Example with custom parameters for disc and seed detection.
    
    Args:
        image_path: Path to the input image
    """
    print("\n" + "=" * 60)
    print("CUSTOM PARAMETERS EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = "example_output_custom"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract discs with custom size parameters
    print(f"\nExtracting discs with custom parameters...")
    extractor = DiscExtractor(
        min_disc_area=10000,  # Larger minimum disc size
        max_disc_area=300000  # Smaller maximum disc size
    )
    discs = extractor.extract_discs(image_path, output_dir=output_dir)
    print(f"Found {len(discs)} disc(s)")
    
    # Count seeds with custom size parameters
    print("\nCounting seeds with custom parameters...")
    counter = SeedCounter(
        min_seed_area=20,    # Larger minimum seed size
        max_seed_area=800    # Smaller maximum seed size
    )
    
    for i, disc in enumerate(discs):
        count, _ = counter.count_seeds(disc)
        print(f"  Disc {i}: {count} seeds")
    
    print(f"\nResults saved to: {output_dir}/")


def example_alternative_methods(image_path: str):
    """
    Example using alternative detection methods.
    
    Args:
        image_path: Path to the input image
    """
    print("\n" + "=" * 60)
    print("ALTERNATIVE METHODS EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = "example_output_alt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try circle-based disc extraction
    print(f"\nExtracting discs using circle detection...")
    extractor = DiscExtractor()
    discs = extractor.extract_disc_with_circle_detection(
        image_path,
        output_dir=output_dir
    )
    print(f"Found {len(discs)} disc(s)")
    
    # Try blob-based seed counting
    print("\nCounting seeds using blob detection...")
    counter = SeedCounter()
    
    for i, disc in enumerate(discs):
        count, _ = counter.count_seeds_blob_detection(disc)
        print(f"  Disc {i}: {count} seeds")
    
    print(f"\nResults saved to: {output_dir}/")


def main():
    """Main function to run all examples."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example.py <path_to_image>")
        print("\nThis script demonstrates various ways to use the seed counter API.")
        print("Provide a path to an image containing white discs with seeds.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Run examples
    try:
        example_basic_usage(image_path)
        example_with_visualization(image_path)
        example_custom_parameters(image_path)
        example_alternative_methods(image_path)
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
