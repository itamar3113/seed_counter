#!/usr/bin/env python3
"""
Main program for seed counter.
This program processes images containing white discs with seeds,
extracts the discs, and counts the seeds on them.
"""

import argparse
import os
import sys
from disc_extractor import DiscExtractor
from seed_counter import SeedCounter


def main():
    """Main entry point for the seed counter program."""
    parser = argparse.ArgumentParser(
        description="Extract white discs from images and count seeds on them"
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image containing white discs with seeds"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save extracted disc images (default: output)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["contour", "circle"],
        default="contour",
        help="Method for disc extraction: contour or circle (default: contour)"
    )
    parser.add_argument(
        "-c", "--count-method",
        choices=["threshold", "blob"],
        default="threshold",
        help="Method for seed counting: threshold or blob (default: threshold)"
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Save annotated images showing detected seeds"
    )
    parser.add_argument(
        "--min-disc-area",
        type=int,
        default=5000,
        help="Minimum disc area in pixels (default: 5000)"
    )
    parser.add_argument(
        "--max-disc-area",
        type=int,
        default=500000,
        help="Maximum disc area in pixels (default: 500000)"
    )
    parser.add_argument(
        "--min-seed-area",
        type=int,
        default=10,
        help="Minimum seed area in pixels (default: 10)"
    )
    parser.add_argument(
        "--max-seed-area",
        type=int,
        default=1000,
        help="Maximum seed area in pixels (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Extract discs from the image
    print(f"Processing image: {args.image_path}")
    print(f"Extraction method: {args.method}")
    
    extractor = DiscExtractor(
        min_disc_area=args.min_disc_area,
        max_disc_area=args.max_disc_area
    )
    
    try:
        if args.method == "circle":
            discs = extractor.extract_disc_with_circle_detection(
                args.image_path,
                args.output_dir
            )
        else:
            discs = extractor.extract_discs(
                args.image_path,
                args.output_dir
            )
        
        print(f"Extracted {len(discs)} disc(s)")
        
        if len(discs) == 0:
            print("Warning: No discs found in the image. Try adjusting the disc area parameters.")
            return 0
        
        # Step 2: Count seeds on each disc
        counter = SeedCounter(
            min_seed_area=args.min_seed_area,
            max_seed_area=args.max_seed_area
        )
        
        print(f"\nCounting method: {args.count_method}")
        print("-" * 50)
        
        total_seeds = 0
        for i, disc in enumerate(discs):
            if args.count_method == "blob":
                count, annotated = counter.count_seeds_blob_detection(disc, args.visualize)
            else:
                count, annotated = counter.count_seeds(disc, args.visualize)
            
            print(f"Disc {i}: {count} seeds")
            total_seeds += count
            
            # Save annotated image if visualization is enabled
            if args.visualize and annotated is not None:
                import cv2
                annotated_path = os.path.join(args.output_dir, f"disc_{i}_annotated.png")
                cv2.imwrite(annotated_path, annotated)
                print(f"  Annotated image saved to: {annotated_path}")
        
        print("-" * 50)
        print(f"\nTotal seeds across all discs: {total_seeds}")
        print(f"\nExtracted disc images saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
