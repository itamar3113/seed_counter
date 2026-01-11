"""
Module for extracting white discs from images.
This module processes images to isolate white discs by removing the background.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


class DiscExtractor:
    """Extract white discs from images containing seeds."""
    
    def __init__(self, min_disc_area: int = 5000, max_disc_area: int = 500000, debug: bool = False):
        """
        Initialize the DiscExtractor.
        
        Args:
            min_disc_area: Minimum area of a valid disc in pixels
            max_disc_area: Maximum area of a valid disc in pixels
            debug: If True, save debug images showing intermediate processing steps
        """
        self.min_disc_area = min_disc_area
        self.max_disc_area = max_disc_area
        self.debug = debug
    
    def extract_discs(self, image_path: str, output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Extract white discs from an image.
        
        Args:
            image_path: Path to the input image
            output_dir: Optional directory to save extracted disc images
            
        Returns:
            List of extracted disc images as numpy arrays
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to isolate white discs
        # White discs should have high pixel values
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Save debug image: threshold result
        if self.debug and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_thresh_path = os.path.join(output_dir, "debug_threshold.png")
            cv2.imwrite(debug_thresh_path, thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save debug image: original image with all contours
        if self.debug and output_dir:
            debug_contours_image = image.copy()
            cv2.drawContours(debug_contours_image, contours, -1, (0, 255, 0), 2)
            debug_contours_path = os.path.join(output_dir, "debug_all_contours.png")
            cv2.imwrite(debug_contours_path, debug_contours_image)
        
        extracted_discs = []
        disc_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area to get only disc-sized objects
            if self.min_disc_area <= area <= self.max_disc_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create a mask for this disc
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Save debug image: individual disc mask
                if self.debug and output_dir:
                    debug_mask_path = os.path.join(output_dir, f"debug_disc_{disc_count}_mask.png")
                    cv2.imwrite(debug_mask_path, mask)
                
                # Extract the disc region
                disc_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Save debug image: full disc image before cropping
                if self.debug and output_dir:
                    debug_full_disc = disc_image.copy()
                    cv2.rectangle(debug_full_disc, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    debug_full_path = os.path.join(output_dir, f"debug_disc_{disc_count}_full.png")
                    cv2.imwrite(debug_full_path, debug_full_disc)
                
                # Crop to bounding box
                cropped_disc = disc_image[y:y+h, x:x+w]
                
                extracted_discs.append(cropped_disc)
                
                # Optionally save the disc
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"disc_{disc_count}.png")
                    cv2.imwrite(output_path, cropped_disc)
                
                disc_count += 1
        
        return extracted_discs
    
    def extract_disc_with_circle_detection(self, image_path: str, output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Extract white discs using circle detection (Hough Circle Transform).
        This is an alternative method that may work better for circular discs.
        
        Args:
            image_path: Path to the input image
            output_dir: Optional directory to save extracted disc images
            
        Returns:
            List of extracted disc images as numpy arrays
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Save debug image: blurred image
        if self.debug and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_blurred_path = os.path.join(output_dir, "debug_blurred.png")
            cv2.imwrite(debug_blurred_path, blurred)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=400
        )
        
        extracted_discs = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Save debug image: original image with detected circles
            if self.debug and output_dir:
                debug_circles_image = image.copy()
                for circle in circles[0, :]:
                    center_x, center_y, radius = circle
                    cv2.circle(debug_circles_image, (center_x, center_y), radius, (0, 255, 0), 2)
                    cv2.circle(debug_circles_image, (center_x, center_y), 2, (0, 0, 255), 3)
                debug_circles_path = os.path.join(output_dir, "debug_detected_circles.png")
                cv2.imwrite(debug_circles_path, debug_circles_image)
            
            disc_count = 0
            
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                
                # Create a circular mask
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # Save debug image: individual circle mask
                if self.debug and output_dir:
                    debug_mask_path = os.path.join(output_dir, f"debug_circle_{disc_count}_mask.png")
                    cv2.imwrite(debug_mask_path, mask)
                
                # Extract the disc region
                disc_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Crop to bounding box
                x1 = max(0, center_x - radius)
                y1 = max(0, center_y - radius)
                x2 = min(image.shape[1], center_x + radius)
                y2 = min(image.shape[0], center_y + radius)
                
                # Save debug image: full disc image with bounding box
                if self.debug and output_dir:
                    debug_full_disc = disc_image.copy()
                    cv2.rectangle(debug_full_disc, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    debug_full_path = os.path.join(output_dir, f"debug_circle_{disc_count}_full.png")
                    cv2.imwrite(debug_full_path, debug_full_disc)
                
                cropped_disc = disc_image[y1:y2, x1:x2]
                
                extracted_discs.append(cropped_disc)
                
                # Optionally save the disc
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"disc_{disc_count}.png")
                    cv2.imwrite(output_path, cropped_disc)
                
                disc_count += 1
        
        return extracted_discs
