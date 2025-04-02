#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
from PIL import Image
import sys
from pathlib import Path

# Class map based on the XML files
# Update this if needed to match your actual classes
class_map = {
    "With Helmet": 0,
    "Without Helmet": 1
}

def convert_voc_to_yolo(image_dir, annotation_dir, label_output_dir):
    """
    Convert VOC format XML annotations to YOLO format TXT files
    
    Args:
        image_dir: Directory containing the images
        annotation_dir: Directory containing the XML annotations
        label_output_dir: Directory where the TXT labels will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Get list of XML files
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    total_files = len(xml_files)
    
    print(f"Found {total_files} XML files to convert")
    
    # Counter for successful conversions
    converted_count = 0
    skipped_count = 0
    
    for i, xml_file in enumerate(xml_files):
        try:
            # Parse XML file
            xml_path = os.path.join(annotation_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image filename and dimensions
            image_filename = root.find("filename").text
            img_path = os.path.join(image_dir, image_filename)
            
            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping {xml_file}")
                skipped_count += 1
                continue
            
            # Get image dimensions
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                skipped_count += 1
                continue
            
            # Create output filename (replace extension with .txt)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(label_output_dir, label_filename)
            
            # Process annotations
            with open(label_path, "w") as f:
                objects_found = 0
                
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    
                    # Check if class is in our class map
                    if class_name not in class_map:
                        print(f"Warning: Unknown class '{class_name}' in {xml_file}, skipping object")
                        continue
                    
                    # Get class ID from map
                    class_id = class_map[class_name]
                    
                    # Get bounding box coordinates
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = (xmin + xmax) / 2.0 / img_w
                    y_center = (ymin + ymax) / 2.0 / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    
                    # Write to file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    objects_found += 1
            
            # If no objects were found or written, remove the empty file
            if objects_found == 0:
                os.remove(label_path)
                print(f"Warning: No valid objects found in {xml_file}, skipping file")
                skipped_count += 1
            else:
                converted_count += 1
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"Progress: {i + 1}/{total_files} files processed")
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            skipped_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Skipped/Failed: {skipped_count} files")
    
    return converted_count, skipped_count

if __name__ == "__main__":
    # Set paths relative to the script location
    script_dir = Path(__file__).parent.absolute()
    base_dir = os.path.join(script_dir, "dataset")
    
    image_dir = os.path.join(base_dir, "images")
    annotation_dir = os.path.join(base_dir, "annotations")
    label_output_dir = os.path.join(base_dir, "labels")
    
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Label output directory: {label_output_dir}")
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found")
        sys.exit(1)
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory {annotation_dir} not found")
        sys.exit(1)
    
    # Run conversion
    print("Starting conversion...")
    convert_voc_to_yolo(image_dir, annotation_dir, label_output_dir)
