import os
import json
import shutil
import random
import xml.etree.ElementTree as ET

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize coordinates
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    # Clamp to [0.0, 1.0] boundary to be robust
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    
    return (x, y, w, h)

def main():
    # Set random seed for reproducible train/val split
    random.seed(42)
    
    # Directories
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    annotated_images_dir = os.path.join(workspace_dir, "annotated-images")
    splits_file = os.path.join(workspace_dir, "splits.json")
    
    # Output structure
    output_dir = os.path.join(workspace_dir, "pothole_dataset")
    
    print("--- YOLO Dataset Preparation Script ---")
    
    # Load splits
    if not os.path.exists(splits_file):
        print(f"Error: splits.json not found at {splits_file}")
        return
        
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    train_xmls = splits.get("train", [])
    test_xmls = splits.get("test", [])
    
    # Shuffle and split train_xmls into train and val (80% train, 20% val)
    random.shuffle(train_xmls)
    split_idx = int(len(train_xmls) * 0.8)
    
    final_splits = {
        "train": train_xmls[:split_idx],
        "val": train_xmls[split_idx:],
        "test": test_xmls
    }
    
    print(f"Original splits: Train={len(train_xmls)}, Test={len(test_xmls)}")
    print(f"New splits: Train={len(final_splits['train'])}, Val={len(final_splits['val'])}, Test={len(final_splits['test'])}")
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
        
    # Class mapping
    classes = {"pothole": 0}
    
    # Process each split
    for split_name, xml_list in final_splits.items():
        print(f"\nProcessing {split_name} split...")
        count = 0
        for xml_filename in xml_list:
            xml_path = os.path.join(annotated_images_dir, xml_filename)
            jpg_filename = xml_filename.replace(".xml", ".jpg")
            jpg_path = os.path.join(annotated_images_dir, jpg_filename)
            
            if not os.path.exists(xml_path) or not os.path.exists(jpg_path):
                print(f"Warning: Missing file pair for {xml_filename}, skipping.")
                continue
                
            # Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Warning: Failed to parse XML {xml_filename}: {e}, skipping.")
                continue
                
            # Get dimensions
            size_node = root.find("size")
            if size_node is not None:
                width = int(size_node.find("width").text)
                height = int(size_node.find("height").text)
            else:
                print(f"Warning: size tag missing in {xml_filename}, skipping.")
                continue
                
            if width == 0 or height == 0:
                print(f"Warning: Width or height is 0 in {xml_filename}, skipping.")
                continue
                
            # Find and write bounding boxes
            yolo_boxes = []
            for obj in root.findall("object"):
                name_node = obj.find("name")
                if name_node is None:
                    continue
                class_name = name_node.text.strip().lower()
                
                # We map all variants of pothole/potholes to index 0
                if "pothole" in class_name:
                    class_id = classes["pothole"]
                else:
                    # Default/fallback if other class names exist
                    class_id = classes["pothole"]
                    
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    try:
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        
                        # Fix degenerate bounding box (width/height 0)
                        if xmin == xmax:
                            xmax += 1.0
                        if ymin == ymax:
                            ymax += 1.0
                            
                        x_center, y_center, w, h = convert_coordinates((width, height), (xmin, xmax, ymin, ymax))
                        yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    except Exception as e:
                        print(f"Warning: Error parsing boxes in {xml_filename}: {e}")
                        
            # Only write text files and copy images if we successfully parse annotations
            # Note: YOLO supports background images (no labels), but we have a fully annotated dataset
            if yolo_boxes:
                # Copy image
                dest_jpg_path = os.path.join(output_dir, split_name, "images", jpg_filename)
                shutil.copy(jpg_path, dest_jpg_path)
                
                # Write YOLO text file
                txt_filename = xml_filename.replace(".xml", ".txt")
                dest_txt_path = os.path.join(output_dir, split_name, "labels", txt_filename)
                with open(dest_txt_path, "w") as f_out:
                    f_out.write("\n".join(yolo_boxes) + "\n")
                
                count += 1
                
        print(f"Successfully processed {count} / {len(xml_list)} images for {split_name}.")

    # Create dataset.yaml
    yaml_content = f"""# YOLO Pothole Detection Dataset
path: {output_dir}  # dataset root dir
train: train/images  # train images (relative to path)
val: val/images      # val images (relative to path)
test: test/images    # test images (relative to path)

# Classes
names:
  0: pothole
"""
    yaml_path = os.path.join(workspace_dir, "dataset.yaml")
    with open(yaml_path, "w") as f_yaml:
        f_yaml.write(yaml_content)
        
    print(f"\nCreated dataset.yaml configuration file at {yaml_path}")
    print("Dataset preparation successfully completed!")

if __name__ == "__main__":
    main()
