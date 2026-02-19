"""
Combines multiple part images into a single combined image for each case folder.

"""
import os
import glob
from PIL import Image

def combine_pdf_parts():
    base_path = "/hspshare/converted_images"
    
    # Go through all case folders
    for case_folder in os.listdir(base_path):
        case_path = os.path.join(base_path, case_folder)
        
        if not os.path.isdir(case_path):
            continue
            
        # Check if main combined file already exists
        main_combined = os.path.join(case_path, f"{case_folder}_combined.png")
        if os.path.exists(main_combined):
            continue
            
        # Look for part files
        part_pattern = os.path.join(case_path, f"{case_folder}_combined_part_*.png")
        part_files = sorted(glob.glob(part_pattern))
        
        if len(part_files) >= 1:  # Changed from > 1 to >= 1
            # Load all part images
            images = []
            total_height = 0
            max_width = 0
            
            for part_file in part_files:
                img = Image.open(part_file)
                images.append(img)
                total_height += img.height
                max_width = max(max_width, img.width)
            
            # Create combined image
            combined = Image.new('RGB', (max_width, total_height), 'white')
            y_offset = 0
            
            for img in images:
                combined.paste(img, (0, y_offset))
                y_offset += img.height
            
            # Save combined image
            combined.save(main_combined)
            print(f"Created combined image: {main_combined}")
            
            # Delete part files
            for part_file in part_files:
                # os.remove(part_file)
                print(f"Deleted: {part_file}")
            
            # Close images
            for img in images:
                img.close()
            combined.close()

if __name__ == "__main__":
    combine_pdf_parts()