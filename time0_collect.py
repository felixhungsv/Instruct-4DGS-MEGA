import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Collect initial frames from a dataset for 4DGS processing.")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'dynerf')")
parser.add_argument("--scene_name", type=str, required=True, help="Name of the specific scene (e.g., 'cook_spinach')")
args = parser.parse_args()

source_root = f"./data/{args.dataset}/{args.scene_name}"
destination_folder = f"./data/{args.dataset}/time0_{args.scene_name}"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

try:
    cam_folders = sorted([d for d in os.listdir(source_root) if d.startswith('cam') and os.path.isdir(os.path.join(source_root, d))])
except FileNotFoundError:
    print(f"❌ Error: Source path '{source_root}' not found. Please check the path.")
    exit(1)

print(f"Found {len(cam_folders)} 'cam' folders.")

for cam_folder in cam_folders:
    source_file = os.path.join(source_root, cam_folder, 'images', "0000.png")
    
    if os.path.exists(source_file):
        cam_index = int(cam_folder[3:])
        dest_filename = f"original_time0_{cam_index}.png"
        shutil.copy(source_file, os.path.join(destination_folder, dest_filename))
    else:
        print(f"Warning: Source file {source_file} does not exist.")

'''
# Technicolor 
for i in range(0, 20):
    cam_folder = f"cam{i:02d}"
    source_file = os.path.join(source_root, cam_folder, "frame_00001.jpg")
    if os.path.exists(source_file):
        dest_filename = f"{args.scene_name}_{i:04d}.png"
        shutil.copy(source_file, os.path.join(destination_folder, dest_filename))
    else:
        print(f"Warning: Source file {source_file} does not exist.")
'''

print(f"✅ Finished copying images to '{destination_folder}'.")