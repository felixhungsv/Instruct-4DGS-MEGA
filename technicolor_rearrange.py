import os
import re
from PIL import Image

def reorganize_and_renumber_frames(train_dir):
    pattern = r'^Painter_undist_(\d{5})_(\d{2})\.png$'

    file_info_list = []  # [(int_frame_idx, int_cam_idx, filename), ...]

    for filename in os.listdir(train_dir):
        match = re.match(pattern, filename)
        if match:
            frame_str = match.group(1)  #  '00050'
            cam_str   = match.group(2)  #  '00'
            frame_idx = int(frame_str)  # 50
            cam_idx   = int(cam_str)    # 0
            file_info_list.append((frame_idx, cam_idx, filename))

    unique_frames = sorted({info[0] for info in file_info_list})
    frame_mapping = {}
    for i, original_frame_idx in enumerate(unique_frames, start=1):
        frame_mapping[original_frame_idx] = i  # 50->1, 51->2, 52->3, ...

    for (original_frame_idx, cam_idx, filename) in file_info_list:
        # cam_idx -> camXX
        cam_folder_name = f'cam{cam_idx+1:02d}'
        cam_folder_path = os.path.join(train_dir, cam_folder_name)
        os.makedirs(cam_folder_path, exist_ok=True)

        new_frame_idx = frame_mapping[original_frame_idx]
        new_frame_str = f'{new_frame_idx:05d}'  # 1->00001, 2->00002 ...

        src_path = os.path.join(train_dir, filename)
        dst_filename = f'frame_{new_frame_str}.jpg'
        dst_path = os.path.join(cam_folder_path, dst_filename)

        with Image.open(src_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(dst_path, 'JPEG', quality=95)

        os.remove(src_path)

if __name__ == '__main__':
    train_folder = './data/multipleview/painter'
    reorganize_and_renumber_frames(train_folder)
