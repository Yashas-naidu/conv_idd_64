import os
import glob
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class VKITTISequenceDataset(data.Dataset):
    def __init__(self, sequence_folders: List[str], n_frames_input: int = 2, n_frames_output: int = 1,
                 frame_stride: int = 5, target_size: int = 512, include_sequence_info: bool = True,
                 motion_threshold: float = 0.01):
        super(VKITTISequenceDataset, self).__init__()

        self.sequence_folders = sequence_folders
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.frame_stride = frame_stride
        self.n_frames_total = (n_frames_input - 1) * frame_stride + frame_stride + 1
        self.target_size = target_size
        self.include_sequence_info = include_sequence_info
        self.motion_threshold = motion_threshold

        self.sequences = self._process_sequences()
        self.index_mapping = self._create_index_mapping()
        self.length = len(self.index_mapping)

    def _detect_motion(self, frames: List[str]) -> bool:
        small_size = (64, 64)
        total_pixels = small_size[0] * small_size[1]

        prev = cv2.imread(frames[0])
        if prev is None:
            return False
        prev = cv2.resize(prev, small_size)
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            cur = cv2.imread(frames[i])
            if cur is None:
                return False
            cur = cv2.resize(cur, small_size)
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev, cur)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            changed = np.sum(thresh) / 255.0
            if (changed / total_pixels) > self.motion_threshold:
                return True
            prev = cur
        return False

    def _process_sequences(self):
        sequences = []
        for seq_folder in self.sequence_folders:
            scene = seq_folder.split(os.sep)[-5] if len(seq_folder.split(os.sep)) >= 5 else "scene"
            variant = seq_folder.split(os.sep)[-4] if len(seq_folder.split(os.sep)) >= 4 else "variant"
            camera = os.path.basename(seq_folder)

            # Collect frames (support .jpg/.jpeg/.png)
            frame_files = []
            for pat in ("*.jpg", "*.jpeg", "*.png"):
                frame_files.extend(glob.glob(os.path.join(seq_folder, pat)))
            frame_files = sorted(frame_files)

            if len(frame_files) >= self.n_frames_total:
                if self._detect_motion(frame_files):
                    sequences.append({
                        'folder': seq_folder,
                        'frames': frame_files,
                        'scene': scene,
                        'variant': variant,
                        'camera': camera,
                        'name': f"{scene}_{variant}_{camera}"
                    })
                else:
                    print(f"Skipping sequence {seq_folder} due to insufficient motion")
            else:
                print(f"Warning: Skipping {seq_folder} with only {len(frame_files)} frames (need {self.n_frames_total})")

        print(f"Found {len(sequences)} VKITTI sequences with {self.n_frames_total}+ frames and sufficient motion")
        return sequences

    def _create_index_mapping(self):
        index_mapping = []
        for seq_idx, sequence in enumerate(self.sequences):
            n_frames = len(sequence['frames'])
            for frame_idx in range(n_frames - self.n_frames_total + 1):
                index_mapping.append((seq_idx, frame_idx))
        print(f"Created {len(index_mapping)} valid VKITTI frame sequences")
        return index_mapping

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.target_size, self.target_size))
        frame = frame.astype(np.float32) / 255.0
        return frame

    def __getitem__(self, idx: int):
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")

        seq_idx, frame_idx = self.index_mapping[idx]
        sequence = self.sequences[seq_idx]

        frames = []
        for i in range(self.n_frames_input + self.n_frames_output):
            pos = frame_idx + i * self.frame_stride
            path = sequence['frames'][pos]
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = self._preprocess_frame(img)
            frames.append(img)

        seq_np = np.array(frames)
        input_frames = seq_np[:self.n_frames_input]
        output_frames = seq_np[self.n_frames_input:]

        frozen = input_frames[-1].copy()

        input_tensor = torch.from_numpy(input_frames).permute(0, 3, 1, 2).contiguous().float()
        output_tensor = torch.from_numpy(output_frames).permute(0, 3, 1, 2).contiguous().float()

        if self.include_sequence_info:
            info = {
                'sequence_name': sequence['name'],
                'scene': sequence['scene'],
                'variant': sequence['variant'],
                'camera': sequence['camera'],
                'frame_idx': frame_idx,
                'frame_paths': [sequence['frames'][frame_idx + i * self.frame_stride]
                                for i in range(self.n_frames_input + self.n_frames_output)]
            }
            return [idx, output_tensor, input_tensor, frozen, info]

        return [idx, output_tensor, input_tensor, frozen, np.zeros(1)]

    def __len__(self):
        return self.length


def find_all_vkitti_sequence_folders(dataset_root: str) -> List[str]:
    """
    Locate all camera folders containing frames in VKITTI structure, e.g.:
    <root>/Scene01/<variant>/frames/rgb/Camera_0
    """
    sequence_folders = []
    if not os.path.isdir(dataset_root):
        print(f"VKITTI root not found: {dataset_root}")
        return sequence_folders

    for scene in os.listdir(dataset_root):
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            continue
        for variant in os.listdir(scene_path):
            var_path = os.path.join(scene_path, variant, 'frames', 'rgb')
            if not os.path.isdir(var_path):
                continue
            for cam in os.listdir(var_path):
                cam_path = os.path.join(var_path, cam)
                if os.path.isdir(cam_path):
                    sequence_folders.append(cam_path)

    print(f"Found {len(sequence_folders)} VKITTI camera folders")
    return sequence_folders


def create_vkitti_datasets(
    dataset_root: str,
    n_frames_input: int = 2,
    n_frames_output: int = 1,
    frame_stride: int = 5,
    target_size: int = 512,
    train_split_ratio: float = 0.8,
    seed: int = None,
    motion_threshold: float = 0.01
) -> Tuple[VKITTISequenceDataset, VKITTISequenceDataset]:
    random.seed(seed)

    all_cam_folders = find_all_vkitti_sequence_folders(dataset_root)
    random.shuffle(all_cam_folders)

    split_idx = int(len(all_cam_folders) * train_split_ratio)
    train_sequences = all_cam_folders[:split_idx]
    val_sequences = all_cam_folders[split_idx:]

    print(f"VKITTI train sequences: {len(train_sequences)}")
    print(f"VKITTI validation sequences: {len(val_sequences)}")

    train_dataset = VKITTISequenceDataset(
        sequence_folders=train_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        include_sequence_info=True,
        motion_threshold=motion_threshold,
    )

    val_dataset = VKITTISequenceDataset(
        sequence_folders=val_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        include_sequence_info=True,
        motion_threshold=motion_threshold,
    )

    return train_dataset, val_dataset


if __name__ == '__main__':
    # Example usage: quick sanity check
    root = "/mnt/local/gs_datasets/vkitti"
    train_ds, val_ds = create_vkitti_datasets(
        dataset_root=root,
        n_frames_input=2,
        n_frames_output=1,
        frame_stride=5,
        target_size=512,
        train_split_ratio=0.8,
        motion_threshold=0.05,
    )

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Optional: build loaders
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    for i, (idx, targetVar, inputVar, frozen, info) in enumerate(train_loader):
        print("Sample shapes:", inputVar.shape, targetVar.shape)
        break


