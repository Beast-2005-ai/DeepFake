import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image

class VideoFrameDataset(Dataset):
    """
    A custom PyTorch Dataset for loading sequences of frames from video files.
    This class is the heart of our data pipeline for Phase 2.
    """
    def __init__(self, data_dir, transform=None, sequence_length=30, max_videos_per_class=None):
        """
        Args:
            data_dir (string): Path to the directory with 'real' and 'fake' subdirectories.
            transform (callable, optional): Transformations to be applied to each frame.
            sequence_length (int): The number of frames to sample from each video.
            max_videos_per_class (int, optional): Maximum number of videos to load per class, for quick testing.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = self._gather_samples(max_videos_per_class)

    def _gather_samples(self, max_videos_per_class):
        """Finds all video files and associates them with a label."""
        samples = []
        # The labels (0, 1) are assigned based on the alphabetical order of folder names ('fake', 'real')
        # To be explicit: 0 = fake, 1 = real
        for label, class_name in enumerate(['fake', 'real']):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found for class '{class_name}' at {class_dir}")
                continue
            
            video_count = 0
            for video_file in sorted(os.listdir(class_dir)):
                if video_file.endswith('.mp4'):
                    if max_videos_per_class and video_count >= max_videos_per_class:
                        break
                    video_path = os.path.join(class_dir, video_file)
                    samples.append((video_path, label))
                    video_count += 1
        return samples

    def __len__(self):
        """Returns the total number of videos in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches a sequence of frames from a video at a given index.
        """
        video_path, label = self.samples[idx]
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < self.sequence_length:
                # If video is too short, we'll try the next one in the list
                cap.release()
                return self.__getitem__((idx + 1) % len(self.samples))

            # Randomly select a starting frame for our sequence
            start_frame = np.random.randint(0, total_frames - self.sequence_length + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    # If reading fails midway, it's safer to skip to the next sample
                    cap.release()
                    return self.__getitem__((idx + 1) % len(self.samples))
                
                # Convert frame from BGR (OpenCV) to RGB (PIL) for standard transforms
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                if self.transform:
                    pil_image = self.transform(pil_image)
                
                frames.append(pil_image)
            
            cap.release()
            
            # Stack the frames into a single tensor of shape: (sequence_length, C, H, W)
            frames_tensor = torch.stack(frames)
            
            return frames_tensor, label
        
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Fallback to the next sample if there's a critical error
            return self.__getitem__((idx + 1) % len(self.samples))
