import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize


class RAFTOpticalFlow(nn.Module):
    def __init__(self, device):
        super(RAFTOpticalFlow, self).__init__()
        self.optical_flow = raft_large(pretrained=True, progress=False)
        self.optical_flow.to(device)
        self.optical_flow.eval()

    def forward(self, left, right):
        return self.optical_flow(left, right)
    
    
class VideoDatasetTorchVision(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.transform = Compose(
            [
                ConvertImageDtype(torch.float32),
                Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1],
                Resize((720, 1280)),
            ]
        )
        
        self.data, _, _ = read_video(video_path)
        self.data = self.data.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        sample1 = self.data[idx]
        sample2 = self.data[idx + 1]

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        
        sample1 = (sample1, idx)
        sample2 = (sample2, idx + 1)

        return sample1, sample2