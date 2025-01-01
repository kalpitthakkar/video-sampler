import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import flow_to_image
from tqdm import tqdm

from optflow import RAFTOpticalFlow, VideoDatasetTorchVision


# TODO: Maybe this will require streaming reads in the future
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _setup_flow_model(name='raft_large', device=None):
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')

    if name == 'raft_large':
        return RAFTOpticalFlow(device=device)
    else:
        raise ValueError(f'Unknown model name: {name}')
    
    
def _setup_video_dataset(video_path):
    dt = VideoDatasetTorchVision(video_path)
    return DataLoader(dt, batch_size=8, shuffle=False, num_workers=4)


def compute_optical_flow(video_path, device=None):
    flow_model = _setup_flow_model(device=device)
    video_dataset = _setup_video_dataset(video_path)

    flows = []
    for sample1, sample2 in tqdm(video_dataset):
        sample1 = sample1[0].to(device)
        sample2 = sample2[0].to(device)
        
        with torch.no_grad():
            flow = flow_model(sample1, sample2)
            # RAFT uses a RNN to predict the flow, so we need to extract the last flow
            flows.append(flow[-1])

    return flows


def save_optical_flow(flows, output_dir, bs=8):
    os.makedirs(output_dir, exist_ok=True)
    for i, flow_b in enumerate(flows):
        for j, flow in enumerate(flow_b):
            curr_idx = i * bs + j
            flow_image = flow_to_image(flow).permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(os.path.join(output_dir, f'flow_{curr_idx:05d}.png'), flow_image)
            flow = flow.cpu().numpy()
            np.save(os.path.join(output_dir, f'flow_{curr_idx:05d}.npy'), flow)
    
    cmd = [
        "ffmpeg", "-framerate", "3", "-i", f"{output_dir}/flow_%05d.png",
        "-c:v", "libx264", f"{output_dir}/optflow.mp4"
    ]
    _ = subprocess.run(cmd)
    
    
def extract_motion_segments(flows, threshold=0.5):
    motion_values = []
    all_flow = torch.cat(flows, dim=0)
    all_motion = torch.mean(all_flow)
    win_size = 5
    for idx in range(win_size // 2, len(all_flow) - win_size // 2):
        flow = all_flow[idx - win_size // 2:idx + win_size // 2]
        motion = torch.mean(flow)
        motion_values.append(motion)
   
    motion_flags = [0.1] * (win_size // 2)
    for idx in range(len(motion_values)):
        if motion_values[idx] > all_motion:
            motion_flags.append(0.9)
        else:
            motion_flags.append(0.1)
    motion_flags += [0.1] * (win_size // 2)
    
    motion_segments = []
    last_idx = 0
    for idx in range(len(motion_flags) - 1):
        if motion_flags[idx] == 0.1 and motion_flags[idx + 1] == 0.9:
            motion_segments.append((last_idx, idx, "slow"))
            last_idx = idx + 1
        elif motion_flags[idx] == 0.9 and motion_flags[idx + 1] == 0.1:
            motion_segments.append((last_idx, idx, "fast"))
            last_idx = idx + 1
    if motion_flags[-1] == 0.9:
        motion_segments.append((last_idx, len(motion_flags) - 1, "fast"))
    else:
        motion_segments.append((last_idx, len(motion_flags) - 1, "slow"))
    
    motion_values = torch.stack(motion_values).cpu().numpy()
    all_motion = all_motion.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    x = np.arange(len(motion_values))
    ax.plot(x, motion_values)
    ax.plot(x, motion_flags[2:-2])
    ax.axhline(y=all_motion, color='r', linestyle='--')
    plt.savefig('motion_values.png')
    plt.close(fig)
            
    return motion_segments


def sample_k_frames_from_segments(motion_segments, k=(0.3, 0.7), regime='random'):
    sampled_frame_indices = []
    total_frames = sum([
        end_idx - start_idx + 1 for start_idx, end_idx, _ in motion_segments
    ])
    for segment in motion_segments:
        start_idx, end_idx, motion_type = segment
        if motion_type == 'slow':
            K = k[0]
        else:
            K = k[1]
        if regime == 'random':
            idxs = np.random.choice(
                range(start_idx, end_idx + 1),
                int(K * (end_idx - start_idx + 1)),
                replace=False
            )
            sampled_frame_indices += idxs.tolist()
        elif regime == 'uniform':
            num_samples = int((end_idx - start_idx + 1) * K)
            sampled_frame_indices += np.linspace(
                start_idx, end_idx, num_samples, dtype=int
            ).tolist()
        else:
            raise ValueError(f'Unknown regime: {regime}')

    return sampled_frame_indices, total_frames

def main(args):
    device = torch.device(args.device)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    
    if os.path.exists(args.output_dir):
        flows = []
        files = [x for x in os.listdir(args.output_dir) if x.endswith('.npy')]
        files = sorted(files)
        for file in files:
            flow = np.load(os.path.join(args.output_dir, file))
            flow = torch.from_numpy(flow)
            flows.append(flow)
    else:
        flows = compute_optical_flow(args.video_path, device=device)
    
    motion_segments = extract_motion_segments(flows)
    sampled_frame_indices, total_frames = sample_k_frames_from_segments(
        motion_segments, k=(args.slow_sampling_rate, args.fast_sampling_rate),
        regime=args.regime
    )
    print(
        f'Sampled {len(sampled_frame_indices)} frames out of {total_frames}' +
        f' ({len(sampled_frame_indices) / total_frames * 100:.2f}%)'
    )
    
    if args.save:
        save_optical_flow(flows, args.output_dir)
        np.save(
            os.path.join(args.output_dir, 'sampled_frame_indices.npy'),
            sampled_frame_indices
        )
        np.save(
            os.path.join(args.output_dir, 'motion_segments.npy'),
            motion_segments
        )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--slow-sampling-rate', type=float, default=0.3)
    parser.add_argument('--fast-sampling-rate', type=float, default=0.7)
    parser.add_argument('--regime', type=str, default='random')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"output_{os.path.splitext(args.video_path)[0]}"
    
    main(args)