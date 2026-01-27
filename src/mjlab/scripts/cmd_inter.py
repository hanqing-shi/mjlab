"""
Script to convert command.csv to command.npz with FPS resampling.
Logic extracted from mjlab/scripts/csv_to_npz.py
"""
import numpy as np
import torch
import tyro
from pathlib import Path

class CommandResampler:
    def __init__(self, csv_file: str, input_fps: float, output_fps: float):
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.device = "cpu"  # Data processing can be done on CPU
        
        # 1. Load Data
        print(f"Loading {csv_file}...")
        raw_data = np.loadtxt(csv_file, delimiter=",")
        if raw_data.ndim == 1:
            raw_data = raw_data[:, None] # Handle 1D case
            
        self.data_input = torch.from_numpy(raw_data).float().to(self.device)
        self.input_frames = self.data_input.shape[0]
        self.dims = self.data_input.shape[1]
        
        # Calculate duration based on (N-1) * dt, same as csv_to_npz.py
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.duration = (self.input_frames - 1) * self.input_dt

    def resample(self) -> np.ndarray:
        # 2. Generate Target Times
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        output_frames = times.shape[0]
        
        # 3. Compute Blend Weights (Exact logic from MotionLoader._compute_frame_blend)
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        # Clamp index_1 to avoid out of bounds
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        
        # 4. Linear Interpolation (Lerp)
        # Expand blend for broadcasting: [T] -> [T, 1]
        blend = blend.unsqueeze(1)
        
        val_0 = self.data_input[index_0]
        val_1 = self.data_input[index_1]
        
        interpolated_data = val_0 * (1 - blend) + val_1 * blend
        
        print(f"Resampling complete:")
        print(f"  Input: {self.input_frames} frames @ {self.input_fps} Hz")
        print(f"  Output: {output_frames} frames @ {self.output_fps} Hz")
        print(f"  Shape: {interpolated_data.shape}")
        
        return interpolated_data.numpy()

def main(
    input_csv: str = "command.csv",
    output_npz: str = "/tmp/command.npz",
    input_fps: float = 30.0,
    output_fps: float = 50.0, # Standard MujocoLab control freq (dt=0.02)
):
    converter = CommandResampler(input_csv, input_fps, output_fps)
    result_data = converter.resample()
    
    # Save to NPZ using a generic key 'data'
    # This matches the "blind read" logic we discussed
    print(f"Saving to {output_npz}...")
    np.savez(output_npz, data=result_data)
    print("Done!")

if __name__ == "__main__":
    tyro.cli(main)