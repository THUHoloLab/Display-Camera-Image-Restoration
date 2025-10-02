import torch
import torch.nn as nn
import os
import time
import numpy as np
import cv2
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from fvcore.nn import FlopCountAnalysis

# Import the model definition
from models import SmallNet_NAF

def load_checkpoint(model, checkpoint_path, device):
    """
    Loads a model checkpoint.

    Args:
        model (nn.Module): The model to load the weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to load the model onto.

    Returns:
        nn.Module: The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Adapt for DataParallel-saved models
    if any(key.startswith('module.') for key in state_dict.keys()):
        if not isinstance(model, nn.DataParallel):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        if isinstance(model, nn.DataParallel):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model

class ImageDataset(torch.utils.data.Dataset):
    """
    A simple dataset class for loading image pairs from directories.
    """
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        # Load input (assuming .npy format for raw data)
        input_image = np.load(input_path).astype(np.float32)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1) # HWC to CHW

        # Load target (assuming .png format for ground truth)
        target_image = Image.open(target_path).convert('RGB')
        target_image = np.array(target_image).astype(np.float32) / 255.0
        target_image = torch.from_numpy(target_image).permute(2, 0, 1) # HWC to CHW
        
        return input_image, target_image

def compute_psnr(output, target, data_range=1.0):
    """Computes the PSNR for a batch of images."""
    psnr_values = []
    for i in range(output.shape[0]):
        output_image = output[i].cpu().numpy().transpose(1, 2, 0)
        target_image = target[i].cpu().numpy().transpose(1, 2, 0)
        psnr_val = psnr(target_image, output_image, data_range=data_range)
        psnr_values.append(psnr_val)
    return np.mean(psnr_values)

def compute_ssim(output, target, data_range=1.0):
    """Computes the SSIM for a batch of images."""
    ssim_values = []
    for i in range(output.shape[0]):
        output_image = output[i].cpu().numpy().transpose(1, 2, 0)
        target_image = target[i].cpu().numpy().transpose(1, 2, 0)
        ssim_val = ssim(target_image, output_image, data_range=data_range, channel_axis=2)
        ssim_values.append(ssim_val)
    return np.mean(ssim_values)

def save_test_images(output, target, idx, save_dir):
    """Saves the first image of a batch."""
    output_image = (output[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    target_image = (target[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"output_{idx:04d}.png"), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"target_{idx:04d}.png"), cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))

def test_model(model, test_loader, device, save_dir, result_file):
    """
    Tests the model and saves the results.
    """
    model.eval()
    total_psnr, total_ssim, total_l1_loss, total_time = 0, 0, 0, 0
    os.makedirs(save_dir, exist_ok=True)

    with open(result_file, "w") as f:
        f.write("Image_Index, PSNR, SSIM, L1_Loss, Inference_Time(s)\n")
        with torch.no_grad():
            for idx, (input_image, target_image) in enumerate(test_loader):
                input_image = input_image.to(device)
                target_image = target_image.to(device)
                
                start_time = time.time()
                output = model(input_image)
                end_time = time.time()
                
                psnr_val = compute_psnr(output, target_image)
                ssim_val = compute_ssim(output, target_image)
                l1_loss = torch.nn.functional.l1_loss(output, target_image).item()
                time_taken = end_time - start_time

                total_psnr += psnr_val
                total_ssim += ssim_val
                total_l1_loss += l1_loss
                total_time += time_taken

                f.write(f"{idx}, {psnr_val:.4f}, {ssim_val:.4f}, {l1_loss:.6f}, {time_taken:.6f}\n")
                save_test_images(output, target_image, idx, save_dir)

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_time = total_time / len(test_loader)

    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average L1 Loss: {avg_l1_loss:.6f}")
    print(f"Average Inference Time: {avg_time:.6f}s")
    
    with open(result_file, "a") as f:
        f.write(f"\nAverage PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average L1 Loss: {avg_l1_loss:.6f}\n")
        f.write(f"Average Inference Time: {avg_time:.6f}s\n")

def calculate_flops(model, input_size=(1, 4, 960, 540), device='cuda'):
    """Calculates the FLOPs of the model."""
    model = model.to(device)
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    flops = FlopCountAnalysis(model, input_tensor).total()
    return flops

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = SmallNet_NAF().to(device)
    
    # Calculate FLOPs
    print("Calculating FLOPs for SmallNet_NAF...")
    flops = calculate_flops(model, input_size=(1, 4, 960, 540), device=device)
    print(f"SmallNet_NAF FLOPs: {flops / 1e9:.2f} GFLOPs")

    # Paths and configurations
    checkpoint_path = "path/to/your/checkpoint.pth"
    input_dir = "path/to/your/test/input_images"
    target_dir = "path/to/your/test/target_images"
    save_dir = "./test_results/SmallNet_NAF"
    result_file = os.path.join(save_dir, "results.txt")

    # Dataset and DataLoader
    test_dataset = ImageDataset(input_dir, target_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model weights
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Test the model
    test_model(model, test_loader, device, save_dir, result_file)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()