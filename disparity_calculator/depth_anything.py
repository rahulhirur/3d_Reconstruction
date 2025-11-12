import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from huggingface_hub import hf_hub_download

# --- 1. Configuration and Setup ---

# Define paths and parameters
MODEL_ENCODER = "vitl"
MODEL_REPO_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILENAME = "depth_anything_v2_vitl.pth"

INPUT_IMAGE_PATH = "sample_input.jpg"

OUTPUT_DIR = "depth_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(encoder, repo_id, filename):
    """Instantiates and loads pre-trained weights for the DepthAnythingV2 model."""
    print(f"Loading Depth Anything V2 model ({encoder})...")

    # Instantiate the model (parameters must match the pre-trained config)
    model = DepthAnythingV2(
        encoder=encoder, 
        features=256, 
        out_channels=[256, 512, 1024, 1024]
    )

    # Download the pre-trained weights
    try:
        filepath = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            repo_type="model"
        )
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        print("Ensure you have 'huggingface_hub' installed and the repo/filename are correct.")
        return None

    # Load the weights onto the model and set to evaluation mode
    state_dict = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
    return model

def save_depth_maps(depth, input_path, output_dir):
    """Normalizes the depth map and saves it in 16-bit raw and 8-bit colorized formats."""
    # Use input filename for meaningful output names
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    print("Saving depth maps...")

    # 4a. Save as Raw 16-bit Grayscale PNG
    # This version preserves the most detail (for 3D reconstruction/analysis).
    depth_min = depth.min()
    depth_max = depth.max()
    max_val_16bit = 65535

    # Rescale and convert to 16-bit integer
    if depth_max > depth_min:
        # Scale to 0-1 range, then multiply by max 16-bit value
        depth_normalized_16bit = ((depth - depth_min) / (depth_max - depth_min)) * max_val_16bit
        depth_normalized_16bit = depth_normalized_16bit.astype(np.uint16)
    else:
        # Handle flat depth map case
        depth_normalized_16bit = np.zeros(depth.shape, dtype=np.uint16)

    raw_depth_path = os.path.join(output_dir, f"{base_filename}_raw_16bit.png")
    cv2.imwrite(raw_depth_path, depth_normalized_16bit)
    print(f"  -> Raw 16-bit depth map saved to: {raw_depth_path}")

    # 4b. Save as Colorized Visualization (8-bit)
    # This version is purely for human viewing (uses a color map).
    # Normalize to 0-255 range for color mapping.
    depth_normalized_8bit = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colorized_depth = cv2.applyColorMap(depth_normalized_8bit, cv2.COLORMAP_INFERNO)

    color_depth_path = os.path.join(output_dir, f"{base_filename}_colorized_8bit.png")
    cv2.imwrite(color_depth_path, colorized_depth)
    print(f"  -> Colorized 8-bit depth map saved to: {color_depth_path}")

def main():
    """Main function to orchestrate model loading, inference, and saving."""
    
    # 1. Load Model
    model = load_model(MODEL_ENCODER, MODEL_REPO_ID, MODEL_FILENAME)
    if model is None:
        return

    # 2. Image Reading
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"\nERROR: Input image not found at '{INPUT_IMAGE_PATH}'.")
        print("Please replace 'sample_input.jpg' with the path to your actual image file.")
        return

    print(f"\nReading image from: {INPUT_IMAGE_PATH}")
    raw_img = cv2.imread(INPUT_IMAGE_PATH)

    if raw_img is None:
        print(f"ERROR: Could not read image file at {INPUT_IMAGE_PATH}. Check file permissions or corruption.")
        return

    # 3. Inference
    # The output 'depth' is a HxW numpy array of floating-point values (relative depth).
    with torch.no_grad():
        print("Running depth inference...")
        depth = model.infer_image(raw_img)

    # 4. Saving
    save_depth_maps(depth, INPUT_IMAGE_PATH, OUTPUT_DIR)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
