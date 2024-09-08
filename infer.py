import torch
import cv2
from PIL import Image
import numpy as np
from WinCLIP import WinClipAD
from utils.metrics import denormalization

def single_image_inference(model, image_path: str, device: str, resolution: int = 400):
    """
    Run inference on a single image using WinClipAD model.
    
    :param model: Initialized WinClipAD model.
    :param image_path: Path to the input image.
    :param device: The device to run the model on ('cpu' or 'cuda:0').
    :param resolution: The resolution for the image (default is 400).
    :return: Output scores for the image.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((resolution, resolution))

    # Transform the image as required by the model
    transformed_image = model.transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension

    # Move image to device
    transformed_image = transformed_image.to(device)

    # Set the model to eval mode
    model.eval_mode()

    # Run inference
    with torch.no_grad():
        output = model(transformed_image)

    # Post-process output
    output = output.cpu().numpy()  # Convert to numpy
    output = denormalization(output[0])  # Apply denormalization if needed

    return output

# Example usage
def run_inference_on_image(model, image_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model with required arguments
    model = WinClipAD()  # Add your model initialization parameters here
    model = model.to(device)

    # Run inference
    output = single_image_inference(model, image_path, device)
    
    print(f"Output for {image_path}: {output}")
    return output

# Call the function with the model and the path to the image
image_path = 'path_to_your_image.jpg'
run_inference_on_image(model, image_path)
