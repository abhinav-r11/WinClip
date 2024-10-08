import torch
import cv2
from PIL import Image
import numpy as np
from WinCLIP import WinClipAD
import argparse
from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from WinCLIP import *
from utils.eval_utils import *
#from utils.metrics import denormalization
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

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
    model.build_text_feature_gallery('candle')
    # Run inference
    with torch.no_grad():
        transformed_image = transformed_image.to(device)
        output = model(transformed_image)

    return output

def display_output(output, image_path):
    """
    Display the model output and the input image.
    
    :param output: The output scores or results from the model.
    :param image_path: The path to the input image.
    """
    # Load the image
    image = Image.open(image_path)

    # Convert output to numpy array if necessary
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    # Plot the image and output
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image
    ax[0].imshow(image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    # Display output
    ax[1].imshow(output, cmap='hot')
    ax[1].set_title('Model Output')
    ax[1].axis('off')
    
    plt.show()

def run_inference_on_image(model, image_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Run inference
    output = single_image_inference(model, image_path, device)
    
    print(f"Output for {image_path}: {output}")
    
    # Display the output
    display_output(output, image_path)
    
    return output

def main(args):
    kwargs = vars(args)

    # Set seed
    seeds = [111, 333, 999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    # Set device
    device = "cuda:0" if kwargs['use_cpu'] == 0 else "cpu"
    kwargs['device'] = device

    # Get model directory, image directory, etc.
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    # Initialize model
    model = WinClipAD(**kwargs)
    model = model.to(device)

    print(f"Model loaded and moved to {device}")
    folder_path = './datasets/VisA_pytorch/1cls/candle/test/bad'  # Update the path as needed
    im_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) ]
    image_path = im_paths[0]
    # Call the function with the model and the path to the image
    run_inference_on_image(model, image_path)

def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='candle')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result_winclip")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=0)
    parser.add_argument('--scales', nargs='+', type=int, default=(2, 3, ))
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)
    
    return parser.parse_args()

if __name__ == '__main__':
    import os
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
