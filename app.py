import gradio as gr
import torch
import torch.nn as nn
from cyclegan_model import ResNetGenerator
from data_utils import get_transforms
from PIL import Image

# Load Pre-trained Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_AB = ResNetGenerator(3, 3, n_blocks=6).to(device)
G_BA = ResNetGenerator(3, 3, n_blocks=6).to(device)

def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# Assume we have paths for the saved model weights
# load_checkpoint(G_AB, 'G_AB.pth')
# load_checkpoint(G_BA, 'G_BA.pth')

transform = get_transforms(img_size=128)

def translate(input_img, direction):
    img = Image.fromarray(input_img).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if direction == "Sketch → Photo":
            output_tensor = G_AB(input_tensor)
        else:
            output_tensor = G_BA(input_tensor)
            
    output_img = output_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    output_img = (output_img * 0.5 + 0.5) * 255
    return output_img.astype('uint8')

interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Radio(["Sketch → Photo", "Photo → Sketch"], label="Translation Direction")
    ],
    outputs=gr.Image(label="Output Image"),
    title="CycleGAN Sketch-Photo Translator",
    description="Translate between sketches and photos using a CycleGAN model trained on unpaired data."
)

if __name__ == "__main__":
    interface.launch()
