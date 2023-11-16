import gradio as gr
import torch
from PIL import Image
import numpy as np
from accelerate import Accelerator
from model import DSD_demo_Model


def image_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h)) 
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0


def compute_score(input_image, input_text):
    model = DSD_demo_Model(model_path = './examples/final-checkpoint-47500')
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    img = Image.fromarray(input_image.astype('uint8'), 'RGB').resize((512, 512))
    texts = [input_text]  
    score = model(texts, img)
    return score 


gr_interface = gr.Interface(
    fn=compute_score,
    inputs=[
        gr.Image(), 
        gr.Textbox(lines=2, placeholder="Enter Description Here...")
    ],
    outputs=gr.Number(),  
    examples=[
        ['./examples/dog.jpg', 'A text sentence you want to compute the matching score, such as: A dog is smiling.']
    ]
)

gr_interface.launch()