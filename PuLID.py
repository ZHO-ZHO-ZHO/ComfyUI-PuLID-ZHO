#测试初版 别急

import sys
import torch, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything

torch.set_grad_enabled(False)

pipeline = PuLIDPipeline()

device = "cuda" if torch.cuda.is_available() else "cpu"


def tensor2nump(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class PuLID_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",), 
                "positive": ("STRING", {"default": "cat", "multiline": True}),
                "negative": ("STRING", {"default": "worst quality, low quality", "multiline": True}),
                #"batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "width": ("INT", {"default": 768, "min": 512, "max": 2024}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2024}),
                "id_scale": ("FLOAT", {"default": 0.8, "min": 0, "max": 5, "step": 0.05}),
                "mode": (['fidelity', 'extremely style'], ),
                "id_mix": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 1, "max": 1.5, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999}),
            },
            "optional": {
                "supp_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "🪐PuLID"
                       
    def generate_image(self, face_image, positive, negative, width, height, id_scale, mode, id_mix, steps, cfg, seed, supp_images=None):

        if mode == 'fidelity':
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif mode == 'extremely style':
            attention.NUM_ZERO = 16
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            raise ValueError

        id_image = tensor2nump(face_image)
      
        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings = pipeline.get_id_embedding(id_image)

            if supp_images is None:
                supp_images = []
            
            for supp_id_image in supp_images:
                if supp_id_image is not None:
                    supp_id_image = tensor2nump(supp_id_image)
                    supp_id_image = resize_numpy_image_long(supp_id_image, 1024)
                    supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
                    id_embeddings = torch.cat(
                        (id_embeddings, supp_id_embeddings if id_mix else supp_id_embeddings[:, :5]), dim=1
                    )
        else:
            id_embeddings = None
      
        #generator = torch.Generator(device=device).manual_seed(seed)
        seed_everything(seed)

        output = pipeline.inference(
            prompt=positive,
            prompt_n=negative,
            size=(1, height, width),
            image_embedding=id_embeddings, 
            id_scale=id_scale, 
            guidance_scale=cfg, 
            steps=steps,
            #generator=generator,
            )[0]
      
        output_t = pil2tensor(output)
        print(output_t.shape)
        
        return (output_t,)


NODE_CLASS_MAPPINGS = {
    "PuLID_Zho": PuLID_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLID_Zho": "🪐PuLID",
}