import os
import shutil

from dotenv import load_dotenv
from gradio_client import Client, handle_file
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# 1. Explicitly load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Initialize clients with error handling
try:
    tripo_client = Client("stabilityai/stable-fast-3d", token=HF_TOKEN)
except Exception as e:
    print(f"Warning: TripoSR Space unavailable: {e}")
    tripo_client = None

hf_client = InferenceClient()


def analyze_image_with_qwen(image_path):
    try:
        img = Image.open(image_path)
        # img = img.convert("RGB")

        text = " a draw of "
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        inputs = processor(img, text, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Erreur Vision : {e}")
        return "a conceptual drawing"


def generate_image_with_flux(prompt):
    """
    Use FLUX.1-schnell to generate the image
    """
    try:
        print(f"Flux generation with the prompt : {prompt}")
        image = hf_client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )

        # locally save the image
        output_path = "generated_flux.png"
        image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error Flux: {e}")
        return None


def generate_3d_with_triposr(image_path):
    if tripo_client is None:
        return "ERROR: 3D Engine offline"

    try:
        img = Image.open(image_path)
        img.thumbnail((512, 512))
        img.save(image_path)

        # parameter from : https://huggingface.co/spaces/stabilityai/stable-fast-3d
        # fr_result = tripo_client.predict(fr=0.85, api_name="/update_foreground_ratio")
        # tmp_result = tripo_client.predict(
        #     x=handle_file(
        #         "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
        #     ),
        #     api_name="/lambda",
        # )
        # remove backgroud
        bg_remove_result = tripo_client.predict(
            image=handle_file(image_path), fr=0.85, api_name="/requires_bg_remove"
        )
        processed_image = bg_remove_result[0]

        # generation of 3D model
        result = tripo_client.predict(
            input_image=handle_file(processed_image),
            foreground_ratio=0.85,
            remesh_option="None",
            vertex_count=-1,
            texture_size=1024,
            api_name="/run_button",
        )

        if isinstance(result, (list, tuple)) and len(result) > 1:
            return result[1]
        return result

    except Exception as e:
        error_msg = str(e)
        if "upstream" in error_msg.lower():
            return "ERROR: The Stability AI server is overload, retry later."
        return f"ERROR_3D: {error_msg}"
