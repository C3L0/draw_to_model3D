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
# This prevents the whole app from crashing if one service is down
try:
    tripo_client = Client("zxhezexin/openlrm-base-obj-1.0", token=HF_TOKEN)
except Exception as e:
    print(f"Warning: TripoSR Space unavailable: {e}")
    tripo_client = None

hf_client = InferenceClient()


def analyze_image_with_qwen(image_path):
    try:
        with open(image_path, "rb") as f:
            img = Image.open(image_path)

            # 2. Conversion cruciale : Convertir en RGB
            # Le canvas Streamlit génère du RGBA (avec transparence).
            # Beaucoup de modèles de vision plantent s'ils reçoivent 4 canaux au lieu de 3.
            img = img.convert("RGB")
        #
        # Changement de modèle pour plus de stabilité
        # 'microsoft/git-base' est très performant pour les descriptions simples
        # response = hf_client.image_to_text(image=img_data, model="microsoft/git-base")
        text = " a draw of "
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        #
        inputs = processor(img, text, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Erreur Vision : {e}")
        return "a conceptual drawing"


def generate_image_with_flux(prompt):
    """
    Utilise FLUX.1-schnell (version rapide) pour générer l'image.
    """
    try:
        print(f"Génération Flux avec prompt: {prompt}")
        image = hf_client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )

        # Sauvegarder l'image localement pour la passer à l'étape suivante
        output_path = "generated_flux.png"
        image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Erreur Flux: {e}")
        return None


def generate_3d_with_triposr(image_path):
    if tripo_client is None:
        print("Error: TripoSR client not initialized.")
        return None

    try:
        # Step-by-step processing as suggested by Chain of Thought
        result = tripo_client.predict(
            input_image=handle_file(image_path),
            preprocess=True,
            api_name="/process_image",
        )
        # result[1] is the path to the .glb file
        return result[1]
    except Exception as e:
        print(f"3D Generation Error: {e}")
        return None
