import os
import shutil

from dotenv import load_dotenv
from gradio_client import Client, handle_file
from huggingface_hub import InferenceClient

# 1. Explicitly load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Initialize clients with error handling
# This prevents the whole app from crashing if one service is down
try:
    hf_client = InferenceClient(token=HF_TOKEN)
except Exception as e:
    print(f"Warning: HF Inference Client failed: {e}")
    hf_client = None

try:
    # Use a try-except here to catch the ValueError from your logs
    tripo_client = Client("stabilityai/TripoSR", hf_token=HF_TOKEN)
except Exception as e:
    print(f"Warning: TripoSR Space unavailable: {e}")
    tripo_client = None


def analyze_image_with_qwen(image_path):
    """
    Utilise Qwen-VL pour décrire le dessin.
    """
    # Note: Qwen-2-VL est multimodal. On peut aussi utiliser un modèle plus léger
    # comme "llava-hf/llava-1.5-7b-hf" si Qwen est trop lent via API gratuite.
    # Ici, exemple avec un modèle VLM générique sur HF API.

    # Pour simplifier l'appel API image-to-text (captioning) :
    model_id = "Salesforce/blip-image-captioning-large"
    # (Qwen-VL via API pure est parfois complexe à configurer, BLIP est très robuste pour décrire)
    # Si vous tenez absolument à Qwen, il faut utiliser gradio_client sur une Space Qwen-VL.

    try:
        with open(image_path, "rb") as f:
            data = f.read()

        # On demande une description
        response = hf_client.image_to_text(image=data, model=model_id)
        return (
            response[0]["generated_text"]
            if isinstance(response, list)
            else response.generated_text
        )
    except Exception as e:
        return f"Erreur Vision: {e}"


def generate_image_with_flux(prompt):
    """
    Utilise FLUX.1-schnell (version rapide) pour générer l'image.
    """
    try:
        print(f"🎨 Génération Flux avec prompt: {prompt}")
        image = hf_client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",  # Modèle très rapide et gratuit
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
        # result[1] is typically the path to the .glb file
        return result[1]
    except Exception as e:
        print(f"3D Generation Error: {e}")
        return None
