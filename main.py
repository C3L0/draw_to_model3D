import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import pipeline

# 1. Configuration
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Chemin vers un de vos dessins sauvegardés (ex: sketch.png)
TEST_IMAGE_PATH = "sketch.png"


def test_vision_pipeline(image_path):
    # Technique ReAct : Pensée -> Action -> Observation [cite: 34, 37]
    print("🧠 Pensée : Tentative d'analyse avec le modèle principal...")

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()

        # Action : Appel au modèle préféré
        raw_description = client.image_to_text(
            image=img_data, model="Salesforce/blip-image-captioning-large"
        )
    except Exception as e:
        # Self-Correction : L'agent critique l'échec et change de stratégie
        print(f"⚠️ Observation : Erreur 404. Application du plan de secours...")

        # Modèle de secours (souvent plus stable sur l'API gratuite)
        raw_description = client.image_to_text(
            image=img_data, model="nlpconnect/vit-gpt2-image-captioning"
        )

    print(f"✅ Résultat final : {raw_description}")

    # ÉTAPE B : Raisonnement (Interprétation d'Expert) [cite: 5, 29]
    # C'est ici que l'agent "décide" de l'objet malgré le dessin approximatif
    print("\n2. Interprétation par l'Agent Expert (Chain of Thought)...")

    messages = [
        {
            "role": "system",
            "content": """Tu es un expert en design industriel spécialisé dans l'interprétation d'esquisses conceptuelles brutes. 
                Ta mission est d'identifier l'INTENTION derrière le dessin, même s'il est simplifié.
                Tu dois toujours penser étape par étape (Chain of Thought).""",
        },
        {
            "role": "user",
            "content": f"""Le modèle de vision a décrit le croquis ainsi : "{raw_description}".
                
                Suis ce plan de raisonnement:
                1. ANALYSE : Interprète l'objet technique voulu derrière cette description brute.
                2. STYLE : Imagine un style visuel professionnel (ex: Cyberpunk, Réaliste, Low-poly).
                3. ACTION : Écris un PROMPT en ANGLAIS optimisé pour Flux.1 (Générateur d'image).
                
                Format de réponse :
                THOUGHT: [Ton raisonnement technique ici]
                PROMPT: [Le prompt final en anglais ici]""",
        },
    ]

    try:
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",  # Modèle stable et gratuit
            messages=messages,
            max_tokens=200,
        )
        print(f"AGENT INTERPRETATION:\n{response.choices[0].message.content}")
    except Exception as e:
        print(f"Erreur Agent : {e}")


if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        test_vision_pipeline(TEST_IMAGE_PATH)
    else:
        print(
            f"Erreur : Le fichier {TEST_IMAGE_PATH} n'existe pas. Dessinez d'abord dans l'app Streamlit."
        )
