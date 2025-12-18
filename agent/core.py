import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .tools import (analyze_image_with_qwen, generate_3d_with_triposr,
                    generate_image_with_flux, tripo_client)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class ArtDirectorAgent:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.llm_client = InferenceClient(
            model="Qwen/Qwen2.5-72B-Instruct", token=self.hf_token
        )

    def run(self, sketch_path):
        """
        Orchestrate the workflow : vision -> Thinking -> Flux.1 -> Stable-fast-3d
        """
        if not HF_TOKEN:
            yield "error", "Missing API Token. Please check your .env file."
            return

        # Verify tools are ready (Self-Correction/Reflection)
        if tripo_client is None:
            yield (
                "critique",
                "3D Engine is offline. I will proceed with 2D enhancement only.",
            )
        # Vision Analyze
        st.write("**Vision :** Analyse du croquis avec le modèle de vision...")
        raw_description = analyze_image_with_qwen(sketch_path)
        yield "analysis", raw_description

        # Thinking (Chain of Thought & Prompt Engineering)
        st.write(
            "**Raisonnement :** L'agent réfléchit à comment améliorer le dessin..."
        )

        # Prepartion of messages for the interface
        messages = [
            {
                "role": "system",
                "content": """Tu es un expert en design industriel et modélisation 3D. 
                Ta mission est d'interpréter des esquisses pour générer des images 2D optimisées pour la reconstruction 3D (modèles photogrammétriques).
                
                RÈGLES CRUCIALES POUR LE PROMPT FLUX.1 :
                - L'objet doit être sur un FOND BLANC PUR (white background).
                - L'image doit être un "Studio shot" avec un éclairage neutre sans ombres portées (no shadows).
                - L'objet doit être ENTIER et centré (no cropping).
                - Style : Réaliste mais épuré, type catalogue produit.
                - Évite les textures trop complexes ou les transparences qui perdent les algorithmes 3D.""",
            },
            {
                "role": "user",
                "content": f"""Le modèle de vision décrit le croquis : "{raw_description}".
                
                RAISONNEMENT :
                1. ANALYSE : Identifie l'objet et sa structure géométrique simple.
                2. ACTION : Crée un PROMPT ANGLAIS pour Flux.1.
                
                CONTRAINTES PROMPT : Inclus impérativement : "isolated on white background, studio lighting, full object view, 3d model front view, high quality, consistent geometry".
                
                Format :
                THOUGHT: [Ton raisonnement]
                PROMPT: [Prompt final]""",
            },
        ]

        try:
            response_obj = self.llm_client.chat_completion(
                messages=messages, max_tokens=500
            )
            response_text = response_obj.choices[0].message.content
        except Exception as e:
            # Self-Correction
            yield (
                "critics",
                f"The thinking service is not accessible Error {e}. Use of second option.",
            )
            response_text = "THOUGHT: Server error, using default prompt. PROMPT: A high quality professional 3D render of the sketch."

        # Parsing answer for the interface
        if "PROMPT:" in response_text:
            thought_part = (
                response_text.split("PROMPT:")[0].replace("THOUGHT:", "").strip()
            )
            final_prompt = response_text.split("PROMPT:")[1].strip()
            print(f"\nPrompt: {final_prompt}")
        else:
            thought_part = "Analyse automatique effectuée."
            final_prompt = response_text
            print(f"\nPrompt automatique: {final_prompt}")

        yield (
            "plan",
            {"thought": thought_part, "prompt": final_prompt},
        )
        #  Action Generation of the image with Flux.1
        st.write("**Action :** Génération de l'image avec Flux.1...")
        improved_image_path = generate_image_with_flux(final_prompt)
        yield "image", improved_image_path

        # Action Generation of the 3d model
        if improved_image_path:
            st.write(
                "**Transformation :** Conversion in 3D model with Stable-fast-3d..."
            )
            model_3d_path = generate_3d_with_triposr(improved_image_path)
            if model_3d_path:
                yield "model_3d", model_3d_path
            else:
                yield (
                    "critique",
                    "La conversion 3D a échoué. L'image est peut-être trop complexe ou le serveur est saturé.",
                )
