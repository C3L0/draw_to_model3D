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
        # On utilise un LLM texte très performant pour le raisonnement
        self.llm_client = InferenceClient(
            model="Qwen/Qwen2.5-72B-Instruct", token=self.hf_token
        )

    def run(self, sketch_path):
        """
        Orchestre le flux : Vision -> Raisonnement -> Flux.1 -> TripoSR
        """
        # Thought: Analyze the environment
        if not HF_TOKEN:
            yield "error", "Missing API Token. Please check your .env file."
            return

        # Thought: Verify tools are ready (Self-Correction/Reflection)
        if tripo_client is None:
            yield (
                "critique",
                "3D Engine is offline. I will proceed with 2D enhancement only.",
            )
        # Étape 1 : Vision (Analyse)
        st.write("**Vision :** Analyse du croquis avec le modèle de vision...")
        raw_description = analyze_image_with_qwen(sketch_path)
        yield "analysis", raw_description

        # Étape 2 : Raisonnement (Chain of Thought & Prompt Engineering)
        # On force le modèle à décomposer le problème [cite: 29]
        st.write(
            "**Raisonnement :** L'agent réfléchit à comment améliorer le dessin..."
        )

        # Préparation des messages pour l'interface "conversational"
        messages = [
            {
                "role": "system",
                "content": """Tu es un expert spécialisé dans l'interprétation d'esquisses conceptuelles brutes. 
                Ta mission est d'identifier l'INTENTION derrière le dessin, même s'il est simplifié.
                Tu dois toujours penser étape par étape (Chain of Thought).""",
            },
            {
                "role": "user",
                "content": f"""Le modèle de vision a décrit le croquis ainsi : "{raw_description}".
                
                Suis ce plan de raisonnement:
                1. ANALYSE : Interprète uniquement l'objet voulu derrière cette description brute.
                3. ACTION : Écris un PROMPT en ANGLAIS optimisé pour Flux.1 (Générateur d'image).
                
                Format de réponse :
                THOUGHT: [Ton raisonnement technique ici]
                PROMPT: [Le prompt final en anglais ici]""",
            },
        ]

        # Utilisation de chat_completion au lieu de text_generation
        try:
            response_obj = self.llm_client.chat_completion(
                messages=messages, max_tokens=500
            )
            response_text = response_obj.choices[0].message.content
        except Exception as e:
            # Ici, l'agent "critique" la situation (Self-Correction)
            yield (
                "critique",
                f"Le service de raisonnement est indisponible (Erreur 500). Utilisation d'un plan de secours.",
            )
            response_text = f"THOUGHT: Server error, using default prompt. PROMPT: A high quality professional 3D render of the sketch."

        # Parsing de la réponse pour l'interface utilisateur
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
        )  # Étape 3 : Action (Génération Image avec Flux.1)
        st.write(f"**Action :** Génération de l'image avec Flux.1...")
        improved_image_path = generate_image_with_flux(final_prompt)
        yield "image", improved_image_path

        # Étape 4 : Action Finale (Génération 3D avec TripoSR)
        if improved_image_path:
            st.write("**Transformation :** Conversion en 3D avec TripoSR...")
            model_3d_path = generate_3d_with_triposr(improved_image_path)
            if model_3d_path:
                yield "model_3d", model_3d_path
            else:
                yield (
                    "critique",
                    "La conversion 3D a échoué. L'image est peut-être trop complexe ou le serveur est saturé.",
                )
