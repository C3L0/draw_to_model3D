import os
import tempfile

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from agent.core import ArtDirectorAgent

# Configuration de la page
st.set_page_config(page_title="Agent Créatif : Dessin vers 3D", layout="wide")

# Initialisation de l'agent
agent = ArtDirectorAgent()

st.title("Agent Créatif : Du Dessin à la 3D")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Dessinez votre concept")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Lancer le Raisonnement de l'Agent"):
        if canvas_result.image_data is not None:
            # Création d'un fichier temporaire pour le croquis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
                img.convert("RGB").save(tmp_file.name)
                sketch_path = tmp_file.name

            with col2:
                st.subheader("2. Résultats de l'Agent")

                # Container pour afficher les étapes de raisonnement (CoT / ReAct)
                thought_container = st.expander("Logique de l'Agent", expanded=True)

                # Exécution du cycle Pensée -> Action -> Observation
                for step_type, data in agent.run(sketch_path):
                    if step_type == "analysis":
                        thought_container.write(f"**Analyse :** {data}")

                    elif step_type == "plan":
                        thought_container.info(
                            f"**Pensée (Chain of Thought) :** {data['thought']}"
                        )
                        st.session_state.prompt = data["prompt"]

                    elif step_type == "image":
                        st.image(data, caption="Amélioration 2D (Flux.1)")

                    elif step_type == "critique":
                        thought_container.warning(f"**Critique/Reflexion :** {data}")

                    elif step_type == "model_3d":
                        if data is not None:
                            st.success("Modèle 3D prêt !")
                            with open(data, "rb") as f:
                                st.download_button(
                                    "Télécharger le modèle 3D",
                                    data=f,
                                    file_name="concept.glb",
                                )
                        else:
                            # L'agent observe un échec et communique le résultat (Observation)
                            st.error(
                                "L'agent a rencontré une erreur lors de la création du modèle 3D."
                            )
        else:
            st.warning("Veuillez d'abord dessiner une esquisse sur le canvas.")
# import os
#
# import streamlit as st
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas
#
# from agent.core import ArtDirectorAgent
#
# # Page Config
# st.set_page_config(page_title="AI Sketch-to-3D Architect", layout="wide")
#
# st.title(" AI Architect: Sketch to 3D Agent")
# st.markdown(
#     """
# This application uses an **Intelligent Agent** capable of reasoning (CoT)
# to interpret your sketch and transform it into a 3D model.
# """
# )
#
# # Sidebar for controls
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")
#
# # Layout: 2 Columns
# col1, col2 = st.columns([1, 1])
#
# with col1:
#     st.subheader("1. Draw your concept")
#     # Canvas for drawing
#     canvas_result = st_canvas(
#         fill_color="rgba(255, 165, 0, 0.3)",
#         stroke_width=3,
#         stroke_color="#000000",
#         background_color="#ffffff",
#         height=400,
#         drawing_mode="freedraw",
#         key="canvas",
#     )
#
#     if st.button(" Launch Agent Reasoning"):
#         if canvas_result.image_data is not None and api_key:
#             # Initialize Agent
#             agent = ArtDirectorAgent(openai_client=api_key)  # Pass client in real app
#
#             # Container for logs
#             log_container = st.expander(" Agent Thought Process (Live)", expanded=True)
#
#             # Convert canvas to image for the agent
#             img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
#
#             # Run the agent loop
#             with col2:
#                 st.subheader("2. Agent Results")
#
#                 # We iterate through the generator to show progress live
#                 for step_type, data in agent.run(img):
#                     if step_type == "analysis":
#                         log_container.markdown(f"** Perception:** {data}")
#
#                     elif step_type == "plan":
#                         log_container.markdown(
#                             f"** Reasoning (CoT):** {data['thought_process']}"
#                         )
#
#                     elif step_type == "image":
#                         st.image(data, caption=" AI Enhanced Design")
#
#                     elif step_type == "critique":
#                         log_container.success(f"** Self-Correction:** {data}")
#
#                     elif step_type == "model_3d":
#                         st.success(" 3D Model Generated!")
#                         # Display 3D model (using a simple iframe or st-model-viewer)
#                         st.components.v1.iframe(
#                             src=f"https://googlewebcomponents.github.io/model-viewer/examples/iframe.html?src={data}",
#                             height=500,
#                         )
#         else:
#             st.error("Please provide an API Key and draw something.")
