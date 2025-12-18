import os
import tempfile

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from agent.core import ArtDirectorAgent

st.set_page_config(page_title="Agent Créatif : Dessin vers 3D", layout="wide")
agent = ArtDirectorAgent()

st.title("Creative Agent : From Draw to 3D")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Draw your concept")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Launch the agent reasoning"):
        if canvas_result.image_data is not None:
            # Création d'un fichier temporaire pour le croquis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
                img.convert("RGB").save(tmp_file.name)
                sketch_path = tmp_file.name

            with col2:
                st.subheader("Resultats")

                # Reasoning step CoT / ReAct
                thought_container = st.expander("Agent logic", expanded=True)

                # Execute Think -> Act -> Observe
                for step_type, data in agent.run(sketch_path):
                    if step_type == "analysis":
                        thought_container.write(f"**Analyze :** {data}")

                    elif step_type == "plan":
                        thought_container.info(
                            f"**Thinking (Chain of Thought) :** {data['thought']}"
                        )
                        st.session_state.prompt = data["prompt"]

                    elif step_type == "image":
                        st.image(data, caption="Improve 2D (Flux.1)")

                    elif step_type == "critique":
                        thought_container.warning(f"**Critics/Reflection :** {data}")

                    elif step_type == "model_3d":
                        if (
                            data
                            and isinstance(data, str)
                            and not data.startswith("ERROR")
                        ):
                            try:
                                with open(data, "rb") as f:
                                    st.success("3D Model successfully generated !")
                                    st.download_button(
                                        label="Download the model (.glb)",
                                        data=f,
                                        file_name="mon_modele_3d.glb",
                                        mime="model/gltf-binary",
                                    )
                            except Exception as e:
                                st.error(f"Error while reading the file : {e}")
                        else:
                            st.warning(f"Information from the agent : {data}")
        else:
            st.warning("Please draw first on the canvas before launching the reasoning")
