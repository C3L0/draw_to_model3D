import streamlit as st

from .tools import generate_3d_model, generate_image_from_prompt


class ArtDirectorAgent:
    """
    This agent uses a Chain of Thought (CoT) approach to analyze a sketch,
    improve it, and convert it to 3D.
    """

    def __init__(self, openai_client):
        self.client = openai_client
        self.history = []  # To store the reasoning steps

    def run(self, user_sketch_image):
        """
        Main execution loop implementing the Reasoning process.
        """
        st.info("🤖 Agent: Starting reasoning process...")

        # Step 1: Perception & Analysis (Vision)
        analysis = self._analyze_sketch(user_sketch_image)
        yield "analysis", analysis

        # Step 2: Reasoning & Planning (Chain of Thought)
        # Explicitly asking the model to think step-by-step [cite: 30]
        plan = self._create_improvement_plan(analysis)
        yield "plan", plan

        # Step 3: Action (Image Generation)
        # Based on the plan, we generate a high-quality 2D image
        enhanced_image_url = generate_image_from_prompt(plan["image_prompt"])
        yield "image", enhanced_image_url

        # Step 4: Self-Correction / Critique [cite: 38]
        # The agent checks if the generated image matches the intent
        critique = self._critique_result(analysis, enhanced_image_url)
        yield "critique", critique

        # NOTE: Here you could add a "while" loop to regenerate if the critique is bad.
        # For this MVP, we proceed to 3D.

        # Step 5: Final Action (3D Generation)
        st.info("🤖 Agent: Validating design. Initiating 3D transformation...")
        model_3d_url = generate_3d_model(enhanced_image_url)
        yield "model_3d", model_3d_url

    def _analyze_sketch(self, image):
        # TODO: Send image to GPT-4o Vision to get a description
        # Mock response for structure
        return "I see a rough sketch of a medieval chair, but the legs are uneven."

    def _create_improvement_plan(self, analysis):
        # TODO: Implement Chain of Thought prompting
        # Prompt: "Think step by step. How can we make this chair realistic?"
        return {
            "thought_process": "1. Correct perspective. 2. Add wood texture. 3. Fix lighting.",
            "image_prompt": "A highly detailed photorealistic medieval wooden chair, studio lighting...",
        }

    def _critique_result(self, original_analysis, generated_image_url):
        # TODO: Self-Correction logic
        return (
            "The generated image matches the description well. Wood texture is visible."
        )
