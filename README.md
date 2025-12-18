# AI-Driven Sketch-to-3D Design Agent
> **A Generative AI Project: Transforming raw conceptual sketches into high-fidelity 3D assets using an Intelligent Agent.**

## Overview
This project implements a **Multi-modal AI Agent** capable of transforming hand-drawn sketches (via a Streamlit interface) into professional product concepts (FLUX.1) and subsequently into 3D models (Stable-Fast-3D). 

The system's core value lies in its **Reasoning Engine**, which interprets the user's intent to compensate for sketch imperfections, ensuring a high-quality design output regardless of the user's drawing skills.

---

## Architecture & Workflow (ReAct Pattern)
The agent follows a **Reasoning + Acting (ReAct)** loop to maintain consistency between the initial sketch and the final 3D asset:

1.  **Perception (Vision):** Uses a multi-model vision pipeline to convert canvas pixels into a raw semantic description.
2.  **Cognition (Chain of Thought):** The LLM analyzes the description, identifies the industrial design intent, and formulates a technical design strategy.
3.  **Action (2D Generation):** Generates a specialized "Studio" prompt for **FLUX.1** to create an image on a pure white background—optimized for 3D reconstruction.
4.  **Action (3D Reconstruction):** Extracts geometry from the 2D image using **Stable-Fast-3D** via a gated API.

---

## Technical Stack
* **Environment:** Python 3.12 managed by `uv` (for lightning-fast, reproducible dependency management).
* **Interface:** Streamlit + `streamlit-drawable-canvas`.
* **Agent Logic:** Hugging Face `InferenceClient` (using Qwen/Llama models).
* **Vision Pipeline:** `microsoft/git-base` & `nlpconnect/vit-gpt2-image-captioning` (Multi-model fallback).
* **Image Generation:** `black-forest-labs/FLUX.1-schnell`.
* **3D Mesh Engine:** `stabilityai/stable-fast-3d` (via Gradio API).

---

## Getting Started

### Prerequisites
* `uv` installed on your machine.
* A Hugging Face Token with access to **Gated Models** (specifically Stable-Fast-3D).
* Set the Hugging Face token in a `.env`  

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd genAI_project

# Pin Python version and sync dependencies
uv python pin 3.12
uv sync

---

### Running the App
* `uv run streamlit run app.py`

---

## Core Features: Reliability & Resilience
To meet the High Reliability project requirements, several robust mechanisms were implemented:

* **Self-Correction Strategy:** If the primary vision API fails (404/500 errors), the agent automatically switches to a backup model to ensure continuity.

* **Graceful Degradation:** If the 3D generation server is saturated or gated, the agent captures the exception, informs the user via the "THOUGHT" block, and provides the high-definition 2D image as a valid fallback.

* **Data Normalization:** Built-in image preprocessing (RGBA to RGB conversion and resizing) ensures compatibility between the frontend canvas and the inference APIs.

* **Chain of Thought (CoT):** Every action is preceded by a "THOUGHT" block visible in the UI, proving the agent's step-by-step reasoning process.

---

## Project Structure
* **app.py:** Streamlit UI, session state management, and file handling.

* **agent/core.py:** The "Brain" of the agent (System prompts, ReAct logic).

* **agent/tools.py:** Technical tools (Vision, Flux, and 3D API integrations).

* **pyproject.toml:** Project metadata and dependency locking.

---

## License
Project developed for the GenAI Certification Course - December 2025.
