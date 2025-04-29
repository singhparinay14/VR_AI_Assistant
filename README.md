# VR_AI_Assistant

> **Note:** This project is currently under development. Features, functionality, and documentation are subject to change.

This project is a research-driven Unity VR application that explores how a large-language-model agent can see, understand, and talk about the user’s virtual surroundings in real time.
The project serves as a living prototype for a master’s thesis on conversational AI integration in VR, tackling key research gaps such as real-time perception and multimodal context grounding. It demonstrates that a lightweight model on a standalone headset can supply just enough world knowledge for a language model to hold context-aware, believable conversations.

## Tech Stack

| Layer            | Details                                              |
|-------------------|------------------------------------------------------|
| **Engine**       | Unity 6000.0.39f1                                     |
| **VR SDK**       | OpenXR (Meta Quest 2 target)                          |
| **Neural Inference** | Unity Sentis (GPU Compute backend)                   |
| **Vision Model** | YOLO11n ONNX                                         |
| **LLM Backend**  | OpenAI API                                           |
| **Audio I/O**    | Speech-to-Text / Text-to-Speech                |
| **Scripting**    | C# (.NET Standard 2.1)                                |

Made with ❤️ for immersive AI research.
