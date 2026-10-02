# AI Course Assistant

## Overview
This project is an AI-powered course assistant designed to help students interact with course materials more effectively. The assistant aims to answer questions, provide clarifications, and support learning by leveraging natural language processing and machine learning techniques.


---

## Motivation
Large language models can generate fluent but incorrect responses and often hallucinate. When studying for a course, inaccurate explanations can reinforce misunderstandings and misguide students.

To address this, this project explores a **retrieval-augmented generation (RAG)** design in which the model first retrieves relevant passages from course materials and then generates answers conditioned on that retrieved context. By grounding responses in source material, the assistant aims to reduce hallucinations.
## Run the app
Install dependencies in the existing virtual environment:

```bash
venv/bin/python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```dotenv
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4.1-mini
```



Start the frontend:

```bash
venv/bin/python -m streamlit run app.py
```

Open the URL printed by Streamlit. The first question builds the PDF index; subsequent questions reuse it. Answers display numbered references and expandable source text. Large sections use excerpts around the matching passage. Page ranges refer to the parent section's PDF pages.
