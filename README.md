# AI Course Assistant (In Progress)

## Overview
This project is an AI-powered course assistant designed to help students interact with course materials more effectively. The assistant aims to answer questions, provide clarifications, and support learning by leveraging natural language processing and machine learning techniques.

**Status:** This project is actively under development.

---

## Motivation
Large language models can generate fluent but incorrect responses and often hallucinate. When studying for a course, inaccurate explanations can reinforce misunderstandings and misguide students.

To address this, this project explores a **retrieval-augmented generation (RAG)** design in which the model first retrieves relevant passages from course materials and then generates answers conditioned on that retrieved context. By grounding responses in source material, the assistant aims to reduce hallucinations.