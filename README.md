# RAG-based Document Q&A System with Analytics Dashboard

> One-paragraph summary: what this project does (RAG Q&A over the AWS Customer Agreement, FastAPI backend, SQL usage logging, Streamlit dashboard).

## Table of Contents
- [RAG-based Document Q\&A System with Analytics Dashboard](#rag-based-document-qa-system-with-analytics-dashboard)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Setup \& Run Instructions](#setup--run-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the backend](#running-the-backend)
    - [Ingesting the document](#ingesting-the-document)
    - [Running the frontend](#running-the-frontend)
    - [Asking a question](#asking-a-question)
  - [Design Decisions \& Justifications](#design-decisions--justifications)
    - [Chunking Strategy](#chunking-strategy)
    - [Embedding Model](#embedding-model)
    - [Vector Store](#vector-store)
    - [Top-k Retrieval](#top-k-retrieval)
    - [LLM / Provider Choice](#llm--provider-choice)
    - [Handling Out-of-Scope Questions](#handling-out-of-scope-questions)
  - [API Reference](#api-reference)
    - [Request/Response examples](#requestresponse-examples)
    - [Error Handling](#error-handling)
  - [SQL Logging Schema](#sql-logging-schema)
  - [Analytics Queries](#analytics-queries)
  - [Frontend](#frontend)
  - [Demo](#demo)
  - [Assumptions \& Known Limitations](#assumptions--known-limitations)

## Architecture Overview
```mermaid
flowchart TD
      Start(["User submits question"]) --> Empty{"Query empty?"}
      Empty -- "Yes" --> Err400["Return 400 error"]
      Empty -- "No" --> Loaded{"Index loaded?<br/>(document ingested)"}
      Loaded -- "No" --> Err400b["Return 400:<br/>call /ingest first"]
      Loaded -- "Yes" --> Retrieve["Retrieve top-k chunks<br/>(ChromaDB similarity
  search)"]
      Retrieve --> Conf{"Top score above<br/>confidence threshold?"}
      Conf -- "No" --> NotFoundMsg["Return:<br/>'cannot find answer in document'"]
      Conf -- "Yes" --> Prompt["Build grounded prompt<br/>with retrieved context"]
      Prompt --> LLMCall["Call LLM (Ollama)"]
      LLMCall --> CheckAns{"LLM says<br/>not found?"}
      CheckAns -- "Yes" --> NotFoundMsg
      CheckAns -- "No" --> CiteParse["Parse cited source chunks"]
      CiteParse --> Log["Log query, answer, found,<br/>latency to SQLite"]
      NotFoundMsg --> Log
      Log --> Return["Return answer + sources<br/>to Streamlit"]
      Err400 --> End(["End"])
      Err400b --> End
      Return --> End
```