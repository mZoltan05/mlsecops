# ML SecOps – End-to-End MLOps Pipeline for Animal Image Classification

A project demonstrating a production-oriented MLOps workflow using FastAPI, MLflow, Airflow, Streamlit, and Docker.  
The goal is to show how an image classification model can be trained, tracked, versioned, deployed, and periodically retrained in a reproducible and maintainable way.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  

---

## Overview

This project implements an end-to-end machine learning pipeline for an animal image classifier based on a convolutional neural network (e.g., ResNet-18 in PyTorch).

The pipeline demonstrates:

- How to structure and deploy ML services in containers.
- How to handle model training, experiment tracking, and model registry.
- How to serve a trained model via an API and a simple web UI.
- How to orchestrate periodic retraining using Apache Airflow.
- How to design the project with reproducibility, observability, and maintainability in mind.


---

## Features

### MLOps and Model Lifecycle Management

- Centralized experiment tracking with MLflow.
- Logging of hyperparameters, metrics, and artifacts (models, plots, etc.).
- Model registration and versioning via MLflow Model Registry.
- Automatic promotion of the best model to a "Staging" stage (or similar) based on accuracy or other criteria.
- Separation between training code and inference code.

### FastAPI Service (App)

- REST API for:
  - Triggering training runs (`/train`).
  - Running inference (`/infer`).
- Uses Pydantic models for request/response validation.
- Loads the latest appropriate model from MLflow (e.g., the latest version in the "Staging" stage).
- Configurable via environment variables (host, port, MLflow URI, etc.).

### Airflow Orchestration

- Apache Airflow DAG(s) to:
  - Schedule and automate training.
  - Potentially perform data preparation or evaluation steps.
- Integration with the training script / FastAPI training endpoint.
- Airflow Web UI for monitoring DAG runs.

### Streamlit Frontend

- Simple web application for displaying server parameters.

### MLflow Tracking Server

- Dedicated MLflow tracking server and backend store.
- Central place to inspect:
  - Runs
  - Metrics (accuracy, loss, etc.)
  - Parameters
  - Artifacts (saved models, evaluation figures)

### Fully Containerized Stack

- All components run as Docker containers orchestrated with Docker Compose.
- Each service (App, Airflow, Streamlit, MLflow) is isolated but networked together.
- Easy to start and stop the complete environment with a single command.
- Reproducible environment independent of the host OS (as long as Docker and Docker Compose are installed).

---

## Architecture

The system is composed of four main logical components:

- **FastAPI App** – exposes training and inference endpoints.
- **MLflow Server** – tracks experiments and stores registered models.
- **Airflow** – orchestrates training and other workflows.
- **Streamlit UI** – serves as a simple front-end client for predictions.

High-level data flow:

1. Training:
   - Airflow or a direct API call triggers the training process.
   - The training code logs metrics and artifacts to MLflow.
   - The best performing model is registered and moved to a staging or production-like stage.

2. Inference:
   - The inference code loads the latest staging model from MLflow.
   - Clients call the FastAPI `/infer` endpoint or use the Streamlit UI.
   - The prediction is computed and returned to the client.

