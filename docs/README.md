# AeroBeacon — Documentation Suite
### ML-Based Question Balancing Engine | DGCA CPL/ATPL Pilot Examination Platform

Welcome to the technical documentation and architectural reference for the **AeroBeacon ML-Based Question Balancing Engine**. This suite contains complete specifications, executive presentation materials, architecture flowcharts, and operational guides for adding questions.

---

## 📚 Documentation Index

| Document | Purpose | Audience |
|---|---|---|
| [**1. Overview & Presentation**](file:///c:/Users/dinesh/OneDrive/Desktop/feature1/docs/1_OVERVIEW_AND_PRESENTATION.md) | Comprehensive system overview, problem statement, feature matrix, and slide-by-slide executive presentation deck. | Executives, Product Managers, Pilots, Instructors |
| [**2. Architecture & Flowcharts**](file:///c:/Users/dinesh/OneDrive/Desktop/feature1/docs/2_ARCHITECTURE_AND_FLOWCHARTS.md) | End-to-end architecture diagrams, sequence workflows, zero-downtime fallback decision trees, and ML pipelines using Mermaid. | Lead Engineers, Architects, Backend Developers |
| [**3. Adding Questions Prompt & Ingestion Guide**](file:///c:/Users/dinesh/OneDrive/Desktop/feature1/docs/3_ADDING_QUESTIONS_PROMPT_GUIDE.md) | Copy-paste AI prompt templates for extracting questions from textbooks/PDFs, schema definitions, and automated ingestion workflows. | Content Teams, SMEs, Developers |

---

## ⚡ Quick Architecture Highlights

- **Dataset**: 806 DGCA questions covering all **28 official chapters** of Group Captain IC Joshi's *Aviation Meteorology* (7th New Edition 2023).
- **Exam Distribution Constraint**: Strictly locked **30% Easy, 50% Medium, 20% Hard** ratio across every generated test.
- **Machine Learning**: 100% self-hosted, offline `RandomForestClassifier` trained with scikit-learn (**100% test accuracy**, 0 external API dependencies).
- **Inference Speed**: Vectorized batch inference evaluates 500 questions in **116 milliseconds**.
- **Resilience**: Zero-downtime heuristic fallback if ML service is unavailable, and offline JSON pool if MongoDB is offline.
- **Microservices**: Flask ML Microservice (Port 5001) & Express API / Glassmorphism Web App (Port 3001).

---

## 🛠️ Quick Start

### 1. Launch Flask ML Microservice
```bash
cd ml
python ml_service.py
```
*Health check:* `GET http://127.0.0.1:5001/health`

### 2. Launch Express Backend & Web UI
```bash
cd backend
node src/server.js
```
*Web Application:* `http://localhost:3001`

### 3. Run Backend Unit & Integration Tests
```bash
cd backend
npm test
```
*(34/34 tests passing in ~1.6s)*
