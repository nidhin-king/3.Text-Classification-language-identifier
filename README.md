<div align="center">

# Language Identification (Text Classification)

Detect the language of any text using Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

A complete, production-ready **language identification** system: an ML model
that classifies input text into one of **30 languages** with a confidence
score. It ships with a training pipeline, extensive EDA, a Streamlit web app,
a FastAPI REST API, a CLI, unit tests and Docker support.

---

## Features

| Area | Details |
| --- | --- |
| Detection | 30 languages, confidence score, top-5 ranking, probability distribution |
| Inputs | Short / long text, punctuation, emojis, URLs, emails, numbers, mixed caps |
| Interface | Streamlit web app + CLI + FastAPI REST API |
| ML pipeline | Auto feature selection, 7 models compared, best model deployed |
| Extras | Real-time detection, batch CSV, drag-and-drop text file, history, dark mode, translation, voice input, TTS |

## Supported languages

English, French, Spanish, German, Italian, Portuguese, Dutch, Russian, Polish,
Turkish, Arabic, Hindi, Tamil, Malayalam, Japanese, Korean, Swedish, Danish,
Finnish, Czech, Greek, Hebrew, Ukrainian, Hungarian, Romanian, Vietnamese,
Thai, Indonesian, Bulgarian, Catalan.

---

## Dataset

The dataset is built from **public Tatoeba sentence exports** (30 languages,
~57k balanced sentences) and cached automatically. A small Kaggle-style CSV
dataset is used first when reachable.

- Downloading and caching happens automatically in [`src/data_loader.py`](src/data_loader.py).
- Column layout: `text`, `language`.

```
python -m src.data_loader
```

---

## Installation

```bash
git clone https://github.com/<your-username>/language-identification.git
cd language-identification

# Optional: virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

```bash
# 1. Train the model (downloads the dataset on first run)
python -m src.train

# 2. Predict from the command line
python cli.py "Bonjour tout le monde"
python cli.py --interactive

# 3. Launch the web app
streamlit run app.py

# 4. Launch the API
uvicorn api:app --reload --port 8000
```

### Sample predictions

```
$ python cli.py "Hello, how are you?"
input  : 'Hello, how are you?'

  🇬🇧 English
  confidence : 99.6%
  source     : model
  latency    : 2.3 ms

$ python cli.py "नमस्ते आप कैसे हैं"
  🇮🇳 Hindi
  confidence : 99.9%

$ python cli.py "வணக்கம்"
  🇮🇳 Tamil
  confidence : 99.5%
```

---

## Usage

### Web app (Streamlit)

Features: text box, **Detect** button, confidence score + progress bar,
language flag, top-5 probability bar chart, real-time detection while typing,
prediction history with **export/clear**, batch CSV prediction, drag-and-drop
text file, copy-result (JSON), dark mode, optional translation, voice input.

### CLI

```
usage: language-cli [-h] [-i] [-b CSV] [--text-column COL] [-o OUTPUT] [-v] [text ...]

positional arguments:
  text                  Text to classify.

options:
  -i, --interactive     Start interactive mode.
  -b, --batch CSV       Batch-predict rows of a CSV file.
  --text-column COL     CSV column holding the text (default: 'text').
  -o, --output OUTPUT   Output CSV path for batch mode.
  -v, --verbose         Show top-5 predictions.
```

### REST API (FastAPI)

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | API status |
| GET | `/health` | Health probe + model info |
| GET | `/languages` | List supported languages |
| POST | `/predict` | Predict one text |
| POST | `/batch` | Predict many texts |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

```json
{
  "language": "English",
  "language_code": "en",
  "confidence": 0.996,
  "probabilities": [
    {"language": "English", "language_code": "en", "confidence": 0.996},
    {"language": "Dutch",   "language_code": "nl", "confidence": 0.002},
    ...
  ],
  "top_k": [ ... ],
  "prediction_time_ms": 2.1,
  "input_text": "Hello, how are you?",
  "source": "model"
}
```

---

## Model pipeline

1. **Preprocessing** — Unicode NFC normalization, URL / email / emoji removal,
   whitespace collapse, case folding, configurable number handling.
2. **Feature engineering** — compares TF-IDF character n-grams, TF-IDF word
   n-grams, bag-of-words, character frequency histograms and Unicode-block
   profiles; the best strategy is selected automatically on validation data.
3. **Model comparison** — Naive Bayes, Logistic Regression, Linear SVM,
   SGD (modified huber), Random Forest and LightGBM are benchmarked with
   5-fold cross-validation; the winner is deployed. XGBoost is available
   opt-in (`ENABLE_XGBOOST=1 python -m src.train`).
4. **Evaluation** — accuracy, macro precision / recall / F1, confusion matrix,
   classification report, ROC-AUC (OvR) and cross-validation results.
5. **Serving** — the model, vectorizer and label encoder are persisted with
   joblib; predictions run in milliseconds.

### Model performance (shipped model)

Metrics on the held-out test split (11,484 rows, 20% of the dataset) are
saved to `reports/final_metrics.json`. The deployed model uses **TF-IDF
character n-grams** with an **SGD classifier (modified huber)** — chosen
automatically as the best probabilistic model after benchmarking:

| Metric | Value |
| --- | --- |
| Accuracy | 99.23% |
| Macro F1 | 99.26% |
| ROC-AUC (macro, OvR) | 0.9997 |
| Prediction latency | < 10 ms |

Full cross-validation comparison (`reports/model_comparison.csv`):

| Model | CV accuracy |
| --- | --- |
| Linear SVM | 0.9812 |
| SGD (modified huber) | 0.9800 |
| Multinomial Naive Bayes | 0.9779 |
| Logistic Regression | 0.9756 |
| LightGBM | 0.9363 |
| Random Forest | 0.9271 |

Run `python -m src.eda` and `python -m src.train` to regenerate the charts
and metrics for your own run.

---

## Project structure

```
language-identification/
│
├── dataset/                  # cached dataset (auto-downloaded)
├── notebooks/                # end-to-end Jupyter notebook
├── models/                   # saved model, vectorizer, label encoder, metadata
├── reports/                  # EDA charts, metrics, model comparison
├── src/
│   ├── config.py             # central configuration
│   ├── data_loader.py        # dataset download + caching
│   ├── preprocessing.py      # text cleaning pipeline
│   ├── feature_engineering.py# BOW / TF-IDF / char frequency / Unicode features
│   ├── eda.py                # exploratory data analysis
│   ├── visualizer.py         # chart helpers
│   ├── train.py              # feature + model selection, final training
│   ├── evaluate.py           # metrics, confusion matrix, CV, ROC
│   ├── predict.py            # prediction pipeline + persistence
│   ├── deep_learning.py      # optional PyTorch LSTM/BiLSTM/GRU
│   └── utils.py              # logging, timing, JSON helpers
├── app.py                    # Streamlit web app
├── api.py                    # FastAPI REST API
├── cli.py                    # command-line interface
├── tests/                    # unit tests (preprocessing, predict, api, loading)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── Makefile
├── LICENSE
└── README.md
```

---

## Testing

```bash
python -m pytest -q
```

The suite covers preprocessing, feature engineering, the prediction pipeline,
model loading and the FastAPI endpoints. Model-dependent tests skip
automatically when no trained model exists.

---

## Docker

```bash
# Build the image (run `python -m src.train` first to bake in the model)
docker build -t language-identification .

# Run API + UI together
docker compose up --build

# API:  http://localhost:8000  (docs at /docs)
# UI:   http://localhost:8501
```

---

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app on https://share.streamlit.io pointing at `app.py`.
3. Optional: add a `requirements.txt` step (auto-detected).

### Hugging Face Spaces

1. Create a Space of type *Docker* (SDK: Docker).
2. Push this repository (the `Dockerfile` is used automatically).

### Render

1. New **Web Service** from the repository.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn api:app --host 0.0.0.0 --port 8000`
   (or `streamlit run app.py --server.port $PORT` for the UI).

### Railway

1. Add the repo via the Railway dashboard.
2. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`.
3. Railway injects `$PORT` automatically.

### Docker

See the [Docker](#docker) section above.

---

## Deep learning (optional)

A PyTorch reference implementation (character-level **LSTM / BiLSTM / GRU**)
is provided in [`src/deep_learning.py`](src/deep_learning.py). It is fully
optional and only activates when PyTorch is installed.

---

## Future improvements

- Add more languages (Tatoeba supports 100+).
- Integrate fastText / CLD3 for extremely short inputs.
- Deploy with ONNX / TensorRT for lower latency.
- Add streaming inference for the API.
- Add text-to-speech and full voice input integration.

---

## License

[MIT](LICENSE)

## Author

[Your Name](https://github.com/your-username) - Language Identification Project

---

*Built as a reference implementation of an end-to-end NLP/ML project:
dataset → EDA → features → model → evaluation → serving → deployment.*
