# Contributing to Language Identification

Thanks for taking the time to contribute! Here are a few guidelines to keep
the project healthy and easy to maintain.

## Development setup

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model (required before running the app / tests)
python -m src.train

# Run the tests
python -m pytest -q
```

## Project layout

```
language-identification/
├── src/            # all logic: preprocessing, features, training, prediction
├── app.py          # Streamlit web app
├── api.py          # FastAPI backend
├── cli.py          # command-line interface
├── tests/          # pytest suite
└── dataset/ models/ reports/ notebooks/
```

## Coding conventions

* Python 3.10+.
* Follow **PEP8** and keep the code formatted consistently.
* Every public function needs **type hints** and a **docstring**.
* Add a short **comment** explaining *why* when the code is non-obvious.
* Use `src.utils.get_logger()` instead of bare `print()` for diagnostics.
* No duplicated logic - put shared code in the `src` modules.

## Adding a new language

1. Add the ISO-639-3 code and display name to `LANGS` in `src/config.py`.
2. Rebuild the dataset: `python -m src.data_loader` (or delete the cache CSV).
3. Retrain: `python -m src.train`.

## Testing

Write or update unit tests in `tests/` for any new behaviour. Keep the
existing tests green:

```bash
python -m pytest -q
```

## Submitting changes

1. Create a feature branch (`260804-feat-my-change`).
2. Commit your changes with a descriptive message.
3. Push and open a pull request using the template in
   `.github/PULL_REQUEST_TEMPLATE.md`.
