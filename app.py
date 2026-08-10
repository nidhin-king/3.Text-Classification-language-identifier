"""Streamlit web application for Language Identification.

Run with::

    streamlit run app.py

Features: text box + detect button, real-time detection while typing,
confidence score, language flag, top-5 probability ranking and chart,
prediction history with export/clear, batch CSV prediction, drag-and-drop
text file, copy result, dark mode, optional translation / TTS / voice input.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import EXAMPLE_TEXTS, FLAG_EMOJI, MODEL_DIR, PROJECT_ROOT
from src.predict import PredictionError, get_predictor
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger("app")

st.set_page_config(
    page_title="Language Identification",
    page_icon="\U0001F310",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the language model ...")
def load_predictor():
    """Load (once) the trained prediction pipeline."""
    return get_predictor(MODEL_DIR)


@st.cache_data(show_spinner=False)
def load_model_meta() -> dict:
    """Read model metadata for the sidebar."""
    meta_path = MODEL_DIR / "model_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "detected" not in st.session_state:
        st.session_state.detected = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def detect_text(text: str, predictor) -> dict:
    """Run prediction and record it into the session history."""
    result = predictor.predict(text)
    st.session_state.history.append(
        {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "input": text,
            "language": result["language"],
            "confidence": round(result["confidence"], 4),
            "source": result["source"],
            "latency_ms": result["prediction_time_ms"],
        }
    )
    return result


def render_result(result: dict) -> None:
    """Render a prediction result with flag, confidence and top-5 chart."""
    lang = result["language"]
    flag = FLAG_EMOJI.get(lang, "\U0001F310")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"### {flag} {lang}")
        st.caption(f"source: {result['source']} - {result['prediction_time_ms']:.1f} ms")
    with c2:
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.progress(min(1.0, float(result["confidence"])))
    with c3:
        lang_code = result["language_code"]
        st.text_input(
            "Detected language code",
            value=f"{lang} ({lang_code})" if lang_code else lang,
            disabled=True,
            label_visibility="collapsed",
            key=f"lang_code_input_{lang}",
        )

    # Top-5 ranking + probability chart
    st.subheader("Probability distribution (top-5)")
    top5 = pd.DataFrame(
        [
            {"language": item["language"], "probability": item["confidence"]}
            for item in result["top_k"]
        ]
    )
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.bar_chart(top5.set_index("language"), height=280)
    with col_table:
        display = top5.copy()
        display["probability"] = (display["probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)

    # Copy result
    copy_payload = json.dumps(
        {
            "language": result["language"],
            "confidence": result["confidence"],
            "top_k": result["top_k"],
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "Copy result (JSON)",
        data=copy_payload,
        file_name="prediction.json",
        mime="application/json",
        key="copy_result_btn",
        help="Downloads the result as JSON; copy it from the download or the top-5 table.",
    )


def render_history(key_prefix: str = "history") -> None:
    """Render the prediction history with clear + export controls.

    Args:
        key_prefix: Prefix used to keep widget keys unique when this view is
            rendered in more than one tab.
    """
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Detect some text to build your history.")
        return

    st.subheader(f"Prediction history ({len(history)})")
    df = pd.DataFrame(history)
    st.dataframe(df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Export history (CSV)",
            data=df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            key=f"{key_prefix}_export_btn",
        )
    with c2:
        if st.button("Clear history", key=f"{key_prefix}_clear_btn", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_batch(predictor) -> None:
    """Batch prediction from an uploaded CSV file."""
    st.subheader("Batch prediction")
    uploaded = st.file_uploader(
        "Upload a CSV with a text column (drag & drop works)",
        type=["csv"],
        key="batch_csv_upload",
        help="The file must contain a column named 'text'.",
    )
    if uploaded is None:
        st.caption("Expected CSV layout: a column named `text` with one sentence per row.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV file: {exc}")
        return

    if "text" not in df.columns:
        st.error(f"CSV must contain a 'text' column. Found: {list(df.columns)}")
        return

    if st.button("Run batch prediction", type="primary"):
        with st.spinner(f"Predicting {len(df)} rows ..."):
            results = predictor.predict_many(df["text"].fillna("").astype(str).tolist())
        df["predicted_language"] = [r["language"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["latency_ms"] = [r["prediction_time_ms"] for r in results]

        st.success(f"Predicted {len(df)} rows")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download predictions (CSV)",
            data=df.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv",
            key="download_batch_btn",
        )


def render_text_file(predictor) -> None:
    """Load a plain text file and run prediction on its first line(s)."""
    st.subheader("Drag & drop a text file")
    uploaded = st.file_uploader("Upload a .txt file", type=["txt", "md"], key="txt_upload")
    if uploaded is None:
        return
    content = uploaded.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()][:5]
    if not lines:
        st.warning("The file appears to be empty.")
        return

    st.markdown("**First lines detected:**")
    for line in lines:
        try:
            result = predictor.predict(line)
            flag = FLAG_EMOJI.get(result["language"], "")
            st.markdown(
                f"- {flag} **{result['language']}** ({result['confidence'] * 100:.0f}%) - `{line[:60]}`"
            )
        except PredictionError as exc:
            st.markdown(f"- could not classify `{line[:40]}`: {exc}")


def render_translation_result(text: str) -> None:
    """Optional translation panel for the currently detected input."""
    with st.expander("Translate the input text"):
        target = st.selectbox("Target language", list(FLAG_EMOJI))
        if st.button("Translate", use_container_width=True):
            import urllib.parse
            import urllib.request

            try:
                source = st.session_state.history[-1]["language"] if st.session_state.history else "English"
                from src.config import LANGDETECT_TO_NAME

                src_code = LANGDETECT_TO_NAME.get(source, "en")
                q = urllib.parse.quote(text)
                pair = f"{src_code}|{target[:2].lower()}"
                url = f"https://api.mymemory.translated.net/get?q={q}&langpair={pair}"
                with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
                    data = json.loads(resp.read().decode("utf-8"))
                translated = data.get("responseData", {}).get("translatedText")
                st.success(translated if translated else "No translation returned.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Translation failed: {exc}")


def render_voice_input() -> None:
    """Optional browser-based voice input (webkitSpeechRecognition)."""
    with st.expander("Voice input (browser-based)"):
        st.caption(
            "Uses your browser's speech recognition. Click the microphone, "
            "speak, then copy the transcript into the text box above."
        )
        voice_html = """
        <div style="text-align:center; padding:8px;">
          <button id="mic" onclick="startRec()" style="font-size:18px;">Mic</button>
          <p id="status">Idle</p>
          <p id="out"></p>
        </div>
        <script>
          const rec = window.SpeechRecognition || window.webkitSpeechRecognition;
          function startRec() {
            if (!rec) { document.getElementById('status').innerText = 'Not supported'; return; }
            const r = new rec();
            r.lang = 'auto';
            r.interimResults = false;
            r.onstart = () => document.getElementById('status').innerText = 'Listening...';
            r.onresult = (e) => {
              const t = e.results[0][0].transcript;
              document.getElementById('out').innerText = t;
              document.getElementById('status').innerText = 'Done - copy the text below';
              if (window.parent) {
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: t}, '*');
              }
            };
            r.onerror = (e) => document.getElementById('status').innerText = 'Error: ' + e.error;
            r.start();
          }
        </script>
        """
        st.components.v1.html(voice_html, height=140)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(predictor, meta: dict) -> None:
    st.sidebar.header("Model")
    st.sidebar.caption(
        f"{meta.get('model', '?')} - {meta.get('feature_strategy', '?')}"
    )
    if meta:
        st.sidebar.metric("Accuracy (test)", f"{meta.get('accuracy', 0) * 100:.1f}%")
        st.sidebar.metric("Languages", meta.get("n_languages", "?"))
        st.sidebar.metric("F1 (macro)", f"{meta.get('f1_macro', 0):.3f}")
    else:
        st.sidebar.warning("Model not trained yet. Run `python -m src.train`.")

    st.sidebar.header("Supported languages")
    langs = getattr(predictor, "languages", [])
    st.sidebar.write(
        ", ".join(f"{FLAG_EMOJI.get(l, '')} {l}" for l in sorted(langs))
    )

    st.sidebar.header("Help")
    st.sidebar.markdown(
        "Type or paste text, press **Detect**, and get the language with a "
        "confidence score. Enable **Real-time** to detect while typing."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    predictor = load_predictor()
    meta = load_model_meta()

    render_sidebar(predictor, meta)

    st.title("Language Identification \U0001F310")
    st.markdown(
        "Detect the language of any text with a machine-learning model "
        "trained on 30 languages."
    )

    tabs = st.tabs(["Detect", "Batch prediction", "History", "Text file", "About"])

    # ---- Detect tab ------------------------------------------------------ #
    with tabs[0]:
        # Example selector
        example = st.selectbox(
            "Try an example",
            ["Custom text"] + EXAMPLE_TEXTS,
        )
        default_text = "" if example == "Custom text" else example

        col_in, col_cfg = st.columns([3, 1])
        with col_in:
            text = st.text_area(
                "Input text",
                value=default_text,
                height=150,
                placeholder="Type or paste text in any of the 30 supported languages...",
                key="input_text",
            )
        with col_cfg:
            st.write("")
            realtime = st.toggle("Real-time detection", value=False)
            st.write("")
            detect_clicked = st.button("Detect", type="primary", use_container_width=True)

        # Real-time detection while typing
        if realtime and text.strip():
            try:
                render_result(detect_text(text, predictor))
                render_translation_result(text)
            except PredictionError as exc:
                st.warning(str(exc))
            render_history(key_prefix="detect")
        elif detect_clicked:
            if not text.strip():
                st.warning("Please enter some text first.")
            else:
                try:
                    render_result(detect_text(text, predictor))
                    render_translation_result(text)
                except PredictionError as exc:
                    st.warning(str(exc))
                render_history(key_prefix="detect")

        render_voice_input()

    # ---- Batch tab ------------------------------------------------------- #
    with tabs[1]:
        render_batch(predictor)

    # ---- History tab ----------------------------------------------------- #
    with tabs[2]:
        render_history()

    # ---- Text file tab --------------------------------------------------- #
    with tabs[3]:
        render_text_file(predictor)

    # ---- About tab ------------------------------------------------------- #
    with tabs[4]:
        st.markdown(
            """
### About

This application classifies text into one of **30 languages** using a
machine-learning pipeline:

1. **Preprocessing** - Unicode normalization, URL / email / emoji removal.
2. **Features** - TF-IDF character n-grams (auto-selected strategy).
3. **Model** - the best classifier from a benchmarked set (Naive Bayes,
   Logistic Regression, Linear SVM, SGD, Random Forest, XGBoost, LightGBM).

### Performance

* Prediction latency is typically **< 10 ms** on a CPU.
* Test accuracy of the shipped model is shown in the sidebar.

### Stack

* Streamlit UI, FastAPI backend (see `api.py`), scikit-learn model.
            """
        )


if __name__ == "__main__":
    main()
