
import gradio as gr
import pandas as pd
import requests
import os

API_URL = "http://127.0.0.1:8000/predict"

PC_COLUMNS = [f"PC_{i}" for i in range(1, 19)]


def predict_from_csv(file):

    if file is None:
        return "❌ Please upload a CSV file."

    try:
        # Read uploaded CSV
        df = pd.read_csv(file.name)

        # Check required columns
        missing = [
            c for c in PC_COLUMNS
            if c not in df.columns
        ]

        if missing:
            return (
                "❌ Missing required columns:\n"
                + ", ".join(missing)
            )

        if len(df) == 0:
            return "❌ CSV contains no records."

        # Use first record
        row = df.iloc[0]

        # ----------------------------------------------------
        # IMPORTANT:
        # FastAPI expects PC_1 ... PC_18 as individual fields
        # ----------------------------------------------------

        payload = {
            c: float(row[c])
            for c in PC_COLUMNS
        }

        # Send request to FastAPI
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            return (
                f"❌ API error: HTTP {response.status_code}\n\n"
                + response.text
            )

        result = response.json()

        prediction = result.get("prediction")
        label = result.get("label")
        probability = result.get(
            "attacker_probability"
        )

        if probability is not None:
            probability_text = f"{float(probability):.4f}"
        else:
            probability_text = "N/A"

        return (
            "✅ PREDICTION COMPLETE\n\n"
            f"Prediction: {prediction}\n"
            f"Behaviour: {label}\n"
            f"Attacker probability: {probability_text}\n\n"
            "The CSV → Gradio → FastAPI → Model pipeline "
            "worked successfully."
        )

    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {e}"


# ------------------------------------------------------------
# GRADIO INTERFACE
# ------------------------------------------------------------

with gr.Blocks(
    title="V2X Driver Behaviour Detection"
) as demo:

    gr.Markdown(
        """
        # V2X Driver Behaviour Detection

        Upload a CSV containing the 18 Phase 2 PCA features:

        **PC_1, PC_2, ..., PC_18**

        The application classifies the first record.
        """
    )

    file_input = gr.File(
        label="Upload Phase 2 PCA Feature CSV",
        file_types=[".csv"],
        type="filepath"
    )

    predict_button = gr.Button(
        "Predict",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result",
        lines=10
    )

    predict_button.click(
        fn=predict_from_csv,
        inputs=file_input,
        outputs=output
    )


if __name__ == "__main__":
    demo.launch(
        share=True
    )
