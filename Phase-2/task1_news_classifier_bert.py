# =============================================================================
# Task 1: News Topic Classifier Using BERT
# =============================================================================
# Objective:
#   Fine-tune a transformer model (BERT) to classify news headlines into
#   topic categories using the AG News Dataset from Hugging Face.
#
# Dataset:   AG News (via Hugging Face Datasets)
# Model:     bert-base-uncased
# Framework: Hugging Face Transformers + PyTorch
# Deploy:    Gradio (live interaction demo at end of file)
#
# Requirements:
#   pip install transformers datasets torch scikit-learn gradio
# =============================================================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"   # Pre-trained model to fine-tune
NUM_LABELS = 4                     # AG News has 4 classes: World, Sports, Business, Sci/Tech
MAX_LENGTH = 128                   # Max token length for headlines (short texts)
BATCH_SIZE = 16                    # Training batch size
NUM_EPOCHS = 3                     # Number of fine-tuning epochs
OUTPUT_DIR = "./bert_agnews"       # Directory to save the fine-tuned model

# AG News label mapping (0-indexed)
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# ---------------------------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------------------------

def load_agnews():
    """
    Load the AG News dataset from Hugging Face.
    Returns train and test splits.
    """
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")

    # The dataset labels are 0-3 for the four categories
    print(f"  Train samples : {len(dataset['train'])}")
    print(f"  Test  samples : {len(dataset['test'])}")
    return dataset


# ---------------------------------------------------------------------------
# 3. Tokenization / Preprocessing
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize all text fields in the dataset using the BERT tokenizer.

    Args:
        dataset   : Hugging Face DatasetDict
        tokenizer : Pre-loaded BertTokenizer

    Returns:
        tokenized_dataset : DatasetDict with input_ids, attention_mask, labels
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",   # Pad shorter sequences to MAX_LENGTH
            truncation=True,        # Truncate longer sequences to MAX_LENGTH
            max_length=MAX_LENGTH,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True)

    # Rename 'label' → 'labels' (required by Hugging Face Trainer)
    tokenized = tokenized.rename_column("label", "labels")

    # Keep only the columns the model needs
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized


# ---------------------------------------------------------------------------
# 4. Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute accuracy and weighted F1-score during evaluation.

    Args:
        eval_pred : EvalPrediction namedtuple (logits, labels)

    Returns:
        dict with 'accuracy' and 'f1' keys
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Take the class with highest logit

    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")

    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# 5. Model Initialisation
# ---------------------------------------------------------------------------

def build_model():
    """
    Load bert-base-uncased with a classification head for NUM_LABELS classes.

    Returns:
        tokenizer : BertTokenizer
        model     : BertForSequenceClassification
    """
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

def train_model(model, tokenized_dataset):
    """
    Fine-tune the BERT model using Hugging Face Trainer.

    Args:
        model             : BertForSequenceClassification
        tokenized_dataset : DatasetDict (train / test splits tokenized)

    Returns:
        trainer : Trained Trainer object
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",   # Evaluate at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,   # Keep the best checkpoint
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",              # Disable wandb / MLflow logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()
    return trainer


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(trainer):
    """
    Evaluate the fine-tuned model on the test set and print results.

    Args:
        trainer : Trained Trainer object
    """
    print("\nEvaluating on test set...")
    results = trainer.evaluate()
    print(f"  Accuracy : {results['eval_accuracy']:.4f}")
    print(f"  F1-Score : {results['eval_f1']:.4f}")
    return results


# ---------------------------------------------------------------------------
# 8. Inference Helper
# ---------------------------------------------------------------------------

def predict_headline(text, tokenizer, model, device):
    """
    Predict the topic category of a single news headline.

    Args:
        text      : Raw headline string
        tokenizer : Fine-tuned BertTokenizer
        model     : Fine-tuned BertForSequenceClassification
        device    : torch.device

    Returns:
        label_name : String category name (e.g. "Sports")
        confidence : Float confidence of the prediction
    """
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    return LABEL_NAMES[pred_label], confidence


# ---------------------------------------------------------------------------
# 9. Gradio Deployment
# ---------------------------------------------------------------------------

def launch_gradio_app(tokenizer, model, device):
    """
    Launch an interactive Gradio web interface for live headline classification.

    Args:
        tokenizer : Fine-tuned BertTokenizer
        model     : Fine-tuned BertForSequenceClassification
        device    : torch.device
    """
    def gradio_predict(headline):
        """Wrapper used by the Gradio interface."""
        if not headline.strip():
            return "Please enter a headline.", ""

        label, conf = predict_headline(headline, tokenizer, model, device)
        return label, f"{conf * 100:.1f}%"

    interface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(
            lines=2,
            placeholder="Enter a news headline here...",
            label="News Headline",
        ),
        outputs=[
            gr.Textbox(label="Predicted Category"),
            gr.Textbox(label="Confidence"),
        ],
        title="📰 News Topic Classifier (BERT)",
        description=(
            "Fine-tuned bert-base-uncased on AG News.\n"
            "Categories: World | Sports | Business | Sci/Tech"
        ),
        examples=[
            ["Stocks rally as Fed hints at rate cuts"],
            ["Brazil wins the World Cup in a dramatic final"],
            ["NASA launches new Mars exploration rover"],
            ["UN Security Council meets over Middle East tensions"],
        ],
    )

    interface.launch(share=False)  # Set share=True to get a public link


# ---------------------------------------------------------------------------
# 10. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Step 1 – Load raw dataset
    dataset = load_agnews()

    # Step 2 – Build tokenizer and model
    tokenizer, model = build_model()
    model.to(device)

    # Step 3 – Tokenize / preprocess
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Step 4 – Fine-tune
    trainer = train_model(model, tokenized_dataset)

    # Step 5 – Evaluate
    evaluate_model(trainer)

    # Step 6 – Save fine-tuned model for reuse
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")

    # Step 7 – Launch Gradio demo
    print("\nLaunching Gradio interface...")
    launch_gradio_app(tokenizer, model, device)


if __name__ == "__main__":
    main()
