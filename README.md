# Text Summarizer using Transformers (T5)

![Text Summarizer Snapshot](Images/Snapshot of text summarizer.png")

Abstractive text summarization built by fine-tuning T5-small on dialogue data, served through a FastAPI backend with a simple HTML frontend.

---

## Table of Contents

- <a href="#overview">Overview</a>
- <a href="#why-this-project">Why This Project</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning--preprocessing">Data Cleaning &Preprocessing</a>
- <a href="#model--fine-tuning">Model & Fine-Tuning</a>
- <a href="#training-configuration">Training Configuration</a>
- <a href="#inference">Inference</a>
- <a href="#results">Results</a>
- <a href="#deployment">Deployment</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#future-improvements">Future Improvements</a>
- <a href="#author">Author</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

This is an abstractive summarizer — it writes new sentences rather than copying lines from the source text, the way extractive summarizers do. The model is a fine-tuned T5-small, trained on the SAMSum dialogue dataset, and it's wrapped in a small FastAPI app so you can paste text into a browser and get a summary back.

---

<h2><a class="anchor" id="why-this-project"></a>Why This Project</h2>


Reading through long chat logs, meeting transcripts, or articles just to pull out the main point takes time. I wanted a model that could do that step automatically, and I wanted it wired up as an API rather than sitting in a notebook, so it could actually be called from something else.

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

Trained on **SAMSum**, a dataset of messenger-style conversations paired with human-written summaries.

- Training set: 14,732 pairs, sampled down to 4,000 for faster fine-tuning
- Validation set: 818 pairs, sampled down to 500
- Columns: `id`, `dialogue`, `summary`

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- Python
- HuggingFace Transformers (`T5Tokenizer`, `T5ForConditionalGeneration`, `Seq2SeqTrainer`)
- PyTorch (CUDA / MPS / CPU)
- T5-small
- FastAPI + Pydantic
- Jinja2 templating
- HTML, CSS, JavaScript
- Jupyter / Google Colab

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>


```
Text-Summarizer-Using-Transformers/
│
├── README.md
├── requirements.txt
│
├── Data/                             # Dataset files
│
├── Images/                           # App screenshot
│
├── Jupyter Notebook/
│   └── Text_Summarizer_Using_Transformers.ipynb
│
├── Scripts/
    ├── saved_summary_model/          # Fine-tuned T5 model + tokenizer
    ├── app.py                        # FastAPI backend & inference logic
    └── index.html                    # Frontend UI
```

---

<h2><a class="anchor" id="data-cleaning--preprocessing"></a>Data Cleaning & Preprocessing</h2>

- Stripped `\r\n` line breaks and collapsed extra whitespace
- Removed any HTML tags in the raw dialogue
- Lowercased everything for consistent tokenization
- Reused the same cleaning function in training, validation, and the live API in `app.py`, so inference sees text in the same shape the model was trained on
- Prefixed every input with `"summarize: "`, following T5's text-to-text convention
- Tokenized dialogues at max length 512 and summaries at max length 80

---

<h2><a class="anchor" id="model--fine-tuning"></a>Model & Fine-Tuning</h2>


Base model is `t5-small`, an encoder-decoder Transformer. The task is framed as text-to-text: `"summarize: <dialogue>"` in, `<summary>` out. Fine-tuned the full model on the SAMSum subset using HuggingFace's `Seq2SeqTrainer`, rather than training from scratch — the pretrained weights already carry a lot of general language understanding, so fine-tuning on a comparatively small dataset was enough to adapt it to dialogue summarization.

---

<h2><a class="anchor" id="training-configuration"></a>Training Configuration</h2>

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-5 |
| Epochs | 8 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Weight decay | 0.01 |
| Eval strategy | Per epoch |
| Save strategy | Per epoch |
| Mixed precision (fp16) | On (CUDA) |
| Best model selection | Lowest eval_loss |

Final training loss after 8 epochs (4,000 steps): **0.7647**

---

<h2><a class="anchor" id="inference"></a>Inference</h2>

Summaries are generated with beam search rather than greedy decoding:

- `num_beams = 6` — considers multiple candidate summaries before picking one
- `no_repeat_ngram_size = 3` — cuts down on repeated phrases
- `length_penalty = 2.5` — nudges the model toward more concise output
- `min_length` / `max_length` — bounds on summary size
- `early_stopping = True` — stops once all beams hit an end token

---

<h2><a class="anchor" id="results"></a>Results</h2>
 
The fine-tuned model successfully generalizes beyond the SAMSum dialogue format to **general long-form text**, such as news-style paragraphs. Example (unseen input, out-of-domain from training data):

> **Input:** a multi-paragraph report on AI adoption across industries and the ethical questions around it.
>
> **Generated Summary (model output, condensed to a single coherent paragraph):** *"reports suggest that ai adoption has significantly increased over the past few years. experts are looking ahead to ensure that the systems are developed and used in a safe and beneficial way."*

It's not copying sentences from the input — it's compressing the idea into new phrasing. That's a reasonable sign the model learned to summarize rather than just memorize the training set, though a proper ROUGE score against held-out data would say more.

---

<h2><a class="anchor" id="deployment"></a>Deployment</h2>

The model sits behind a small FastAPI app:

- `GET /` serves the HTML page (Jinja2 template)
- `POST /summarize/` takes `{"dialogue": "..."}` as JSON, validated with Pydantic, and returns the generated summary
- Device selection falls back automatically: CUDA → MPS → CPU
- The frontend is plain JS with `fetch()`, a loading state, and basic error handling

Nothing fancy — just enough to move the model out of a notebook and into something you can actually hit with a request.

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. Clone the repo:

```bash
git clone https://github.com/Tisha34/Abstractive-Text-Summarizer-T5-Transformers.git
cd Abstractive-Text-Summarizer-T5-Transformers
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the fine-tuned model is at `./saved_summary_model` (or retrain it using the notebook in `Notebook/`)

4. Run the app:

```bash
uvicorn app:app --reload
```

5. Open `http://127.0.0.1:8000/` and try it out.

---

<h2><a class="anchor" id="future-improvements"></a>Future Improvements</h2>

- Actually compute ROUGE-1 / ROUGE-2 / ROUGE-L on the validation set instead of eyeballing outputs
- Try larger models (`t5-base`, BART, PEGASUS) to see if summary quality improves
- Handle longer documents through chunking or hierarchical summarization
- Dockerize the app and deploy it somewhere (Render, AWS, HuggingFace Spaces)
- Add basic rate-limiting if this ever gets exposed publicly
- Batch endpoint for summarizing multiple documents in one call

---
<h2><a class="anchor" id="author"></a>Author</h2>

**Tisha Gandhi**

Data Analyst | AI & Machine Learning Enthusiast

📧 Email: gandhitishav@gmail.com

🔗 [LinkedIn](www.linkedin.com/in/tisha-gandhi-994b4a24a)