# text-summarizer-project
AI-powered text summarization using Python, FastAPI and T5 Transformer
# AI-Powered Text Summarizer

An AI-based text summarization web application built using Python, FastAPI, Hugging Face Transformers, and the T5 model.

This application takes lengthy text as input and generates a concise summary using Natural Language Processing (NLP) and Transformer-based deep learning.

## Features

- AI-powered text summarization.
- T5 Transformer model integration.
- FastAPI backend.
- Text preprocessing and cleaning.
- Automatic device selection (CPU, CUDA, or Apple MPS).
- Web-based user interface.
- REST API for text summarization.

## Technologies Used

- Python
- FastAPI
- Hugging Face Transformers
- T5 Transformer
- PyTorch
- Pydantic
- HTML
- Regular Expressions

## Project Structure

```text
textsummerizer/
│
├── app.py
├── index.html
├── requirements.txt
├── .gitignore
│
└── saved_summary_model/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── generation_config.json
```

**Note:** The model weights file (`model.safetensors`) is not included in this GitHub repository because of its large size. A compatible trained T5 model must be placed inside `saved_summary_model/` before running the application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shravan960/text-summarizer-project.git
```

### 2. Navigate to the project directory

```bash
cd text-summarizer-project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the model weights

Place the compatible `model.safetensors` file inside:

```text
saved_summary_model/
```

Ensure the model weights and configuration belong to the same trained T5 model.

### 6. Run the application

```bash
python -m uvicorn app:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000
```

## API Endpoint

### POST `/summarize/`

Accepts dialogue or text and returns an AI-generated summary.

Example request:

```json
{
    "dialogue": "Enter your lengthy text here..."
}
```

Example response:

```json
{
    "summary": "Generated concise summary..."
}
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## How It Works

1. User submits lengthy text through the web interface.
2. FastAPI receives the input.
3. Text preprocessing removes unwanted spaces and formatting.
4. T5 Tokenizer converts text into tokens.
5. The T5 Transformer model generates a summary.
6. The generated summary is returned to the user.

## Future Improvements

- Support for PDF and DOCX files.
- Multilingual text summarization.
- Summary length customization.
- Improved user interface.
- Cloud deployment.
- Download summarized text.

## Author

**Shravan Kumar**

GitHub: https://github.com/shravan960

---

If you find this project useful, feel free to explore the repository!
