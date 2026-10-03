# PDF Analyzer

A Streamlit proof of concept for text-based PDFs. It creates bullet-point, executive, or detailed summaries and answers conversational questions with page citations.

The application retains the uploaded document, extracted chunks, FAISS index, summary, and conversation only in Streamlit session memory. It does not write the PDF or index to disk. PDF text is sent to OpenAI for embeddings and responses.

## Included features

- Upload and manage multiple PDFs in the same chat session
- Switch between active documents while preserving per-document chat history
- Ask follow-up questions using the recent conversation context
- Get page-number citations in answers and summaries
- Summarise the currently selected document with a dedicated button
- Reuse cached PDF indexes to reduce repeated processing work
- Log key runtime events and display clear errors to the user

## Setup

1. Create and activate a Python virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Set your API key for the current PowerShell session:

   ```powershell
   $env:OPENAI_API_KEY = "your-api-key"
   ```

4. Start the application:

   ```powershell
   streamlit run app.py
   ```

The configured upload limit is 200 MB. Password-protected and scanned/image-only PDFs are intentionally rejected.
