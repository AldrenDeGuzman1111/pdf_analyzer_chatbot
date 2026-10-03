"""PDF Analyzer app.

This application uploads a PDF, extracts text, builds a vector index for retrieval,
creates summaries with the OpenAI API, and answers user questions using the most
relevant document excerpts.
"""

import hashlib
import io
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional runtime dependency
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover - optional runtime dependency
    convert_from_bytes = None


# Keep upload and chunking limits reasonable for browser-based processing.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
CHUNK_SIZE = 1_300
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 6

# Different summary styles use the same underlying PDF content but vary in tone.
SUMMARY_INSTRUCTIONS = {
    "Bullet points": "Write concise bullet points that capture the main facts, findings, decisions, and actions.",
    "Executive": "Write an executive summary for a decision-maker: purpose, major findings, implications, risks, and recommendations.",
    "Detailed": "Write a structured, thorough summary that covers the document's important sections, reasoning, facts, and conclusions.",
}


def initialize_session() -> None:
    # Seed session state with stable defaults so the app can safely reference
    # values before a PDF has been uploaded or processed.
    defaults: dict[str, Any] = {
        "documents": [],
        "active_document_index": 0,
        "document_hash": None,
        "document_name": None,
        "page_count": 0,
        "chunk_count": 0,
        "vector_store": None,
        "chunks": [],
        "summary": None,
        "summary_style": None,
        "messages": [],
        "uploader_revision": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_active_document(state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    current_state = st.session_state if state is None else state
    documents = current_state.get("documents", [])
    index = current_state.get("active_document_index", 0)
    if not documents:
        return None
    if index < 0:
        index = 0
    if index >= len(documents):
        index = len(documents) - 1
    return documents[index]


def sync_active_document_state(document: dict[str, Any] | None) -> None:
    if document is None:
        st.session_state.document_hash = None
        st.session_state.document_name = None
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.vector_store = None
        st.session_state.chunks = []
        st.session_state.summary = None
        st.session_state.summary_style = None
        st.session_state.messages = []
        return

    st.session_state.document_hash = document.get("hash")
    st.session_state.document_name = document.get("name")
    st.session_state.page_count = document.get("page_count", 0)
    st.session_state.chunk_count = document.get("chunk_count", 0)
    st.session_state.vector_store = document.get("vector_store")
    st.session_state.chunks = document.get("chunks", [])
    st.session_state.summary = document.get("summary")
    st.session_state.summary_style = document.get("summary_style")
    st.session_state.messages = document.get("messages", [])


def clear_document(clear_uploader: bool = False, clear_all: bool = True) -> None:
    # Reset the document-specific state when the user clears an uploaded file or
    # when a new PDF replaces the current one.
    if clear_all:
        st.session_state.documents = []
        st.session_state.active_document_index = 0
    else:
        documents = st.session_state.get("documents", [])
        index = st.session_state.get("active_document_index", 0)
        if documents:
            del documents[index]
            if not documents:
                st.session_state.active_document_index = 0
            else:
                st.session_state.active_document_index = min(index, len(documents) - 1)
        st.session_state.documents = documents

    sync_active_document_state(get_active_document())
    if clear_uploader:
        st.session_state.uploader_revision += 1


def get_openai_api_key() -> str | None:
    if api_key := os.getenv("OPENAI_API_KEY"):
        return api_key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        return None


def ocr_pdf_pages(pdf_bytes: bytes, filename: str) -> list[Document]:
    # OCR is a fallback used when the PDF does not contain extractable text.
    # We convert each page to an image, run Tesseract, and wrap the recognized
    # text as LangChain Document objects with page metadata.
    if convert_from_bytes is None or pytesseract is None:
        raise ValueError(
            "OCR requires both pdf2image and pytesseract to be installed. Please install Tesseract OCR and restart the app."
        )

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as exc:  # pragma: no cover - depends on external tool availability
        raise ValueError("OCR could not convert the PDF into images.") from exc

    extracted = []
    for page_number, image in enumerate(pages, start=1):
        try:
            text = pytesseract.image_to_string(image, lang="eng").strip()
        except Exception as exc:  # pragma: no cover - depends on external tool availability
            raise ValueError(f"OCR failed on page {page_number}.") from exc
        if text:
            extracted.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number, "source": filename, "ocr": True},
                )
            )

    if not extracted:
        raise ValueError("No text could be extracted from the PDF, even after OCR.")
    return extracted


def extract_pages(pdf_bytes: bytes, filename: str) -> list[Document]:
    # Read the PDF page by page and collect only text-bearing pages. If the PDF is
    # encrypted, reject it early; otherwise fall back to OCR for scanned files.
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on PDF reader behavior
            raise ValueError("Password-protected PDFs are not supported.") from exc
        if decrypted == 0:
            raise ValueError("Password-protected PDFs are not supported.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number, "source": filename},
                )
            )

    if not pages:
        try:
            pages = ocr_pdf_pages(pdf_bytes, filename)
        except ValueError as exc:
            raise ValueError(
                "No selectable text was found, and OCR could not extract text from this scanned document."
            ) from exc
    return pages


def build_index(pdf_bytes: bytes, filename: str, api_key: str) -> tuple[list[Document], FAISS, int]:
    # Split extracted PDF text into overlapping chunks so semantic retrieval can
    # perform targeted lookups instead of answering from the full document at once.
    pages = extract_pages(pdf_bytes, filename)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return chunks, vector_store, len(pages)


def invoke_text(prompt: ChatPromptTemplate, values: dict[str, str], api_key: str) -> str:
    # Create a minimal LLM call: prompt -> model -> plain text output.
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(values)


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(f"[p. {document.metadata['page']}]\n{document.page_content}" for document in documents)


def document_groups(documents: list[Document], max_characters: int = 24_000) -> list[list[Document]]:
    groups: list[list[Document]] = []
    current: list[Document] = []
    character_count = 0
    for document in documents:
        size = len(document.page_content)
        if current and character_count + size > max_characters:
            groups.append(current)
            current, character_count = [], 0
        current.append(document)
        character_count += size
    if current:
        groups.append(current)
    return groups


def create_summary(documents: list[Document], style: str, api_key: str) -> str:
    # Long PDFs are summarized in batches so the model can focus on manageable
    # chunks before the partial results are merged into one final summary.
    instructions = SUMMARY_INSTRUCTIONS[style]
    groups = document_groups(documents)
    map_prompt = ChatPromptTemplate.from_template(
        """You are analyzing part of a PDF. {instructions}
Use only the supplied text. Keep page references in the form [p. N] for material you mention.

PDF EXCERPT:
{content}"""
    )
    partial_summaries = [
        invoke_text(map_prompt, {"instructions": instructions, "content": format_documents(group)}, api_key)
        for group in groups
    ]
    if len(partial_summaries) == 1:
        return partial_summaries[0]

    reduce_prompt = ChatPromptTemplate.from_template(
        """Create one coherent PDF summary from the partial summaries below.
{instructions}
Preserve useful page citations in the form [p. N]. Do not invent facts or citations.

PARTIAL SUMMARIES:
{content}"""
    )
    return invoke_text(
        reduce_prompt,
        {"instructions": instructions, "content": "\n\n".join(partial_summaries)},
        api_key,
    )


def answer_question(question: str, api_key: str) -> tuple[str, list[Document]]:
    # Search the embedded document chunks for the passages most relevant to the
    # user's question, then answer only from those retrieved excerpts.
    active_document = get_active_document()
    if active_document is None or active_document.get("vector_store") is None:
        raise ValueError("No PDF is currently active. Upload a document to begin.")

    sources = active_document["vector_store"].similarity_search(question, k=RETRIEVAL_COUNT)
    history = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in active_document.get("messages", [])[-8:]
    ) or "No previous conversation."
    prompt = ChatPromptTemplate.from_template(
        """Answer the user's question using only the retrieved PDF excerpts. The conversation is provided only for context.
If the excerpts do not support an answer, say that the uploaded PDF does not provide the answer.
Every factual statement drawn from the PDF must include a page citation in the form [p. N]. Do not invent citations.

CONVERSATION:
{history}

RETRIEVED PDF EXCERPTS:
{sources}

QUESTION: {question}"""
    )
    answer = invoke_text(
        prompt,
        {"history": history, "sources": format_documents(sources), "question": question},
        api_key,
    )
    return answer, sources


def main() -> None:
    # Load environment variables from a local .env file (if present) before
    # checking for the OpenAI key and initializing the Streamlit UI.
    load_dotenv()
    st.set_page_config(page_title="PDF Analyzer", page_icon="PDF", layout="wide")
    initialize_session()

    st.title("PDF Analyzer")
    st.caption("Generate intelligent summaries and ask cited questions about a text-based PDF. Documents stay only in this browser session.")

    api_key = get_openai_api_key()
    if not api_key:
        st.error("Set OPENAI_API_KEY in your environment or Streamlit secrets before using the analyzer.")
        st.stop()

    with st.sidebar:
        st.header("Document")
        uploaded_files = st.file_uploader(
            "Upload one or more text-based PDFs",
            type="pdf",
            help="Maximum size: 200 MB per PDF",
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.uploader_revision}",
        )

        if st.session_state.get("documents"):
            document_names = [document["name"] for document in st.session_state.documents]
            selected_document_index = st.selectbox(
                "Active document",
                options=range(len(document_names)),
                format_func=lambda idx: document_names[idx],
                index=st.session_state.active_document_index,
            )
            if selected_document_index != st.session_state.active_document_index:
                st.session_state.active_document_index = selected_document_index
                sync_active_document_state(get_active_document())

            active_document = get_active_document()
            if active_document:
                st.caption(f"Active: {active_document['name']}")
                st.caption(f"{active_document['page_count']} text pages, {active_document['chunk_count']} indexed chunks")
                if st.button("Clear active document", width="stretch"):
                    clear_document(clear_uploader=True, clear_all=False)
                    st.rerun()

        if st.button("Clear all documents", width="stretch"):
            clear_document(clear_uploader=True, clear_all=True)
            st.rerun()

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error(f"The PDF '{uploaded_file.name}' exceeds the 200 MB upload limit.")
                continue

            pdf_bytes = uploaded_file.getvalue()
            upload_hash = hashlib.sha256(pdf_bytes).hexdigest()
            existing_document_index = next(
                (
                    index
                    for index, document in enumerate(st.session_state.documents)
                    if document.get("hash") == upload_hash
                ),
                None,
            )

            if existing_document_index is not None:
                active_document_index = existing_document_index
            else:
                with st.spinner(f"Extracting text and building the in-memory search index for {uploaded_file.name}..."):
                    try:
                        chunks, vector_store, page_count = build_index(pdf_bytes, uploaded_file.name, api_key)
                    except Exception as error:
                        st.error(f"Could not process '{uploaded_file.name}': {error}")
                        continue

                    active_document_index = len(st.session_state.documents)
                    st.session_state.documents.append(
                        {
                            "hash": upload_hash,
                            "name": uploaded_file.name,
                            "page_count": page_count,
                            "chunk_count": len(chunks),
                            "chunks": chunks,
                            "vector_store": vector_store,
                            "summary": None,
                            "summary_style": None,
                            "messages": [],
                        }
                    )

            st.session_state.active_document_index = active_document_index
            sync_active_document_state(get_active_document())
            st.success(f"'{uploaded_file.name}' is ready for summarization and Q&A.")

    active_document = get_active_document()
    if active_document is None or active_document.get("vector_store") is None:
        st.info("Upload one or more text-based PDFs to begin. Scanned PDFs and password-protected files are not supported.")
        return

    summary_tab, chat_tab = st.tabs(["Summary", "Ask the PDF"])
    with summary_tab:
        style = st.segmented_control(
            "Summary format",
            options=list(SUMMARY_INSTRUCTIONS),
            default="Bullet points",
            required=True,
        )
        if st.button("Summarise this document", type="primary"):
            with st.spinner("Creating the summary..."):
                try:
                    active_document["summary"] = create_summary(active_document["chunks"], style, api_key)
                    active_document["summary_style"] = style
                    sync_active_document_state(active_document)
                except Exception as error:
                    st.error(f"Could not generate the summary: {error}")
        if active_document.get("summary"):
            st.subheader(f"{active_document['summary_style']} summary")
            st.markdown(active_document["summary"])

    with chat_tab:
        if active_document.get("messages"):
            for message in active_document["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        question = st.chat_input(f"Ask a question about {active_document['name']}")
        if question:
            active_document["messages"].append({"role": "user", "content": question})
            sync_active_document_state(active_document)
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the PDF..."):
                    try:
                        answer, _ = answer_question(question, api_key)
                    except Exception as error:
                        answer = f"I could not answer that question: {error}"
                    st.markdown(answer)
            active_document["messages"].append({"role": "assistant", "content": answer})
            sync_active_document_state(active_document)


if __name__ == "__main__":
    main()
