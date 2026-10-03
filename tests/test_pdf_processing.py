from langchain_core.documents import Document

import app


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, *, encrypted: bool, pages):
        self.is_encrypted = encrypted
        self.pages = pages

    def decrypt(self, password: str):
        if password == "":
            return 1
        return 0


def test_extract_pages_allows_empty_password_for_unlocked_pdf(monkeypatch):
    fake_reader = FakeReader(
        encrypted=True,
        pages=[FakePage("Hello from the PDF")],
    )

    monkeypatch.setattr(app, "PdfReader", lambda _: fake_reader)

    pages = app.extract_pages(b"test-bytes", "sample.pdf")

    assert len(pages) == 1
    assert pages[0].page_content == "Hello from the PDF"
    assert pages[0].metadata["page"] == 1


def test_extract_pages_uses_ocr_when_text_is_empty(monkeypatch):
    fake_reader = FakeReader(encrypted=False, pages=[FakePage("")])

    monkeypatch.setattr(app, "PdfReader", lambda _: fake_reader)
    monkeypatch.setattr(
        app,
        "ocr_pdf_pages",
        lambda pdf_bytes, filename: [Document(page_content="OCR result", metadata={"page": 1, "source": filename})],
    )

    pages = app.extract_pages(b"scan-bytes", "scanned.pdf")

    assert len(pages) == 1
    assert pages[0].page_content == "OCR result"


def test_get_active_document_returns_selected_pdf():
    state = {
        "documents": [
            {"name": "first.pdf", "chunks": ["a"], "vector_store": "store-a"},
            {"name": "second.pdf", "chunks": ["b"], "vector_store": "store-b"},
        ],
        "active_document_index": 1,
    }

    active_document = app.get_active_document(state)

    assert active_document["name"] == "second.pdf"
    assert active_document["vector_store"] == "store-b"
