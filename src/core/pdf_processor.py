import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean # Optional: for cleaning text
import io
import os # Needed for temporary file handling if partition_pdf requires paths
import pandas as pd
import numpy as np  # Add numpy import for nan values
import re
from pypdf import PdfReader
import camelot
import tempfile
from langchain.docstore.document import Document

def get_pdf_documents(pdf_files: list) -> list[Document]:
    """Turn uploaded PDFs into a list of Documents (text and table chunks)."""
    docs = []
    for uploaded in pdf_files:
        # write to temp file so Camelot can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            path = tmp.name

        # --- extract tables with Camelot ---
        tables = camelot.read_pdf(path, pages="all", flavor="stream")
        for tbl in tables:
            st.write(tbl, unsafe_allow_html=True)
            df = tbl.df
            meta = {
                "source": uploaded.name,
                "page": int(tbl.page),
                "type": "table",
                "table_accuracy": tbl.parsing_report.get("accuracy", None)
            }
            # serialize as HTML instead of CSV
            content = df.to_html(index=False)
            docs.append(Document(page_content=content, metadata=meta))

        # --- extract plain text per page ---
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                docs.append(
                    Document(
                        page_content=txt,
                        metadata={
                            "source": uploaded.name,
                            "page": i,
                            "type": "text"
                        },
                    )
                )
        os.unlink(path)
    return docs

def get_text_chunks_from_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[Document]:
    """Split large text‐only Documents into smaller text chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    out = []
    for doc in docs:
        if doc.metadata.get("type") == "text":
            chunks = splitter.split_text(doc.page_content)
            for c in chunks:
                out.append(Document(page_content=c, metadata=doc.metadata))
        else:
            # tables are kept as‐is
            out.append(doc)
    return out

# def get_pdf_elements(pdf_docs):
#     """Extract text from uploaded PDF documents."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         text += pdf_reader.pages[0].extract_text()
#     return text

def get_text_chunks(text):
    """Split text into chunks for vectorization."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks 