"""Data Splitter module for chunking documents into smaller, semantically coherent segments.
Supports Recursive Character Text Splitting, Sentence-based splitting, and Token-aware splitting with overlap.
"""
import re
from typing import List, Optional
import uuid

from rag.document import Document


class DataSplitter:
    """Splits documents into smaller chunks for optimal embedding and retrieval."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function=len,
    ):
        """Initialize the DataSplitter.

        Args:
            chunk_size: Maximum chunk size (measured by length_function, default characters).
            chunk_overlap: Overlap between adjacent chunks.
            separators: Hierarchy of split delimiters from largest to smallest.
            length_function: Function measuring segment length (default len).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function

    def split_text(self, text: str) -> List[str]:
        """Split a string into chunks using recursive separator hierarchy."""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Core recursive splitting algorithm."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        # Find the highest-priority separator that occurs in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = self._split_on_separator(text, separator)

        # Merge pieces into chunks up to chunk_size
        good_splits: List[str] = []
        for piece in splits:
            if self.length_function(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(piece)
                else:
                    sub_chunks = self._split_text_recursive(piece, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return [c.strip() for c in final_chunks if c.strip()]

    def _split_on_separator(self, text: str, separator: str) -> List[str]:
        if separator:
            return text.split(separator)
        return list(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge short splits together up to chunk_size while respecting chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for piece in splits:
            piece_len = self.length_function(piece)
            sep_len = self.length_function(separator) if current_doc else 0

            if total + piece_len + sep_len > self.chunk_size:
                if total > 0:
                    joined = separator.join(current_doc)
                    if joined.strip():
                        docs.append(joined.strip())

                    # Apply overlap: keep pieces from the tail of current_doc
                    while total > self.chunk_overlap or (
                        total + piece_len + sep_len > self.chunk_size and total > 0
                    ):
                        removed = current_doc.pop(0)
                        total -= self.length_function(removed) + (
                            self.length_function(separator) if current_doc else 0
                        )

                current_doc.append(piece)
                total += piece_len + (self.length_function(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(piece)
                total += piece_len + sep_len

        if current_doc:
            joined = separator.join(current_doc)
            if joined.strip():
                docs.append(joined.strip())

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Document objects into chunked Document objects.
        
        Preserves original metadata and adds chunk-specific fields.
        """
        chunked_docs: List[Document] = []

        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue

            # If document is already small enough, keep as single chunk
            if self.length_function(text) <= self.chunk_size:
                chunk_meta = {
                    **doc.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_char_count": len(text),
                    "parent_id": doc.id,
                }
                chunked_docs.append(
                    Document(
                        id=f"{doc.id}_chk0",
                        page_content=text,
                        metadata=chunk_meta,
                    )
                )
                continue

            chunks = self.split_text(text)
            total_chunks = len(chunks)

            for idx, chunk_text in enumerate(chunks):
                chunk_meta = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_char_count": len(chunk_text),
                    "parent_id": doc.id,
                }
                chunk_id = f"{doc.id}_chk{idx}"
                chunked_docs.append(
                    Document(
                        id=chunk_id,
                        page_content=chunk_text,
                        metadata=chunk_meta,
                    )
                )

        return chunked_docs

