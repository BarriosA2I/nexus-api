"""
Nexus Assistant Unified - Local TF-IDF RAG Service
Zero-cost retrieval-augmented generation using scikit-learn
"""
import os
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Document chunk with metadata"""
    id: str
    content: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int


class LocalRAGService:
    """
    Local TF-IDF based RAG service.

    Features:
    - Zero API cost (runs entirely locally)
    - Fast retrieval (~10-50ms for typical queries)
    - Automatic chunking with overlap
    - Relevance scoring with configurable threshold
    """

    def __init__(
        self,
        knowledge_dir: Optional[Path] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        similarity_threshold: float = None,
        max_results: int = None,
    ):
        self.knowledge_dir = knowledge_dir or settings.knowledge_path
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.similarity_threshold = similarity_threshold or settings.RAG_SIMILARITY_THRESHOLD
        self.max_results = max_results or settings.RAG_MAX_RESULTS

        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.loaded_files: List[str] = []
        self.load_time_ms: Optional[int] = None

        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def load(self) -> bool:
        """
        Load and index all knowledge files.

        Returns True if successful, False otherwise.
        """
        start_time = time.time()

        try:
            # Find knowledge files
            knowledge_path = Path(self.knowledge_dir)
            if not knowledge_path.exists():
                logger.warning(f"Knowledge directory does not exist: {knowledge_path}")
                knowledge_path.mkdir(parents=True, exist_ok=True)
                self._is_loaded = True
                return True

            # Load .txt and .md files
            files = list(knowledge_path.glob("*.txt")) + list(knowledge_path.glob("*.md"))

            if not files:
                logger.warning(f"No knowledge files found in {knowledge_path}")
                self._is_loaded = True
                return True

            # Process each file
            all_chunks = []
            for file_path in files:
                try:
                    chunks = self._process_file(file_path)
                    all_chunks.extend(chunks)
                    self.loaded_files.append(file_path.name)
                    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

            self.chunks = all_chunks

            if self.chunks:
                # Build TF-IDF index
                self._build_index()

            self.load_time_ms = int((time.time() - start_time) * 1000)
            self._is_loaded = True

            logger.info(
                f"RAG loaded: {len(self.chunks)} chunks from {len(self.loaded_files)} files "
                f"in {self.load_time_ms}ms"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load RAG: {e}")
            self._is_loaded = False
            return False

    def _process_file(self, file_path: Path) -> List[Chunk]:
        """Process a single file into chunks"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Clean content
        content = self._clean_text(content)

        # Split into chunks with overlap
        chunks = []
        words = content.split()
        total_words = len(words)

        if total_words == 0:
            return chunks

        chunk_index = 0
        start_word = 0

        while start_word < total_words:
            end_word = min(start_word + self.chunk_size, total_words)
            chunk_words = words[start_word:end_word]
            chunk_content = " ".join(chunk_words)

            # Calculate character positions (approximate)
            start_char = len(" ".join(words[:start_word]))
            end_char = start_char + len(chunk_content)

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(file_path.name, chunk_index, chunk_content)

            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                source_file=file_path.name,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_words),
            ))

            # Move to next chunk with overlap
            start_word = end_word - self.chunk_overlap
            if start_word >= total_words - self.chunk_overlap:
                break
            chunk_index += 1

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/\%\$\#\@\&\*\+\=]', '', text)
        return text.strip()

    def _generate_chunk_id(self, filename: str, index: int, content: str) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{filename}:{index}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _build_index(self):
        """Build TF-IDF index from chunks"""
        if not self.chunks:
            return

        # Create vectorizer with optimized settings
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words="english",
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,  # Apply sublinear tf scaling
        )

        # Build matrix
        documents = [chunk.content for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        logger.info(f"Built TF-IDF index: {self.tfidf_matrix.shape}")

    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Maximum results to return
            threshold: Minimum relevance score (0-1)

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if not self._is_loaded or not self.chunks:
            return []

        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        top_k = top_k or self.max_results
        threshold = threshold or self.similarity_threshold

        start_time = time.time()

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top results above threshold
            results = []
            sorted_indices = np.argsort(similarities)[::-1]

            for idx in sorted_indices[:top_k * 2]:  # Get extra for filtering
                score = float(similarities[idx])
                if score >= threshold:
                    results.append((self.chunks[idx], score))
                if len(results) >= top_k:
                    break

            search_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Search completed in {search_time}ms, found {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get context string and sources for a query.

        Args:
            query: Search query
            max_tokens: Maximum tokens in context (approximate)

        Returns:
            Tuple of (context_string, sources_list)
        """
        results = self.search(query)

        if not results:
            return "", []

        context_parts = []
        sources = []
        token_count = 0  # Approximate: 1 token ≈ 4 chars

        for chunk, score in results:
            chunk_tokens = len(chunk.content) // 4

            if token_count + chunk_tokens > max_tokens:
                break

            context_parts.append(f"[Source: {chunk.source_file}]\n{chunk.content}")
            sources.append({
                "title": chunk.source_file.replace("-", " ").replace("_", " ").title(),
                "chunk_id": chunk.id,
                "relevance": round(score, 3),
                "excerpt": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "file": chunk.source_file,
            })
            token_count += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)

        return context, sources

    def health_info(self) -> Dict[str, Any]:
        """Get health information"""
        return {
            "loaded": self._is_loaded,
            "chunks": len(self.chunks),
            "knowledge_files": self.loaded_files,
            "load_time_ms": self.load_time_ms,
        }


# Global RAG instance
_rag_instance: Optional[LocalRAGService] = None


def get_rag_service() -> LocalRAGService:
    """Get or create RAG service instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LocalRAGService()
    return _rag_instance


def init_rag_service() -> LocalRAGService:
    """Initialize and load RAG service"""
    service = get_rag_service()
    if not service.is_loaded:
        service.load()
    return service
