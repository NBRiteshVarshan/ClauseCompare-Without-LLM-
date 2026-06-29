import re
import fitz  # PyMuPDF
import docx2txt
from typing import List, Dict, Any

class ClauseExtractor:
    """Extract clauses from legal documents using text-block strategy"""

    def __init__(self, min_clause_length: int = 60, merge_threshold: int = 30):
        self.min_clause_length = min_clause_length
        self.merge_threshold = merge_threshold

    def extract_from_pdf(self, file_content: bytes) -> List[Dict[str, Any]]:
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self.extract_clauses(text)
        except Exception as e:
            raise Exception(f"Failed to extract from PDF: {str(e)}")

    def extract_from_docx(self, file_content: bytes) -> List[Dict[str, Any]]:
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            text = docx2txt.process(tmp_path)
            os.unlink(tmp_path)
            return self.extract_clauses(text)
        except Exception as e:
            raise Exception(f"Failed to extract from DOCX: {str(e)}")

    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = self._clean_noise(text)
        segments = re.split(r'\n\s*\n', text)
        cleaned = []
        for seg in segments:
            seg = seg.strip()
            seg = re.sub(r'\s+', ' ', seg)
            if seg:
                cleaned.append(seg)
        merged = self._merge_short_segments(cleaned)
        clauses = []
        for idx, seg in enumerate(merged):
            if self._is_valid_clause(seg):
                clauses.append({
                    'number': str(idx + 1),
                    'text': seg,
                    'metadata': {
                        'word_count': len(seg.split()),
                        'char_count': len(seg),
                        'has_conditions': self._check_conditions(seg),
                        'has_exceptions': self._check_exceptions(seg),
                        'is_title': self._is_likely_title(seg)
                    }
                })
        for i, c in enumerate(clauses):
            c['number'] = str(i + 1)
        return clauses

    def _clean_noise(self, text: str) -> str:
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        text = re.sub(r'Confidential\s*[-–]\s*Draft', '', text, flags=re.IGNORECASE)
        return text

    def _merge_short_segments(self, segments: List[str]) -> List[str]:
        if not segments:
            return []
        merged = []
        i = 0
        while i < len(segments):
            current = segments[i]
            if len(current) < self.merge_threshold and i + 1 < len(segments):
                merged.append(current + " " + segments[i+1])
                i += 2
            else:
                merged.append(current)
                i += 1
        return merged

    def _is_valid_clause(self, text: str) -> bool:
        if len(text) < self.min_clause_length:
            return False
        if re.match(r'^[\d\s\.\,\;\:\-]+$', text):
            return False
        return True

    def _is_likely_title(self, text: str) -> bool:
        if len(text) < 50 and (text.isupper() or text.endswith(':') or text.endswith('.')):
            return True
        return False

    def _check_conditions(self, text: str) -> bool:
        words = ['if', 'provided', 'unless', 'subject to', 'when', 'where', 'in the event', 'upon']
        return any(w in text.lower() for w in words)

    def _check_exceptions(self, text: str) -> bool:
        words = ['except', 'excluding', 'other than', 'notwithstanding', 'unless', 'without prejudice']
        return any(w in text.lower() for w in words)


def get_document_summary(clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not clauses:
        return {'total': 0, 'avg_length': 0, 'has_conditions': 0, 'has_exceptions': 0}
    total = len(clauses)
    avg_len = sum(c['metadata']['word_count'] for c in clauses) / total
    cond = sum(1 for c in clauses if c['metadata'].get('has_conditions', False))
    exc = sum(1 for c in clauses if c['metadata'].get('has_exceptions', False))
    return {
        'total': total,
        'avg_length': round(avg_len, 2),
        'has_conditions': cond,
        'has_exceptions': exc
    }