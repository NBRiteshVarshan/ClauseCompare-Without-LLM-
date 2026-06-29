import re
import hashlib
import json
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

# ------------------------------------------------------------
# Helper to recursively convert NumPy types to Python natives
# ------------------------------------------------------------
def sanitize_for_json(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\-\"\'\?\$\%\&\(\)]', '', text)
    return text.strip()

def generate_clause_id(text: str, document_name: str) -> str:
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    doc_hash = hashlib.md5(document_name.encode()).hexdigest()[:6]
    return f"{doc_hash}_{content_hash}"

def format_report(results: Dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("LEGAL DOCUMENT COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append("\n📄 CLAUSES IN DOCUMENT 1 BUT NOT IN DOCUMENT 2")
    lines.append("-" * 60)
    for idx, clause in enumerate(results.get('only_in_doc1', []), 1):
        lines.append(f"{idx}. {clause['text']}")
        if clause.get('closest_match'):
            lines.append(f"   → Closest match in Doc2: {clause['closest_match'][:100]}...")
            lines.append(f"   → Similarity score: {clause.get('similarity', 0):.2f}")
        lines.append("")

    lines.append("\n📄 CLAUSES IN DOCUMENT 2 BUT NOT IN DOCUMENT 1")
    lines.append("-" * 60)
    for idx, clause in enumerate(results.get('only_in_doc2', []), 1):
        lines.append(f"{idx}. {clause['text']}")
        if clause.get('closest_match'):
            lines.append(f"   → Closest match in Doc1: {clause['closest_match'][:100]}...")
            lines.append(f"   → Similarity score: {clause.get('similarity', 0):.2f}")
        lines.append("")

    lines.append("\n📊 SUMMARY STATISTICS")
    lines.append("-" * 60)
    lines.append(f"Total clauses in Document 1: {results.get('total_doc1', 0)}")
    lines.append(f"Total clauses in Document 2: {results.get('total_doc2', 0)}")
    lines.append(f"Clauses only in Document 1: {len(results.get('only_in_doc1', []))}")
    lines.append(f"Clauses only in Document 2: {len(results.get('only_in_doc2', []))}")
    lines.append(f"Matching clauses found: {results.get('matching_count', 0)}")
    lines.append(f"Comparison time: {results.get('processing_time', 0):.2f} seconds")
    lines.append("=" * 80)
    return "\n".join(lines)

def save_report(results: Dict, filename: str = None) -> str:
    clean_results = {
        'only_in_doc1': results.get('only_in_doc1', []),
        'only_in_doc2': results.get('only_in_doc2', []),
        'total_doc1': results.get('total_doc1', 0),
        'total_doc2': results.get('total_doc2', 0),
        'matching_count': results.get('matching_count', 0),
        'processing_time': results.get('processing_time', 0),
        'timestamp': datetime.now().isoformat()
    }
    sanitised = sanitize_for_json(clean_results)
    json_str = json.dumps(sanitised, indent=2)
    if filename:
        with open(filename, 'w') as f:
            f.write(json_str)
    return json_str

# ------------------------------------------------------------
# FIXED categorisation – only actual matches go to exact/partial
# ------------------------------------------------------------
def categorize_results(results, doc1_clauses, doc2_clauses):
    """
    Returns three lists:
      - exact_matches   (similarity >= 0.999 AND found_match == True)
      - partial_matches (0.5 <= similarity < 0.999 AND found_match == True)
      - unique_clauses  (all other clauses: unmatched from both docs)
    """
    exact = []
    partial = []
    unique = []

    # Build set of doc2 indices that are actually matched
    matched_doc2_indices = set()
    for detail in results.get('matching_details', []):
        if detail.get('found_match') and detail.get('best_match'):
            idx = detail['best_match']['clause_idx']
            if idx >= 0:
                matched_doc2_indices.add(idx)

    # Process doc1 clauses – only matched ones go to exact/partial
    for i, clause1 in enumerate(doc1_clauses):
        detail = results['matching_details'][i] if i < len(results.get('matching_details', [])) else {}
        sim = detail.get('top_similarity', 0.0)
        idx = detail.get('top_match_idx', -1)
        found = detail.get('found_match', False)

        if found and sim >= 0.999 and idx >= 0 and idx < len(doc2_clauses):
            clause2 = doc2_clauses[idx]
            exact.append({
                'doc1_num': clause1.get('number', str(i+1)),
                'doc1_text': clause1['text'],
                'doc2_num': clause2.get('number', str(idx+1)),
                'doc2_text': clause2['text'],
                'similarity': sim
            })
        elif found and sim >= 0.5 and idx >= 0 and idx < len(doc2_clauses):
            clause2 = doc2_clauses[idx]
            partial.append({
                'doc1_num': clause1.get('number', str(i+1)),
                'doc1_text': clause1['text'],
                'doc2_num': clause2.get('number', str(idx+1)),
                'doc2_text': clause2['text'],
                'similarity': sim
            })
        else:
            # unmatched doc1
            unique.append({
                'text': clause1['text'],
                'document': 'Document 1',
                'number': clause1.get('number', str(i+1)),
                'similarity': sim
            })

    # Process doc2 clauses – add all that are NOT matched
    for j, clause2 in enumerate(doc2_clauses):
        if j not in matched_doc2_indices:
            doc2_best_sims = results.get('doc2_best_similarities', [])
            sim = doc2_best_sims[j] if j < len(doc2_best_sims) else 0.0
            unique.append({
                'text': clause2['text'],
                'document': 'Document 2',
                'number': clause2.get('number', str(j+1)),
                'similarity': sim
            })

    unique.sort(key=lambda x: x['similarity'], reverse=True)
    return exact, partial, unique

# ------------------------------------------------------------
# PDF report generator (uses the same categorisation)
# ------------------------------------------------------------
def generate_pdf_report(results, doc1_clauses, doc2_clauses,
                        doc1_name="Document 1", doc2_name="Document 2") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    clause_style = ParagraphStyle(
        'ClauseStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=20,
        spaceAfter=6,
        fontName='Helvetica'
    )
    
    story = []
    story.append(Paragraph("Legal Document Comparison Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.25*inch))
    
    exact, partial, unique = categorize_results(results, doc1_clauses, doc2_clauses)
    summary_data = [
        ['Metric', 'Value'],
        ['Total Clauses in Document 1', str(len(doc1_clauses))],
        ['Total Clauses in Document 2', str(len(doc2_clauses))],
        ['Exact Matches', str(len(exact))],
        ['Partial Matches', str(len(partial))],
        ['Unique Clauses', str(len(unique))],
        ['Processing Time', f"{results['processing_time']:.2f} seconds"],
    ]
    table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25*inch))
    
    story.append(PageBreak())
    story.append(Paragraph(f"Exact Matches (Similarity ≥ 0.999) — {len(exact)} pairs", heading_style))
    for match in exact:
        story.append(Paragraph(f"Doc1 Clause {match['doc1_num']}: {match['doc1_text'][:200]}...", clause_style))
        story.append(Paragraph(f"Doc2 Clause {match['doc2_num']}: {match['doc2_text'][:200]}...", clause_style))
        story.append(Paragraph(f"Similarity: {match['similarity']:.3f}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    if not exact:
        story.append(Paragraph("No exact matches found.", normal_style))
    
    story.append(PageBreak())
    story.append(Paragraph(f"Partial Matches (0.5 ≤ Similarity < 0.999) — {len(partial)} pairs", heading_style))
    for match in partial:
        story.append(Paragraph(f"Doc1 Clause {match['doc1_num']}: {match['doc1_text'][:200]}...", clause_style))
        story.append(Paragraph(f"Doc2 Clause {match['doc2_num']}: {match['doc2_text'][:200]}...", clause_style))
        story.append(Paragraph(f"Similarity: {match['similarity']:.3f}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    if not partial:
        story.append(Paragraph("No partial matches found.", normal_style))
    
    story.append(PageBreak())
    story.append(Paragraph(f"Unique Clauses (not matched) — {len(unique)} clauses", heading_style))
    for clause in unique:
        story.append(Paragraph(f"{clause['document']} – Clause {clause['number']}: {clause['text'][:200]}...", clause_style))
        story.append(Paragraph(f"Best similarity: {clause['similarity']:.3f}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    if not unique:
        story.append(Paragraph("All clauses were matched.", normal_style))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes