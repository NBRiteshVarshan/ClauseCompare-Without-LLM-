import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from document_processor import ClauseExtractor, get_document_summary
from clause_matcher import LegalClauseMatcher
from utils import format_report, save_report, generate_pdf_report, categorize_results

# Page config
st.set_page_config(
    page_title="Legal Document Comparator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
    .clause-box { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #3B82F6; }
    .exact-box { background-color: #F0FDF4; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #059669; }
    .partial-box { background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #D97706; }
    .unique-box { background-color: #FEF2F2; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #DC2626; }
    .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 0.5rem; text-align: center; }
    .stButton > button { width: 100%; background-color: #1E3A8A; color: white; font-weight: bold; }
    .stButton > button:hover { background-color: #1E40AF; color: white; }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'doc1_clauses' not in st.session_state:
        st.session_state.doc1_clauses = None
    if 'doc2_clauses' not in st.session_state:
        st.session_state.doc2_clauses = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'extraction_config' not in st.session_state:
        st.session_state.extraction_config = {'min_clause_length': 60, 'merge_threshold': 30}

def main():
    st.markdown('<h1 class="main-header">⚖️ Legal Document Comparator</h1>', unsafe_allow_html=True)
    st.markdown("Compare two legal documents and identify differences in clauses")
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📄 Extraction Settings")
        min_len = st.slider("Minimum Clause Length (characters)", 20, 200, 60, 10)
        merge_len = st.slider("Merge Short Segments (characters)", 10, 100, 30, 5)
        st.session_state.extraction_config['min_clause_length'] = min_len
        st.session_state.extraction_config['merge_threshold'] = merge_len

        st.subheader("🎯 Match Settings")
        match_threshold = st.slider(
            "Match Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Minimum similarity to consider a match. Higher = fewer but more confident matches."
        )

        st.divider()

        if st.session_state.doc1_clauses and st.session_state.doc2_clauses:
            st.subheader("📊 Document Summary")
            s1 = get_document_summary(st.session_state.doc1_clauses)
            s2 = get_document_summary(st.session_state.doc2_clauses)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Document 1", s1['total'], "clauses")
                st.caption(f"Avg: {s1['avg_length']} words")
            with c2:
                st.metric("Document 2", s2['total'], "clauses")
                st.caption(f"Avg: {s2['avg_length']} words")

    # Upload columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Document 1")
        doc1 = st.file_uploader("Upload Document 1 (PDF or DOCX)", type=['pdf', 'docx'], key="doc1")
        if doc1 and st.button("📥 Process Document 1", key="process1"):
            with st.spinner("Extracting clauses..."):
                extractor = ClauseExtractor(min_len, merge_len)
                try:
                    data = doc1.read()
                    clauses = extractor.extract_from_pdf(data) if doc1.type == 'application/pdf' else extractor.extract_from_docx(data)
                    st.session_state.doc1_clauses = clauses
                    st.success(f"✅ Extracted {len(clauses)} clauses")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.doc1_clauses:
            with st.expander(f"📋 View Clauses ({len(st.session_state.doc1_clauses)})"):
                for idx, c in enumerate(st.session_state.doc1_clauses[:5]):
                    st.markdown(f"""
                        <div class="clause-box">
                            <strong>Clause {c.get('number', idx+1)}</strong>
                            <span style="color: #6B7280; font-size:0.8rem;">({c['metadata']['word_count']} words)</span><br>
                            {c['text'][:300]}...
                        </div>
                    """, unsafe_allow_html=True)
                if len(st.session_state.doc1_clauses) > 5:
                    st.caption(f"... and {len(st.session_state.doc1_clauses)-5} more")

    with col2:
        st.subheader("📄 Document 2")
        doc2 = st.file_uploader("Upload Document 2 (PDF or DOCX)", type=['pdf', 'docx'], key="doc2")
        if doc2 and st.button("📥 Process Document 2", key="process2"):
            with st.spinner("Extracting clauses..."):
                extractor = ClauseExtractor(min_len, merge_len)
                try:
                    data = doc2.read()
                    clauses = extractor.extract_from_pdf(data) if doc2.type == 'application/pdf' else extractor.extract_from_docx(data)
                    st.session_state.doc2_clauses = clauses
                    st.success(f"✅ Extracted {len(clauses)} clauses")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.doc2_clauses:
            with st.expander(f"📋 View Clauses ({len(st.session_state.doc2_clauses)})"):
                for idx, c in enumerate(st.session_state.doc2_clauses[:5]):
                    st.markdown(f"""
                        <div class="clause-box">
                            <strong>Clause {c.get('number', idx+1)}</strong>
                            <span style="color: #6B7280; font-size:0.8rem;">({c['metadata']['word_count']} words)</span><br>
                            {c['text'][:300]}...
                        </div>
                    """, unsafe_allow_html=True)
                if len(st.session_state.doc2_clauses) > 5:
                    st.caption(f"... and {len(st.session_state.doc2_clauses)-5} more")

    # Compare button
    st.divider()
    _, mid, _ = st.columns([1,2,1])
    with mid:
        compare = st.button("🔄 Compare Documents", use_container_width=True,
                            disabled=not (st.session_state.doc1_clauses and st.session_state.doc2_clauses))

    if compare and st.session_state.doc1_clauses and st.session_state.doc2_clauses:
        with st.spinner("Comparing documents (embedding‑only)... This may take a few minutes."):
            st.session_state.processing = True
            try:
                matcher = LegalClauseMatcher()
                results = matcher.match_documents(
                    st.session_state.doc1_clauses,
                    st.session_state.doc2_clauses,
                    match_threshold=match_threshold,
                    top_k=3
                )
                st.session_state.comparison_results = results
                st.session_state.processing = False
                st.success(f"✅ Comparison complete! Processed {results['total_doc1']+results['total_doc2']} clauses in {results['processing_time']:.2f}s")
            except Exception as e:
                st.session_state.processing = False
                st.error(f"Error: {e}")

    # Display results
    if st.session_state.comparison_results and not st.session_state.processing:
        results = st.session_state.comparison_results
        doc1_clauses = st.session_state.doc1_clauses
        doc2_clauses = st.session_state.doc2_clauses

        # ---- Build categories using the fixed function ----
        exact, partial, unique = categorize_results(results, doc1_clauses, doc2_clauses)

        total_clauses = results['total_doc1'] + results['total_doc2']
        total_matched = len(exact) + len(partial)
        unique_doc1 = sum(1 for u in unique if u['document'] == 'Document 1')
        unique_doc2 = sum(1 for u in unique if u['document'] == 'Document 2')

        # ---- Summary metrics ----
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>{total_clauses}</h3>
                    <p>Total Clauses</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">
                    <h3>{total_matched}</h3>
                    <p>Matched Clauses</p>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>{unique_doc1}</h3>
                    <p>Only in Doc 1</p>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>{unique_doc2}</h3>
                    <p>Only in Doc 2</p>
                </div>
            """, unsafe_allow_html=True)

        # ---- Bar chart ----
        st.subheader("📊 Visual Analysis")
        fig = go.Figure(data=[
            go.Bar(
                x=['Doc 1 Only', 'Doc 2 Only', 'Matching'],
                y=[unique_doc1, unique_doc2, total_matched],
                marker_color=['#EF4444', '#3B82F6', '#22C55E'],
                text=[unique_doc1, unique_doc2, total_matched],
                textposition='auto',
            )
        ])
        fig.update_layout(height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # ---- Three Categories ----
        st.subheader("📂 Clause Categories")
        with st.expander(f"✅ Exact Matches (Similarity ≥ 0.999) — {len(exact)} pairs"):
            if exact:
                for m in exact:
                    st.markdown(f"""
                        <div class="exact-box">
                            <strong>📄 Doc1 – Clause {m['doc1_num']}</strong><br>{m['doc1_text']}<br><br>
                            <strong>📄 Doc2 – Clause {m['doc2_num']}</strong><br>{m['doc2_text']}<br><br>
                            <span style="color:#059669;">✅ Similarity: {m['similarity']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No exact matches found.")

        with st.expander(f"🟡 Partial Matches (0.5 ≤ Similarity < 0.999) — {len(partial)} pairs"):
            if partial:
                for m in partial:
                    st.markdown(f"""
                        <div class="partial-box">
                            <strong>📄 Doc1 – Clause {m['doc1_num']}</strong><br>{m['doc1_text']}<br><br>
                            <strong>📄 Doc2 – Clause {m['doc2_num']}</strong><br>{m['doc2_text']}<br><br>
                            <span style="color:#D97706;">🔶 Similarity: {m['similarity']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No partial matches found.")

        # ---- Unique expander with two sub-sections ----
        with st.expander(f"🔴 Unique Clauses (not matched) — {len(unique)} total"):
            if unique:
                unique_doc1_list = [u for u in unique if u['document'] == 'Document 1']
                unique_doc2_list = [u for u in unique if u['document'] == 'Document 2']

                if unique_doc1_list:
                    st.markdown("**📄 Document 1 Unique:**")
                    for u in unique_doc1_list:
                        st.markdown(f"""
                            <div class="unique-box">
                                <strong>Clause {u['number']}</strong><br>{u['text']}<br>
                                <span style="color:#DC2626;">❌ Best similarity: {u['similarity']:.3f}</span>
                            </div>
                        """, unsafe_allow_html=True)

                if unique_doc2_list:
                    st.markdown("**📄 Document 2 Unique:**")
                    for u in unique_doc2_list:
                        st.markdown(f"""
                            <div class="unique-box">
                                <strong>Clause {u['number']}</strong><br>{u['text']}<br>
                                <span style="color:#DC2626;">❌ Best similarity: {u['similarity']:.3f}</span>
                            </div>
                        """, unsafe_allow_html=True)

                if not unique_doc1_list and not unique_doc2_list:
                    st.info("No unique clauses found.")
            else:
                st.info("All clauses are matched (no unique clauses).")

        # ---- Download reports ----
        st.subheader("📥 Download Reports")
        txt_report = format_report(results)
        json_report = save_report(results)  # returns JSON string
        pdf_report = generate_pdf_report(results, doc1_clauses, doc2_clauses)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                label="📄 PDF Report",
                data=pdf_report,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        with d2:
            st.download_button(
                label="📄 TXT Report",
                data=txt_report,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        with d3:
            st.download_button(
                label="📊 JSON Report",
                data=json_report,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # ---- Processing details ----
        with st.expander("🔧 Processing Details"):
            st.json({
                'extraction': 'Text-block (\\n\\n split)',
                'min_len': min_len,
                'merge_len': merge_len,
                'match_threshold': match_threshold,
                'processing_time': f"{results['processing_time']:.2f}s",
                'matches_found': results['matching_count']
            })

if __name__ == "__main__":
    main()