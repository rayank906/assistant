from pathlib import Path
import os
import streamlit as st

from answer import answer_question, build_knowledge_base

st.set_page_config(page_title="EECS 280 Notes Assistant", page_icon="📚")
st.title("EECS 280 Notes Assistant")
st.caption("Ask a question about the course notes. Answers include references to the PDF.")


@st.cache_resource(show_spinner=False)
def load_notes(pdf_path, modified_ns):
    return build_knowledge_base(pdf_path)


with st.form("question_form"):
    question = st.text_area("Your question", placeholder="How do pointers differ from references?", max_chars=2000)
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        st.session_state.pop("result", None)
        st.session_state.pop("answered_question", None)
        try:
            pdf = Path(os.getenv("PDF_PATH", str(Path(__file__).with_name("eecs280notes.pdf"))))
            with st.spinner("Preparing notes and finding an answer…"):
                knowledge = load_notes(str(pdf.resolve()), pdf.stat().st_mtime_ns)
                result = answer_question(question, knowledge)
            st.session_state.result = result
            st.session_state.answered_question = question.strip()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

if "result" in st.session_state:
    st.markdown(f"**Question:** {st.session_state.answered_question}")
    st.markdown(st.session_state.result["answer"])
    for source in st.session_state.result["sources"]:
        title = source["section"] or source["chapter"]
        with st.expander(f"[{source['id']}] {title} · PDF pages {source['page_start']}–{source['page_end']}"):
            st.caption(source["chapter"] + (" · Selected excerpt" if source["excerpt"] else " · Full section"))
            st.text(source["text"])
