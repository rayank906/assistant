import streamlit as st

from answer import answer_question, build_course_knowledge_base, course_documents

st.set_page_config(page_title="EECS 280 Notes Assistant", page_icon="📚")
st.title("EECS 280 Notes Assistant")
course = "280"
st.caption(f"Ask about EECS {course}. Answers use only this course's notes.")

if st.session_state.get("active_course") != course:
    st.session_state.pop("result", None)
    st.session_state.pop("answered_question", None)
    st.session_state.active_course = course


@st.cache_resource(show_spinner=False)
def load_notes(course, documents):
    # Both course and its file manifest form the cache key.
    return build_course_knowledge_base(course)


with st.form("question_form"):
    question = st.text_area("Your question", placeholder="Ask about EECS 280…", max_chars=2000, key=f"question_{course}")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        st.session_state.pop("result", None)
        st.session_state.pop("answered_question", None)
        try:
            with st.spinner(f"Preparing EECS {course} notes and finding an answer…"):
                knowledge = load_notes(course, course_documents(course))
                result = answer_question(question, knowledge)
            st.session_state.result = result
            st.session_state.answered_question = question.strip()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

if "result" in st.session_state:
    st.markdown(f"**Question:** {st.session_state.answered_question}")
    st.markdown(st.session_state.result["answer"])
    for source in st.session_state.result["sources"]:
        title = source["subsection"] or source["section"] or source["chapter"]
        with st.expander(f"[{source['id']}] {title} · {source['source_file']} · PDF pages {source['page_start']}–{source['page_end']}"):
            st.caption(source["chapter"] + (" · Selected excerpt" if source["excerpt"] else " · Full section"))
            st.text(source["text"])
