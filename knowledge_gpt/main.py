import streamlit as st
from openai.error import OpenAIError

from knowledge_gpt.components.sidebar import sidebar
from knowledge_gpt.utils import (
    embed_docs,
    get_answer,
    get_sources,
    parse_docx,
    parse_pdf,
    parse_txt,
    search_docs,
    text_to_docs,
    wrap_text_in_html,
    tweet,
)


def clear_submit():
    st.session_state["submit"] = False


st.set_page_config(page_title="KnowledgeGPT", page_icon="📖", layout="wide")
st.header("📖KnowledgeGPT")

sidebar()

uploaded_files = st.file_uploader(
    "Upload a pdf, docx, txt or md file, you can also upload multiple files",
    type=["pdf", "docx", "txt", "md"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
    accept_multiple_files=True,
)

index = None
docs = []
#parsed_text = {}
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            text = parse_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = parse_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = parse_txt(uploaded_file)
        elif uploaded_file.name.endswith(".md"):
            text = parse_txt(uploaded_file)
        else:
            raise ValueError("File type not supported!")
        docs += text_to_docs(text)
        #parse_txt[uploaded_file.name] = text
    try:
        with st.spinner("Indexing document... This may take a while⏳"):
            index = embed_docs(docs)
        st.session_state["api_key_configured"] = True
    except OpenAIError as e:
        st.error(e._message)

search = st.text_area("Search the document with some keywords or a short sentence", on_change=clear_submit)
# with st.expander("Advanced Options"):
#    show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    # show_full_doc = st.checkbox("Show parsed contents of the document")

#if show_full_doc and docs:
#    with st.expander("Document"):
       # Hack to get around st.markdown rendering LaTeX
#        st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not st.session_state.get("api_key_configured"):
        st.error("Please configure your OpenAI API key!")
    elif not index:
        st.error("Please upload a document!")
    elif not query:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        answer_col, sources_col = st.columns(2)

        try:
            sources = search_docs(index, search)

            with sources_col:
                st.markdown("#### Sources")
                for source in sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")        

            with answer_col:
                task = st.text_area("Ask AI to help you with a editorial task, based on the source documents", on_change=clear_submit)
                if task:
                    answer = get_answer(sources, task)
                    answer_text = answer["answer"]
                    if "answer_text" not in st.session_state:
                        st.session_state["answer_text"] = answer_text
                
                    st.markdown("#### Original Answer")
                    st.markdown(st.session_state["answer_text"])

                    with st.form("edit_form"):
                        if st.session_state.get("edit_answer"):
                            edited_answer = st.text_area("Edit Answer", value=st.session_state["answer_text"])
                            if st.form_submit_button("Finish Editing"):
                                st.session_state["updated_answer_text"] = edited_answer
                                st.session_state["edit_answer"] = False
                                st.experimental_rerun()
                        else:
                            if st.form_submit_button("Edit Answer"):
                                st.session_state["edit_answer"] = True
                                st.experimental_rerun()
                    
                    if "updated_answer_text" in st.session_state:
                        st.markdown("#### Updated Answer")
                        st.markdown(st.session_state["updated_answer_text"])
                
        except OpenAIError as e:
            st.error(e._message)

tweet_button = st.button("Tweet Answer")
if tweet_button:
    if "updated_answer_text" in st.session_state:
        result = tweet(st.session_state["updated_answer_text"])
    elif "answer_text" in st.session_state:
        result = tweet(st.session_state["answer_text"])
    else:
        st.error("Please submit a task first!")