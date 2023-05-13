import re
import os
import pandas as pd
from io import BytesIO
from typing import Any, Dict, List

import docx2txt
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError
from pypdf import PdfReader

from knowledge_gpt.embeddings import OpenAIEmbeddings
from knowledge_gpt.prompts import STUFF_PROMPT


@st.experimental_memo()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.experimental_memo()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


@st.experimental_memo()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache(allow_output_mutation=True)
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache(allow_output_mutation=True, show_spinner=False)
def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.get("OPENAI_API_KEY")
        )  # type: ignore
        index = FAISS.from_documents(docs, embeddings)

        return index


@st.cache(allow_output_mutation=True)
def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    # Query against OpenAI's Chat API to retrieve the most relevant document's page number
    page = get_most_relevant_docs(docs, query)
    # Retrieve the documents from the page number, based on the metadata
    return retrieve_docs(page, index)

def get_most_relevant_docs(docs: List[Document], query: str) -> List[Document]:
    prompt_template = """Use the following pieces of context and provided keywords to find the most relevant document. Notice that the provided context are always five documents, each with page_content and metadata. Format your answer in two sections, the page_content of the document and the metadata of the document. 
    =========
    Context: {context}
    =========
    Keywords: {question}
    =========
    Page Content:
    =========
    Metadata:"""
    qa_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    messages=[
            {"role": "system", "content": "You are a useful AI assistant and you understand how to retrieve document based on relevant keywords."},
            {"role": "user", "content": qa_prompt.format(context=docs, question=query)}
        ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0301',
        messages=messages,
        temperature=0
        )
    
    print(response['choices'][0]['message']['content'])
    return int(re.search(r"'page': (\d+)", response['choices'][0]['message']['content'].split('\n\n')[2]).group(1))

@st.cache(allow_output_mutation=True)
def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    =========
    Context: {context}
    =========
    Question: {question}
    =========
    Final Answer:"""
    qa_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    messages=[
            {"role": "system", "content": "You are a marketing expert and you know how to write engaging twitter threads."},
            {"role": "user", "content": qa_prompt.format(context=docs, question=query)}
        ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0301',
        messages=messages,
        temperature=0
        )
    return {"answer":response['choices'][0]['message']['content'], "sources":docs}


@st.cache(allow_output_mutation=True)
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s['metadata']['source'] for s in answer["sources"]]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])

# A function using tweepy and twitter API to set up a twitter client, keys and secrets are stored in .env file
# given a list of text, seprated by newlines, it will tweet each line in a thread
# it will also store the tweet id, tweet url, and tweet text in a dataframe
# it will also return the dataframe
def tweet(text: str | List[str]) -> pd.DataFrame:
    # set up twitter client
    client = tweepy.Client(consumer_key=os.getenv("TWITTER_API_KEY"),
                       consumer_secret=os.getenv("TWITTER_API_SECRET_KEY"),
                       access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                       access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
                       )

    # if text is a string, split it into a list
    if isinstance(text, str):
        text = text.split("\n")

    # create a dataframe to store tweet id, url, and text
    df = pd.DataFrame(columns=["id", "original_text"])

    # tweet each line in a thread
    for i, line in enumerate(text):
        # if it's the first line, tweet it
        if i == 0:
            response = client.create_tweet(text=line)
            tweet_id = response.data['id']
        # if it's not the first line, reply to the previous tweet
        else:
            response = client.create_tweet(text=line, in_reply_to_tweet_id=tweet_id)
        # add the tweet id, url, and text to the dataframe
    df = df.append({"id": tweet_id, "original_text": " ".join(text)}, ignore_index=True)

    return df

def parse_source_document(source_text):

# Split the text into separate document strings
    document_strings = source_text.split('\n- Document')
# Initialize an empty list to hold the parsed documents
    documents = []
# Parse each document string into a dictionary
    for document_string in document_strings[1:]:
    # Extract the page_content and metadata values from the string
        page_content = document_string.split("page_content='")[1].split("', metadata=")[0]
        metadata_string = document_string.split("metadata=")[1].strip(")")
        metadata = eval(metadata_string) # Parse the metadata string into a dictionary
    
    # Create a dictionary for the document and append it to the documents list
        document = {'page_content': page_content, 'metadata': metadata}
        documents.append(document)
    
    return documents

def retrieve_docs(page, index) -> List[Document]:
    result = []
    for key, value in index.docstore._dict.items():
        if value.metadata['page'] == page:
            result.append(value)
    return result