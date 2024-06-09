import os
from flask import Flask, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

folder_path = "db"

google_api_key=os.getenv("GOOGLE_API_KEY")

cached_llm = ChatGoogleGenerativeAI(google_api_key=google_api_key,model="gemini-pro",convert_system_message_to_human=True)
memory = ConversationSummaryBufferMemory(llm=cached_llm, max_token_limit=200)


embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80
)

contextualize_q_system_prompt = """Given a conversation history and the user's latest question, \
which may refer to previous conversation context, rephrase the question so it's clear without \
needing the full conversation history. Do NOT provide an answer, simply rephrase if necessary, \
otherwise keep the question unchanged."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)



raw_prompt = """ 
[INST] <s>
**Your name is campusAI. You are a user support chatbot for J.C. Bose University.** Use the provided {context}  to accurately 
answer questions about the university's history, departments, courses,
 facilities, admissions, events, and contact details. 
 Ensure your responses are precise, concise, and relevant .
 [INST]
 If you cannot find an answer in the PDF, 
inform the replying "For more Info , Visit  https://www.jcboseust.ac.in/".
[/INST]

**Remember:**
* Use natural language and avoid technical jargon.
* Be polite and helpful.
* Provide clear and concise answers.
* limit answer to at most 40 words.
[/INST]

[INST]
{input}
   Context: {context}
   Answer:
[/INST]
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", raw_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@app.route('/')
def welcome():
    return "hello"

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response.content}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query:{query}")

    print("Loading Vector Store...")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("creating chain...")
    retriver = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.1,
        },
    )

    history_aware_retriever = create_history_aware_retriever(
        cached_llm, retriver, contextualize_q_prompt
    )

    document_chain = create_stuff_documents_chain(cached_llm, qa_prompt)
    chain = create_retrieval_chain(history_aware_retriever, document_chain)

    chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = chain.invoke({"input": query}, config={
        "configurable": {"session_id": "abc123"}
    }, )
    print(result)

    response_answer = {"answer": result['answer']}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
