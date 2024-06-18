from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

import asyncio
import nest_asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

load_dotenv()
folder_path = "db"
cached_llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-pro",
                                    convert_system_message_to_human=True)

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80
)

raw_prompt = PromptTemplate.from_template(
    """ 
    [INST] <s>
    Your name is campusAI. You are a user support chatbot for J.C. Bose University. Use the provided {context}  to accurately 
    answer questions about the university's history, departments, courses,
     facilities, admissions, events, and contact details. 
     Ensure your responses are precise, concise, and relevant .
     [/INST] </s>
     
     [INST]
     If you cannot find an answer in the PDF, 
    inform by replying "For more Info , Visit  https://www.jcboseust.ac.in/".
    [/INST]

    [INST]
    **Remember:**
    * always start with welcome message.
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
)


def askPDFPost(query):

    print("query", query)
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

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriver, document_chain)

    result = chain.invoke({"input": query})
    print(result)
    return result['answer']


def pdfPost(file):
    file_name = file.name
    save_file = file_name
    with open(file_name, "wb") as f:
        f.write(file.getbuffer())
    print(f"File {file_name} saved successfully.")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"docs len={len(chunks)}")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=folder_path)

    response = {"status": "Successfully Uploaded", "filename": file_name}
    return response
