from typing import Any

import bs4
import chromadb

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import langchain_core
import langchain_community

from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()

# Create a persistent chromaDB client
client = chromadb.PersistentClient(path="C:\\Users\\jnsep\\OneDrive\\Desktop\\Projects\\ChromaDB_storage")


def reset_chroma_client() -> bool:
    """Empties and completely resets the database. ⚠️ This is destructive and not reversible."""
    return client.reset()


def load_webpage(url: str) -> list[langchain_core.documents.base.Document]:
    """Load post title, headers, and content from the full HTML of a page."""
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    return loader.load()


def load_pdf(path: str) -> list[langchain_core.documents.base.Document]:
    """Load the text from a PDF file"""
    loader = PyPDFLoader(path)
    return loader.load_and_split()


def split_text(docs: list[langchain_core.documents.base.Document],
               chunk_size: int = 1000,
               chunk_overlap: int = 200) -> list[langchain_core.documents.base.Document]:
    """Takes a loaded Document and splits it in chunks of a given size with a given overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def index_in_chroma(split_document: list[langchain_core.documents.base.Document]) \
        -> langchain_community.vectorstores.chroma.Chroma:
    """Takes a list of chunks from a split text and returns a chroma vectorstore with the embeddings for each chunk"""
    return Chroma.from_documents(documents=split_document, embedding=OpenAIEmbeddings())


def retriever_from_vectorstore(vectorstore: langchain_community.vectorstores.chroma.Chroma,
                               search_kwargs=None) -> langchain_core.vectorstores.VectorStoreRetriever:
    """Takes a Chroma vectorstore and returns a retriever that uses similarity search"""
    if search_kwargs is None:
        search_kwargs = {"k": 6}
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def get_rag_chain(retriever: langchain_core.vectorstores.VectorStoreRetriever) \
        -> RunnableSerializable[Any, str]:
    """Takes a document retriever and returns a chain to ask questions avout the documents"""

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Usa el contexto proporcionado para responder la pregunta que del final. Si no sabes la respuesta a 
    la pregunta, simplemente responde "No sé", no intentes inventar una respuesta. Si la información del cntexto no 
    es suficiente para responder la pregunta, responde "No tengo información suficiente para responder". Utiliza 
    máximo cuatro (4) oraciones y responde de la manera más concisa posible.

    {context}
    
    Pregunta: {question}

    Respuesta:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


class PDFQA:

    def __init__(self, path, bigger_chunks):
        self.document = load_pdf(path)
        if bigger_chunks:
            self.split_text = split_text(self.document, chunk_size=4000, chunk_overlap=500)
        else:
            self.split_text = split_text(self.document)
        self.vectorstore = index_in_chroma(self.split_text)
        self.retriever = retriever_from_vectorstore(self.vectorstore)
        self.rag_chain = get_rag_chain(self.retriever)

    def ask(self, question):
        print(self.rag_chain.invoke(question))


def PDFSummarize(path: str, lang: str = "en", chain_type: str = "stuff", bigger_chunks: bool = False) -> str:
    # Loader
    loader = PyPDFLoader(path)
    docs = loader.load()

    if chain_type == "stuff":
        prompt_dict = {
            "en": """Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:""",

            "es": """Escribe un resumen conciso de lo siguiente:
            "{text}"

            RESUMEN CONCISO:""",
        }

        prompt_template = prompt_dict[lang]
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        return stuff_chain.run(docs)

    elif chain_type == "map_reduce":

        llm = ChatOpenAI(temperature=0)

        map_template_dict = {
            "en": """The following is a set of documents
            {docs}
            Based on this list of docs, please identify the main themes 
            Helpful Answer:""",

            "es": """La siguiente es una lista de documentos
            {docs}
            Basándote en esta lista, identifica los tópicos principales.
            Respuesta Útil:""",
        }

        map_prompt = PromptTemplate.from_template(map_template_dict[lang])
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        reduce_template_dict = {
            "en": """The following is set of summaries:
            {docs}
            Take these and distill it into a final, consolidated summary of the main themes. 
            Helpful Answer:""",

            "es": """La siguiente es una lista de resúmenes:
            {docs}
            Toma estos resúmenes y destílalos en dos párrafos de un resumen consolidado con los tópicos principales.
            
            Respuesta Útil:""",
        }

        reduce_prompt = PromptTemplate.from_template(reduce_template_dict[lang])

        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=10_000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        if bigger_chunks:
            split_docs = split_text(docs, chunk_size=5000, chunk_overlap=0)
        else:
            split_docs = split_text(docs)

        # split_docs = text_splitter.split_documents(docs)

        return map_reduce_chain.run(split_docs)


class WebQA:

    def __init__(self, url, bigger_chunks):
        self.document = load_webpage(url)
        if bigger_chunks:
            self.split_text = split_text(self.document, chunk_size=4000, chunk_overlap=500)
        else:
            self.split_text = split_text(self.document)
        self.vectorstore = index_in_chroma(self.split_text)
        self.retriever = retriever_from_vectorstore(self.vectorstore)
        self.rag_chain = get_rag_chain(self.retriever)

    def ask(self, question):
        print(self.rag_chain.invoke(question))
