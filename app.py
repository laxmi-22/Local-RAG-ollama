# import rquired libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings,ChatOllama
from langchain_community.vectorstores import FAISS
#from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
#from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import re
import fitz  # PyMuPDF

# Emoji pattern
emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)

# Removing header,footer,emojis to reduce noise in retrival process
def clean_page_text(page, header_ratio=0.08, footer_ratio=0.08):
    rect = page.rect
    body_area = fitz.Rect(rect.x0,
                          rect.y0 + rect.height * header_ratio,
                          rect.x1,
                          rect.y1 - rect.height * footer_ratio)
    
    text = page.get_text("text", clip=body_area)
    
    # Remove emojis  
    text = emoji_pattern.sub(r'', text)        
    return text

# step 1 - preprocessing file and return list of docuemnts
def preprocess_pdf(input_pdf):
    doc = fitz.open(input_pdf)
    cleaned_pages = []
    for page in doc:
        cleaned_pages.append(clean_page_text(page))  # calling clean_page_text
    
    # convert list to list of Document object
    docs = [Document(page_content=text, metadata={"page": i})
         for i, text in enumerate(cleaned_pages)]
    
    return docs


# step 2 - splitting text (chunking), used RecursiveCharacterTextSplitter
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    chunks = []
    for doc in documents:
        page_chunks = text_splitter.split_text(doc.page_content)
        for chunk in page_chunks:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )
    return chunks



# Step 3: generate embeddings using OllamaEmbedings
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="llama3") #just use mistral its run on local btw
    vectorstore = FAISS.from_documents(chunks, embeddings)    
    return vectorstore


# Step 4: Build the RAG pipeline
def build_rag():
    llm = OllamaLLM(model="llama3")  # Specify llama3 model

    
    prompt=PromptTemplate(input_variables=["context","question"],
                          template = """
                                You are a strict document-based assistant.

                                Rules:
                                - Use ONLY the information present in the context
                                - Do NOT add explanations
                                - Do NOT use external knowledge
                                - You may list multiple items if they appear in different parts of the context
                                - If the answer is not explicitly stated, say "I don't know"

                                Context:
                                {context}

                                Question:
                                {question}

                                Answer (copy or lightly rephrase from context only):
                                """)
    qa_chain=prompt | llm
    return qa_chain


# trimming final retrived text
def get_context_string(docments):
    context_text=""
    for doc in docments:
        context_text+=doc.page_content + "\n"
    return context_text.strip()


# main program
def main():
    pdf_path = "data/cric.pdf"  # Path to your PDF file 

    documents = preprocess_pdf(pdf_path)
    print("document load completed")
    
    chunks = split_text(documents)
    print("split text completed")

    vectorstore = create_vectorstore(chunks)
    print("vector store completed")

   # Use MMR to improve diversity and reduce redundant chunks
    retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
    
    qa_chain = build_rag()
    print("build rag completed")
    

     # Chat with the PDF
    while True:
        query = input("\nAsk a question about the document : ")
        if query.lower() == "exit":
            break

        retrived_docs=retriver.invoke(query)
        final_retrived_content=get_context_string(retrived_docs)

        response=qa_chain.invoke({
            "question":query,
            "context":final_retrived_content
        })

        print(f"\nBot Answer : {response}")
    

if __name__ == "__main__":
    main() # calling main()