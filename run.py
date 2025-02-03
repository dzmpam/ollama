from langchain import hub
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model = "deepseek-r1:8b"
db_path = "./chroma_db"


llm = OllamaLLM(model=model)
vectorstore = Chroma(
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),    
    persist_directory=db_path
)
retriever = vectorstore.as_retriever()



# setup RAG chain
#chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
chat_prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions first based on the context.
            Context:
            {context}
            
            Question:
            {input}
            
            Answer concisely and accurately in three sentences or less.
            """
        )

llm_chain = create_stuff_documents_chain(llm, chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, llm_chain)


# TEST
while True:
    question = input("Question('exit' to quit): ")
    if question.lower() == "exit":
        break

    # Retrieve the context
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_docs_with_scores = retriever.similarity_search_with_relevance_scores(question)
    print("\nRetrieved documents with similarity scores:")
    for doc, score in retrieved_docs_with_scores:
        print(f"Document: {doc.page_content}")
        print(f"Score: {score:.4f}\n")
    print(retrieved_docs)


    
    # Print the final prompt
    final_prompt = chat_prompt.format(context=retrieved_docs, input=question)
    print("\nFinal Prompt Sent to LLM:")

    print(final_prompt)

    response = llm_chain.invoke({"input": question, "context": retrieved_docs})
    print(f"\nAnswer: {response}\n")
