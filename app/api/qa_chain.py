from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app.api.vector_store import load_vectorstore


def build_qa_chain(index_path: str = "faiss_index") -> RetrievalQA:
    vectorstore = load_vectorstore(index_path)
    llm = ChatOpenAI(model="gpt-4o")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    return qa_chain


if __name__ == "__main__":
    qa_chain = build_qa_chain()

    questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "What is the net income?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = qa_chain.invoke(q)
        print(f"A: {result['result']}")
