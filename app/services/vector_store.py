from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def build_vectorstore(chunks: list, save_path: str = "faiss_index") -> FAISS:
    """
    청크 리스트를 받아 FAISS 벡터 DB를 구축하고 저장합니다.

    Args:
        chunks: split_documents()로 분할된 Document 리스트
        save_path: 벡터 DB 저장 경로

    Returns:
        FAISS vectorstore 객체
    """
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore


def load_vectorstore(save_path: str = "faiss_index") -> FAISS:
    """
    저장된 FAISS 벡터 DB를 불러옵니다.
    """
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    import sys
    from app.services.pdf_loader import load_pdf
    from app.services.text_splitter import split_documents

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"

    print("1. PDF 로드 중...")
    pages = load_pdf(path)

    print("2. 청크 분할 중...")
    chunks = split_documents(pages)
    print(f"   총 {len(chunks)}개 청크")

    print("3. 벡터 DB 구축 중... (시간이 걸릴 수 있어요)")
    vectorstore = build_vectorstore(chunks)
    print("   faiss_index/ 폴더 생성 완료")

    print("\n4. 검색 테스트: 'revenue'")
    results = vectorstore.similarity_search("revenue", k=3)
    if results:
        for i, r in enumerate(results):
            print(f"\n--- 결과 {i + 1} ---")
            print(r.page_content[:200])
    else:
        print("결과 없음")
