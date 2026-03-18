"""
data/raw/10k/ 의 15개 10-K TXT 파일을 모두 로드하여
FAISS 인덱스를 구축(또는 기존 인덱스에 병합)합니다.
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from app.api.text_splitter import split_documents

TXT_DIR = "data/raw/10k"
INDEX_PATH = "faiss_index"


def load_txt_files(directory: str) -> list:
    docs = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(directory, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        pages = loader.load()
        # 메타데이터에 파일명(종목) 기록
        for page in pages:
            page.metadata["source_file"] = filename
        docs.extend(pages)
        print(f"  로드: {filename} ({len(pages)}개 문서)")
    return docs


if __name__ == "__main__":
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print(f"1. {TXT_DIR} 에서 TXT 파일 로드 중...")
    docs = load_txt_files(TXT_DIR)
    print(f"   총 {len(docs)}개 문서 로드 완료")

    print("\n2. 청크 분할 중...")
    chunks = split_documents(docs)
    print(f"   총 {len(chunks)}개 청크")

    print("\n3. 벡터 DB 구축/병합 중...")
    new_vectorstore = FAISS.from_documents(chunks, embeddings)

    if os.path.exists(INDEX_PATH):
        print(f"   기존 인덱스({INDEX_PATH}) 발견 → 병합")
        existing = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        existing.merge_from(new_vectorstore)
        existing.save_local(INDEX_PATH)
    else:
        print(f"   새 인덱스 생성 → {INDEX_PATH}")
        new_vectorstore.save_local(INDEX_PATH)

    print(f"\n완료: {INDEX_PATH}/ 에 저장됨")

    print("\n4. 검색 테스트: 'What was the total revenue?'")
    final = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    results = final.similarity_search("What was the total revenue?", k=3)
    for i, r in enumerate(results):
        print(f"\n--- 결과 {i+1} ({r.metadata.get('source_file', '')}) ---")
        print(r.page_content[:200])
