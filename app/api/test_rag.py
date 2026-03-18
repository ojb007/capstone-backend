import sys
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 임베딩 모델 로드
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 2. PDF 로드 (경로를 인자로 받음, 없으면 기본값 사용)
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
print(f"PDF 경로: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 3. 텍스트 분할 (너무 길면 잘라야 해요)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)
print(f"총 {len(docs)}개 청크로 분할됨")

# 4. FAISS 벡터 DB에 저장
vectorstore = FAISS.from_documents(docs, embeddings)
print("벡터 DB 저장 완료!")

# 5. 질문으로 검색
query = "이 PDF는 무엇입니까?"
results = vectorstore.similarity_search(query, k=1)
print(f"\n질문: {query}")
print(f"답변 근거: {results[0].page_content}")