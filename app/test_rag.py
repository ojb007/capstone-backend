from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("BGE 모델 다운로드 성공!")