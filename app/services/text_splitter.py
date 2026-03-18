from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(pages: list) -> list:
    """
    PDF에서 추출한 페이지 리스트를 청크로 분할합니다.

    Args:
        pages: PyPDFLoader로 추출한 Document 리스트

    Returns:
        청크로 분할된 Document 리스트 (각 청크 길이 512 이하)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    return chunks


if __name__ == "__main__":
    import sys
    from app.services.pdf_loader import load_pdf

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    pages = load_pdf(path)
    chunks = split_documents(pages)

    print(f"총 {len(chunks)}개 청크로 분할됨\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i + 1} (길이: {len(chunk.page_content)}) ---")
        print(chunk.page_content[:100])
        print()

    over = [c for c in chunks if len(c.page_content) > 512]
    if over:
        print(f"[경고] 512 초과 청크 {len(over)}개 발견")
    else:
        print("모든 청크 길이 512 이하 - 완료 기준 충족")
