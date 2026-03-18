from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str) -> list:
    """
    SEC 재무제표 PDF 경로를 입력받아 텍스트를 추출합니다.

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        페이지별 Document 리스트 (page_content, metadata 포함)
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    pages = load_pdf(path)
    print(f"총 {len(pages)}페이지 추출됨")
    for i, page in enumerate(pages):
        print(f"\n--- Page {i + 1} ---")
        print(page.page_content[:300])
