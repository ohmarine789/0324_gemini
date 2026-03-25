import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 텍스트를 추출하고 기본적인 정제를 수행합니다.
    """
    text_list = []
    try:
        # PDF 문서 열기
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 텍스트 추출 (표 형식이나 레이아웃 유지를 원하면 "dict" 등 옵션 가능)
            page_text = page.get_text("text")
            
            if page_text.strip():
                # 불필요한 공백이나 제어 문자 정리
                clean_text = page_text.replace('\x00', '') 
                text_list.append(clean_text)
                
        doc.close()
    except Exception as e:
        raise ValueError(f"PDF 읽기 중 오류 발생: {e}")
        
    return "\n\n".join(text_list)

class PdfLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        text = extract_text_from_pdf(self.file_path)
        metadata = {"source": self.file_path}
        # LangChain의 Document 객체로 반환
        return [Document(page_content=text, metadata=metadata)]

# ----------------- 실행 테스트 -----------------
if __name__ == "__main__":
    # 파일 경로를 실제 PDF 파일명으로 변경하세요.
    file_name = ".\srcdata\XG5000IEC_Manual_V4.1_202510_KR.pdf" 
    
    try:
        print(f"[{file_name}] PDF 텍스트 추출 및 정밀 분석 중...")
        loader = PdfLoader(file_name)
        documents = loader.load()
        
        # 텍스트 청크 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        
        print(f"성공! 총 {len(docs)}개의 조각으로 분할되었습니다.\n")
        print("--- 정제된 텍스트 추출 미리보기 ---")
        if docs:
            print(docs[0].page_content)
        
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")