# pip install olefile langchain-core langchain-community langchain-text-splitters

import olefile
import zlib
import struct
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_hwp(file_path):
    if not olefile.isOleFile(file_path):
        raise ValueError(f"{file_path}는 유효한 HWP 파일이 아닙니다.")
        
    f = olefile.OleFileIO(file_path)
    dirs = f.listdir()
    sections = [d for d in dirs if d[0] == 'BodyText']
    text_list = []
    
    # 💡 16바이트를 차지하는 HWP 확장 컨트롤 코드 목록
    EXTENDED_CONTROLS = {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23}
    
    for section in sections:
        stream = f.openstream(section)
        data = stream.read()
        
        try:
            unpacked_data = zlib.decompress(data, -15)
        except zlib.error:
            unpacked_data = data
            
        i = 0
        while i < len(unpacked_data):
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3FF
            rec_len = (header >> 20) & 0xFFF
            i += 4
            
            if rec_len == 0xFFF:
                rec_len = struct.unpack_from("<I", unpacked_data, i)[0]
                i += 4
                
            if rec_type == 67: # HWPTAG_PARA_TEXT (본문 텍스트)
                rec_data = unpacked_data[i:i+rec_len]
                
                # 💡 핵심 해결 로직: 컨트롤 코드를 식별하고 쓰레기 데이터 건너뛰기
                j = 0
                chars = []
                while j < len(rec_data):
                    if j + 2 > len(rec_data):
                        break
                        
                    char_code = struct.unpack_from("<H", rec_data, j)[0]
                    
                    if char_code in EXTENDED_CONTROLS:
                        # 확장 컨트롤 코드는 16바이트를 차지하므로 스킵 (한자 깨짐 방지)
                        j += 16
                    elif char_code in (9, 10, 13): 
                        # 탭(9), 줄바꿈(10), 단락바꿈(13)은 유지
                        chars.append(chr(char_code))
                        j += 2
                    elif char_code < 32:
                        # 기타 기본 제어 문자는 2바이트 스킵
                        j += 2
                    else:
                        # 정상적인 텍스트
                        chars.append(chr(char_code))
                        j += 2
                        
                clean_text = "".join(chars)
                if clean_text.strip():
                    text_list.append(clean_text)
            
            i += rec_len
            
    return "\n".join(text_list)

class HwpLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        text = extract_text_from_hwp(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

# ----------------- 실행 테스트 -----------------
if __name__ == "__main__":
    file_name = "./srcdata/2026년도_전통시장_및_상점가_활성화_지원사업_공고문.hwp" 
    
    try:
        print(f"[{file_name}] 바이너리 레코드 정밀 분석 중...")
        loader = HwpLoader(file_name)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        print(f"성공! 총 {len(docs)}개의 조각으로 분할되었습니다.\n")
        print("--- 정제된 텍스트 추출 미리보기 ---")
        print(docs[0].page_content)
        
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")