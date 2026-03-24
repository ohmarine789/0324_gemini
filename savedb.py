# 쓸데없는 특수기호를 에스케이프하지 않고 아예 깔끔한 '공백'으로 치환하여 날려버리는 최적화 코드
import hwp_loader 
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

def clean_text(text):
    """
    HWP 추출 텍스트의 악성 찌꺼기를 제거하고, 임베딩에 방해되는 무의미한 특수기호를 
    공백으로 치환하여 벡터 DB의 검색 성능을 극대화합니다.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 임베딩 모델을 터뜨리는 유니코드 대리쌍(Surrogate) 찌꺼기 완벽 제거
    cleaned = re.sub(r'[\ud800-\udfff]', '', text)
    
    # 2. 눈에 보이지 않는 제어문자(Null 등) 제거
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    
    # 3. 🔥 핵심 최적화: 악세사리 기호 완벽 제거 (유저 인사이트 반영)
    # 한글, 영문, 숫자(\w), 공백(\s), 그리고 필수 문장부호(.,?!%()~-)를 제외한 
    # 모든 잡다한 특수기호(◦, ▪, *, ^ 등)를 '공백(띄어쓰기)'으로 바꿔버립니다.
    cleaned = re.sub(r'[^\w\s.,?!%()~-]', ' ', cleaned)
    
    # 4. 특수기호를 지우면서 생긴 다중 공백을 하나의 공백으로 깔끔하게 압축
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def main():
    # --- [Step 1] 데이터 로드 ---
    file_path = "./srcdata/2026년도_전통시장_및_상점가_활성화_지원사업_공고문.hwp"
    print("1. 문서를 로드하는 중...")
    
    raw_data = hwp_loader.extract_text_from_hwp(file_path) 

    # 추출된 데이터를 하나의 거대한 문자열로 합치기
    full_text = ""
    if isinstance(raw_data, list):
        for item in raw_data:
            if hasattr(item, 'page_content'):
                full_text += str(item.page_content) + "\n"
            else:
                full_text += str(item) + "\n"
    else:
        full_text = str(raw_data)

    # 🔥 텍스트 최적화 세탁기 가동
    full_text = clean_text(full_text)

    # --- [Step 2] 문서 분할 (Chunking) ---
    print("2. 문서를 적절한 크기로 분할하는 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_text(full_text)
    
    # 빈 문자열을 걸러내고 유효한 텍스트만 리스트로 생성
    valid_texts = [str(t).strip() for t in texts if t and str(t).strip()]
    print(f" -> 총 {len(valid_texts)}개의 노이즈 없는 순수 조각(Chunk)으로 정제되었습니다.")

    # --- [Step 3] 임베딩 모델 설정 ---
    print("3. 임베딩 모델을 불러오는 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )

    # --- [Step 4] 벡터 DB 구축 (FAISS) ---
    print("4. 벡터 DB를 구축하고 있습니다...")
    # 모든 찌꺼기와 노이즈가 제거되었으므로 완벽하고 빠르게 통과합니다.
    vectorstore = FAISS.from_texts(
        texts=valid_texts, 
        embedding=embeddings
    )

    # --- [Step 5] 벡터 DB 로컬에 저장 ---
    # 완성된 DB를 'my_faiss_index'라는 폴더에 저장합니다.
    vectorstore.save_local("my_faiss_index")
    print("🎉 드디어 벡터 DB 구축 및 로컬 저장이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()