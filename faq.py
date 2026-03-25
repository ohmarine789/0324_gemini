import streamlit as st
import json
import os

# 1. 외부 JSON 파일 로드 함수 (캐싱 적용)
@st.cache_data
def load_faq_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return {"faq": []}

# 2. 페이지 설정
st.set_page_config(page_title="전통시장 지원사업 FAQ", page_icon="📂")

# 3. 데이터 불러오기
# 파일 경로를 지정합니다. (파일명: faq_data.json)
DATA_PATH = "./faq/faq.json"
faq_data = load_faq_data(DATA_PATH)

# 4. 메인 화면 구성
st.title("📂 외부 JSON 연동 FAQ 시스템")
st.info("`faq.json` 파일을 수정하면 웹 페이지에 즉시 반영됩니다.")
st.markdown("---")

# 5. FAQ 리스트 출력
if faq_data["faq"]:
    for i, item in enumerate(faq_data["faq"]):
        with st.expander(f"Q{i+1}. {item['question']}"):
            st.write(f"**A.** {item['answer']}")
else:
    st.warning("표시할 FAQ 데이터가 없습니다.")

# 6. 관리자용: 데이터 원본 확인 (선택 사항)
if st.checkbox("JSON 원본 데이터 보기"):
    st.json(faq_data)