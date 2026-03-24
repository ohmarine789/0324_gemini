# Streamlit 기반의 챗봇 인터페이스 코드
# 클라우드 기반의 LLM API(예: https://aistudio.google.com/  제미나이)를 사용하고, 코드를 무료 호스팅 서비스에 배포
# pip install streamlit langchain langchain-community langchain-google-genai langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers olefile

# pip list --format=json

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# OpenAI 대신 Google Gemini 라이브러리 임포트
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import os

st.set_page_config(page_title="지원사업 Q&A 봇 (Gemini)", page_icon="📝")
st.title("전통시장 활성화 지원사업 Q&A 봇 💬 (무료버전)")
st.caption("공고문 기반으로 질문에 답변해 드립니다. (Powered by Gemini)")

@st.cache_resource
def load_rag_pipeline():
    try:
        # 1. 임베딩 모델 (기존 savedb.py에서 사용한 것과 동일해야 함)
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask") 
        
        # 2. FAISS DB 로드
        vectorstore = FAISS.load_local("./my_faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 3. LLM 설정 (Google Gemini 최신 무료 모델 적용)
        # secrets에서 구글 API 키를 가져옵니다. AIzaSyBNGwF3nr85fVvNG6q1_fwhrNHryNYDG1s
        google_api_key = st.secrets["GOOGLE_API_KEY"].apikey
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # <--- 1.5를 2.5로 변경!
            temperature=0, 
            google_api_key=google_api_key
        )

        # 4. 프롬프트 설정
        system_prompt = (
            "당신은 중소벤처기업부의 전통시장 및 상점가 활성화 지원사업 공고문을 안내하는 친절한 챗봇입니다.\n"
            "제공된 문맥(Context)을 바탕으로 사용자의 질문에 정확하고 간결하게 답변하세요.\n"
            "문맥에 없는 내용이라면 '제공된 문서에서는 해당 내용을 찾을 수 없습니다'라고 솔직하게 답변하세요.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # 5. 체인 조립
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        return qa_chain
    except Exception as e:
        st.error(f"DB 또는 모델을 불러오는 중 오류가 발생했습니다: {e}")
        return None

qa_chain = load_rag_pipeline()

# --- 대화 UI 부분 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 2026년도 전통시장 지원사업 공고문에 대해 무엇이든 물어보세요. (Gemini가 답변합니다)"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("질문을 입력하세요 (예: 문화관광형 사업의 지원 한도는?)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if qa_chain is None:
            st.error("RAG 파이프라인이 정상적으로 로드되지 않았습니다.")
        else:
            with st.spinner("문서를 검색하고 답변을 생성 중입니다..."):
                try:
                    response = qa_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("참고한 문서 조각 보기"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**[{i+1}]** {doc.page_content[:150]}...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")


# 📦 설치되는 핵심 라이브러리 설명
# streamlit: 챗봇의 웹 인터페이스(UI)를 만들어주는 라이브러리입니다.

# langchain & langchain-community & langchain-text-splitters: 문서를 자르고, 검색하고, LLM과 연결하는 RAG(검색 증강 생성) 뼈대를 담당합니다.

# langchain-google-genai: Google Gemini API를 LangChain 환경에서 사용할 수 있게 해줍니다. (OpenAI 대신 들어간 핵심 패키지입니다.)

# langchain-huggingface & sentence-transformers: 문서를 벡터 숫자로 변환하는 임베딩(jhgan/ko-sroberta-multitask) 작업을 처리합니다.

# faiss-cpu: 메타(페이스북)에서 만든 빠르고 강력한 벡터 데이터베이스(DB)입니다. 문서를 저장하고 검색하는 데 쓰입니다.

# olefile: HWP 파일의 압축 및 구조를 풀어서 텍스트를 추출할 때(hwp_loader.py) 필수적으로 사용됩니다.

# streamlit run app.py