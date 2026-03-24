## 📦 주요 사용 기술 및 라이브러리 (Dependencies)

이 프로젝트는 원활한 RAG(검색 증강 생성) 파이프라인 구축을 위해 아래의 핵심 라이브러리들을 사용합니다. 

| 라이브러리 (Library) | 핵심 용도 (Purpose) | 버전 (Version) |
| :--- | :--- | :--- |
| **`streamlit`** | 웹 기반 챗봇 사용자 인터페이스(UI) 구현 | 1.51.0 |
| **`langchain`** | LLM 애플리케이션 프레임워크 뼈대 | 1.2.13 |
| **`langchain-classic`** | LangChain v1.0 업데이트 이후 분리된 체인(Chain) 모듈 지원 | 1.0.3 |
| **`langchain-google-genai`** | Google Gemini API 연동 (LLM) | 4.2.1 |
| **`langchain-huggingface`** | HuggingFace 오픈소스 임베딩 모델 연결 | 1.2.1 |
| **`faiss-cpu`** | 메타(Meta) 오픈소스 벡터 데이터베이스 (문서 검색) | 1.13.2 |
| **`sentence-transformers`** | 텍스트를 벡터로 변환하는 임베딩 처리 엔진 | 5.3.0 |
| **`olefile`** | 한글 파일(.hwp) 바이너리 구조 분석 및 텍스트 추출 | 0.47 |
| **`torch`** | 임베딩 모델 구동을 위한 딥러닝 백엔드 (PyTorch) | 2.11.0 |

<br>

## 🚨 다른 프로젝트 적용 및 배포 시 주의사항 (Troubleshooting)

이 프로젝트를 다른 컴퓨터로 옮기거나 클라우드(Streamlit Cloud 등)에 배포할 때 다음 사항을 반드시 확인하세요.

1. **무거운 PyTorch(`torch`) 의존성 문제**
   * `sentence-transformers`를 설치하면 딥러닝 엔진인 `torch`가 자동으로 설치됩니다. 파일 용량이 매우 크고(약 2GB 이상) OS 환경(Windows/Mac/Linux)에 따라 설치 에러가 잦습니다.
   * 클라우드 배포 시 용량 초과 에러가 난다면, `requirements.txt`에 GPU 버전이 아닌 가벼운 CPU 전용 PyTorch가 설치되도록 옵션을 지정해야 할 수 있습니다.

2. **FAISS 패키지 분리 (`faiss-cpu` vs `faiss-gpu`)**
   * 현재 로컬 훈련용으로 `faiss-cpu`를 사용 중입니다. 만약 속도 향상을 위해 GPU가 있는 서버로 프로젝트를 이관한다면, 기존 `faiss-cpu`를 지우고 `faiss-gpu`를 설치해야 코드가 최적화되어 돌아갑니다.

3. **LangChain 모듈 파편화 (Version 분리)**
   * LangChain은 최근 업데이트가 매우 잦아 패키지가 여러 개로 쪼개졌습니다. 
   * 다른 프로젝트에서 `pip install langchain`만 실행하면 기존의 `create_retrieval_chain` 같은 기능이 임포트되지 않아 에러가 납니다. 반드시 `langchain-classic`과 `langchain-community`를 세트로 명시해서 설치해야 합니다.

4. **Anaconda 환경과의 충돌 방지**
   * 현재 개발 환경에는 `conda` 관련 패키지가 다수 설치되어 있습니다. 배포 시 `pip freeze > requirements.txt`를 무작정 실행하면 이 프로젝트와 무관한 수십 개의 패키지가 딸려 들어가 배포 서버가 터집니다. 반드시 위에 명시된 **핵심 패키지만 `requirements.txt`에 수동으로 기재**하여 공유하세요.





## 🛠️ Development Environment & Stack

- **Python Version:** `3.11`
- **Main Framework:** LangChain & LangGraph
- **Interface:** Streamlit

### 1. 핵심 라이브러리 (Core AI Stack)
프로젝트의 LLM 연결, 워크플로우 및 벡터 검색을 담당하는 핵심 패키지입니다.

| 분류 | 라이브러리 명 | 버전 | 설명 |
| :--- | :--- | :--- | :--- |
| **Framework** | `langchain` | `1.2.13` | 메인 프레임워크 (Core/Community 포함) |
| **Workflow** | `langgraph` | `1.1.3` | 상태 기반 멀티 에이전트(Multi-agent) 제어 |
| **Google AI** | `langchain-google-genai` | `4.2.1` | Gemini 모델 연동 및 Google Generative AI 활용 |
| **HuggingFace** | `langchain-huggingface` | `1.2.1` | 로컬 임베딩 및 오픈소스 모델 인터페이스 |
| **Vector DB** | `faiss-cpu` | `1.13.2` | CPU 기반 고성능 벡터 검색 엔진 |
| **Tracing** | `langsmith` | `0.7.22` | 체인 실행 추적 및 성능 모니터링 |

### 2. 주요 의존성 및 주의 라이브러리 (Critical Dependencies)
아래 라이브러리들은 업데이트 시 다른 패키지와의 충돌(Dependency Hell) 가능성이 높으므로 버전을 고정하여 사용합니다.

| 라이브러리 명 | 버전 | 주의사항 (Conflict Risk) |
| :--- | :--- | :--- |
| `pydantic` | `2.12.5` | **중요:** v1/v2 혼용 시 데이터 검증 에러 발생 주의 |
| `numpy` | `2.4.3` | 최신 2.x 버전으로, 하위 호환성 확인 필수 |
| `torch` | `2.11.0` | Python 3.11 및 CUDA 환경 최적화 버전 |
| `streamlit` | `1.55.0` | 웹 인터페이스 및 대시보드 배포용 |
| `aiohttp` | `3.13.3` | 비동기 호출 시 LangChain 내부 통신 안정성 관련 |

---

## 🚀 Quick Start (Installation)

현재 환경과 동일한 라이브러리를 설치하려면 아래 명령어를 사용하세요.

```bash
# 1. 가상환경 활성화 (이미 되어 있다면 생략)
conda activate p311

# 2. 필수 패키지 설치
pip install langchain==1.2.13 langgraph==1.1.3 langchain-google-genai==4.2.1 streamlit==1.55.0 pydantic==2.12.5 numpy==2.4.3