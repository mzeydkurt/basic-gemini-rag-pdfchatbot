import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import vertexai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os
import tempfile

# .env dosyasını yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
project_id = "your-gcp-project-id"  # <<< Burayı kendi projenizle değiştirin
region = "us-central1"              # <<< Burayı kendi GCP bölgenize göre değiştirin

# API anahtarı kontrolü
if not api_key:
    raise ValueError("GEMINI_API_KEY çevresel değişkeni tanımlanmamış.")

# Google API ortam değişkenini ayarla
os.environ["GOOGLE_API_KEY"] = api_key

# Vertex AI başlat
try:
    vertexai.init(project=project_id, location=region)
except Exception as e:
    st.error(f"Vertex AI başlatılamadı: {e}")
    st.stop()

# Streamlit sayfa ayarları
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("📄 PDF Destekli Sohbet Botu")
st.write("PDF belgelerinize dayalı olarak doğal dilde sorular sorabilirsiniz.")

# PDF yükleme
uploaded_file = st.file_uploader("Bir PDF dosyası yükleyin", type="pdf")

# Eğer yeni bir PDF yüklendiyse, vektörleri hazırla
if uploaded_file:
    if "last_uploaded" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded:
        with st.spinner("PDF işleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vectordb = FAISS.from_documents(docs, embedding)

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                verbose=True
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.messages = []  # mesaj geçmişi (chat ekranı için)

        st.success("PDF başarıyla işlendi!")

# Sohbet Arayüzü
if "qa_chain" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Sorunuzu yazın...")

    if user_input:
        # Kullanıcı mesajını göster
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # RAG üzerinden cevap al
        response = st.session_state.qa_chain.invoke(user_input)
        answer = response["answer"]

        # Asistan mesajını göster
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
