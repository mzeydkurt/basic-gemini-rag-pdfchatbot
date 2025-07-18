"""
Problem Tanımı : PDF Chatbot Destek Sistemi
    - Müşterilen sık sık sorduğu benzer sorulara hızlı cevap veren bir chatbot geliştirmek.
    - Chatbot, geçmiş müşteri etkileşimlerinden öğrenerek sürekli gelişen bir yapıya sahip olacak.

    çözüm : 
    - pdf dosyasını vektör veri tabanına dönüştür
    - kullanıcıdan gelen soruları vektörleştirip veri tabanında sorgula ve LLM ile cevap üret.

Kullanılan Teknolojiler :
    - Langchain : RAG mimarisi için kullanılıcak
    - faiss : vektör veri tabanı için kullanılıcak
    - openai veya gemini : soru cevap LLM için kullanılıcak
    - streamlit : kullanıcı arayüzü için kullanılıcak

Veri seti: (Ulusal staj programı S.S.S. Soru ve Cevapları)
    Soru ve Cevap şeklinde bir verisetimiz pdf dosyamız olacak.
    
Plan/Program:
    -SSS bilgilerini içeren bir PDF
    -Kullanıcı dosyayı arayüzden yüklenecek
    -PDF dosyası parçalara ayrılıcak ve vektör veri tabanına dönüştürülecek
    -Kullanıcı soru sorduğu zaman cektör db den benzer içerikler getirilir, llm ile cevap üretilir.

import libraries : freeze

"""

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import ChatGoogleGenerativeAI # Doğrudan Gemini için bu tercih edilir

from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# API anahtarı kontrolü
if not api_key:
    raise ValueError("GEMINI_API_KEY çevresel değişkeni tanımlanmamış.")

# Google API ortam değişkenini ayarla
os.environ["GOOGLE_API_KEY"] = api_key

# Embedding modelini başlatıyoruz (text to vector dönüşümü için)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Vektör veri tabanını yüklüyoruz
vectordb = FAISS.load_local(
    "sss_vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

# Konuşma geçmişi için memory oluşturma
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Dil Modelimiz 0 rastlantısallıkla çalışır ve sabit cevaplar verir.
# init_chat_model yerine ChatGoogleGenerativeAI kullanın.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)

# RAG + Memory Zincir Oluşturuyoruz
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose = True
)

print("Chatbot başarıyla başlatıldı. SSS veritabanı yüklendi ve hazır.") # Başarılı mesajı yazdırıyoruz

while True:
    user_input = input("Siz : ")
    if user_input.lower() in ["exit", "quit", "q","çık"]:
        break

    # Kullanıcının sorusu LLM + RAG + Memory ile cevaplanıyor
    response = qa_chain.run(user_input)
    print("Chatbot : ", response) # Chatbot cevabını yazdırıyoruz
