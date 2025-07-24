import socketio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([api_key, endpoint, deployment_name, api_version]):
    raise ValueError("Eksik ortam değişkeni!")

# Embedding
embedding = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version=api_version
)

# Vectorstore
try:
    vector_store = FAISS.load_local("Backend/vector_store", embedding, allow_dangerous_deserialization=True)
except Exception as e:
    raise ValueError(f"Vector store yüklenirken hata: {e}")

# Question generation prompt - orijinal kodunuzdaki gibi
question_gen_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Aşağıda bir sohbet geçmişi ("chat_history") ve ardından bir takip sorusu ("question") var.
Bu bilgileri kullanarak özgün, açıklayıcı ve açık uçlu bir soru oluştur:

Sohbet geçmişi:
{chat_history}

Takip sorusu:
{question}

Özgün soru:
"""
)

qa_system_prompt ="""
Kurum içi bilgilere erişimi olan ve RAG (Retrieval-Augmented Generation) destekli bir asistan olarak, yalnızca sunulan iç kaynaklara ve belgelere dayanarak bilgi sağlayabilirsin. Aşağıdaki yönergeler senin iletişim tarzını ve yanıtlarını şekillendirmek içindir:

1. Konu, elimizdeki belgeler dışında ise, kibarca yardımcı olamayacağını belirt. Örneğin: 
    “Üzgünüm, bu konuda size yardımcı olamıyorum çünkü elimdeki bilgiler bu alanı kapsamıyor.”

2. Yanıtların her zaman:
    - Açık, net ve anlaşılır,
    - Resmi ancak samimi bir dilde,
    - Kullanıcıyı ön planda tutan bir yaklaşımla hazırlanmalı.

3. Sohbet başlatılırsa, anlayışlı ve ilgili bir diyalog kurmaya çalış.

4. Her yanıtının sonunda, kullanıcıyı iletişime devam etmeye teşvik etmek için: 
    “Başka bir sorunuz var mı?” 
    cümlesini mutlaka ekle.

Elindeki bilgilere sadık kalarak, en iyi desteği vermeye çalış.

Bilgiler:
{context}

Soru:
{question}

Yanıtın açık, net ve kullanıcıyı destekleyici bir dille hazırlanmalı.
"""

# QA prompt template
qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)

# Memory session bazlı tutulacak
session_memories = {}
session_temperatures = {} # Yeni: Her session için sıcaklık değeri tutulacak

# Varsayılan sıcaklık
DEFAULT_TEMPERATURE = 0.2

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
    return session_memories[session_id]

def get_temperature(session_id: str) -> float:
    """Belirli bir session için sıcaklık değerini döndürür, yoksa varsayılanı kullanır."""
    if session_id not in session_temperatures:
        session_temperatures[session_id] = DEFAULT_TEMPERATURE
    return session_temperatures[session_id]

def set_temperature(session_id: str, temperature: float):
    """Belirli bir session için sıcaklık değerini ayarlar."""
    if 0.0 <= temperature <= 1.0: # Sıcaklık aralığı kontrolü
        session_temperatures[session_id] = temperature
        return True
    return False

def format_chat_history(messages):
    """Chat history'yi string formatına çevir"""
    if not messages:
        return ""
    
    formatted = []
    # Messages liste halinde gelirse
    if isinstance(messages, list):
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i] if isinstance(messages[i], str) else str(messages[i])
                ai_msg = messages[i + 1] if isinstance(messages[i + 1], str) else str(messages[i + 1])
                formatted.append(f"İnsan: {human_msg}")
                formatted.append(f"Asistan: {ai_msg}")
    else:
        # String formatında gelirse
        formatted.append(str(messages))
    
    return "\n".join(formatted)

def get_conversational_rag_chain(llm, retriever):
    """Conversational RAG zinciri oluşturur"""
    def contextualize_question(inputs):
        """Soruyu chat history ile bağlamlaştır"""
        chat_history = inputs.get("chat_history", "")
        question = inputs.get("input", "")

        # Eğer chat_history list ise, stringe çevir
        if isinstance(chat_history, list):
            chat_history = format_chat_history(chat_history)
        if not chat_history or not chat_history.strip():
            # Eğer chat history yoksa, orijinal soruyu döndür
            return question
        # Question generator chain'i çalıştır
        question_generator = question_gen_prompt | llm | StrOutputParser()
        formatted_history = chat_history
        contextualized_question = question_generator.invoke({
            "chat_history": formatted_history,
            "question": question
        })
        return contextualized_question.strip()
    # Document combination chain
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    # RAG chain with contextualized question
    rag_chain = (
        RunnablePassthrough.assign(
            question=RunnableLambda(contextualize_question)
        ) |
        create_retrieval_chain(retriever, Youtube_chain)
    )
    return rag_chain

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*'   # Tüm origin'leri kabul et
)

### FastAPI uygulaması ve endpoint'ler ###

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],   # Geliştirme için gerekli, frontend'in çalıştığı port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# 'ask' olayı artık Socket.IO üzerinden gelecek
@sio.event
async def ask(sid, data):
    session_id = data.get("session_id")
    question = data.get("question")
    # Temperature'ı client'tan al, yoksa session'ın varsayılanını kullan
    temperature = data.get("temperature", get_temperature(session_id)) 

    if not session_id or not question:
        await sio.emit("error", {"error": "session_id ve question gereklidir"}, to=sid)
        return

    try:
        memory = get_memory(session_id)
        
        # LLM'i dinamik olarak temperature ile başlat
        llm = AzureChatOpenAI(
            azure_deployment=deployment_name,
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            openai_api_version=api_version,
            temperature=temperature # Burası güncellendi
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        rag_chain = get_conversational_rag_chain(llm, retriever)

        chat_history = memory.buffer if isinstance(memory.buffer, str) else format_chat_history(memory.buffer)

        response = await rag_chain.ainvoke({
            "input": question,
            "chat_history": chat_history
        })

        answer = response.get("answer") if isinstance(response, dict) else str(response)

        memory.save_context({"input": question}, {"output": answer})

        await sio.emit("answer", {"answer": answer}, to=sid) # Cevabı 'answer' olayı ile gönder
    except Exception as e:
        await sio.emit("error", {"error": str(e)}, to=sid)

# Yeni Socket.IO olayı: sohbeti temizle
@sio.event
async def clear_chat(sid, data):
    session_id = data.get("session_id")
    if session_id in session_memories:
        del session_memories[session_id]
        # Session'ın sıcaklık bilgisini de temizleyebiliriz veya varsayılana çekebiliriz
        if session_id in session_temperatures:
            session_temperatures[session_id] = DEFAULT_TEMPERATURE # Varsayılana sıfırla
        print(f"Session {session_id} sohbet geçmişi temizlendi.")
        await sio.emit("chat_cleared_ack", {"message": "Sohbet geçmişiniz temizlendi."}, to=sid)
    else:
        await sio.emit("error", {"error": "Session bulunamadı veya zaten temizlenmiş."}, to=sid)

# Yeni Socket.IO olayı: sıcaklık ayarla
@sio.event
async def set_temperature(sid, data):
    session_id = data.get("session_id")
    new_temperature = data.get("temperature")

    if not session_id or new_temperature is None:
        await sio.emit("error", {"error": "session_id ve temperature gereklidir."}, to=sid)
        return

    try:
        new_temperature = float(new_temperature)
        if set_temperature(session_id, new_temperature):
            print(f"Session {session_id} için sıcaklık {new_temperature} olarak ayarlandı.")
            await sio.emit("temperature_updated_ack", {"temperature": new_temperature, "message": "Sıcaklık başarıyla güncellendi."}, to=sid)
        else:
            await sio.emit("error", {"error": "Geçersiz sıcaklık değeri. 0.0 ile 1.0 arasında olmalı."}, to=sid)
    except ValueError:
        await sio.emit("error", {"error": "Sıcaklık sayısal bir değer olmalı."}, to=sid)
    except Exception as e:
        await sio.emit("error", {"error": str(e)}, to=sid)

# FastAPI endpoint'leri (istemci artık bunları doğrudan kullanmayacak ama backend'de kalabilir)
@app.get("/")
async def root():
    return {"message": "RAG API çalışıyor"}
    
# Not: Aşağıdaki FastAPI endpoint'leri frontend tarafından doğrudan kullanılmayacak
# ancak eğer başka bir uygulama bu API'yi RESTful olarak kullanmak isterse kalabilir.
# Sohbet temizleme ve sıcaklık ayarı artık Socket.IO üzerinden yönetiliyor.

@app.delete("/session/{session_id}")
async def clear_session_rest(session_id: str):
    """Belirli bir session'ın memory'sini temizler (RESTful - isteğe bağlı)."""
    if session_id in session_memories:
        del session_memories[session_id]
        if session_id in session_temperatures:
            session_temperatures[session_id] = DEFAULT_TEMPERATURE
        return {"message": f"Session {session_id} temizlendi"}
    return {"message": "Session bulunamadı"}

@app.get("/session/{session_id}/history")
async def get_session_history_rest(session_id: str):
    """Session'ın chat history'sini döndürür (RESTful - isteğe bağlı)."""
    if session_id in session_memories:
        memory = session_memories[session_id]
        return {"history": memory.buffer if hasattr(memory, 'buffer') else ""}
    return {"message": "Session bulunamadı"}
