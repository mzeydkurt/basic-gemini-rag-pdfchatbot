from langchain_community.vectorstores import FAISS #faiss kütüphanesini kullanarak vektörleri depolayacağız
from langchain.text_splitter import RecursiveCharacterTextSplitter #metinleri parçalara ayırmak için kullanacağız
from langchain_community.embeddings import HuggingFaceEmbeddings #HuggingFace gömme modelini kullanacağız
from langchain_community.document_loaders import PyPDFLoader #PDF dosyalarını yüklemek için kullanacağız

from dotenv import load_dotenv #çevresel değişkenleri yüklemek için kullanacağız
import os #dosya ve dizin işlemleri için kullanacağız

load_dotenv() #çevresel değişkenleri yüklüyoruz
api_key = os.getenv("GEMINI_API_KEY") #GEMINI_API_KEY çevresel değişkenini alıyoruz

if not api_key: #eğer api_key tanımlanmamışsa
    raise ValueError("GEMINI_API_KEY çevresel değişkeni tanımlanmamış.") #hata fırlatıyoruz

# Google API anahtarını ayarlıyoruz
os.environ["GOOGLE_API_KEY"] = api_key #GOOGLE_API_KEY çevresel değişkenini api_key ile ayarlıyoruz


#PDF dosyasını yükleme
loader = PyPDFLoader("KariyerKapisiSıkcaSorulanSorularogrenci_sss.pdf") #PyPDFLoader kullanarak PDF dosyasını yüklüyoruz
documents = loader.load() # langchain documents objesi PDF dosyasını yüklüyoruz

#metinleri 500 karakterlik parçalara ayırıyoruz, 50 karakterlik bir örtüşme ile
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap = 50
)

#metinleri parçalara ayırıyoruz
docs = text_splitter.split_documents(documents)

# HuggingFace gömme modelini kullanarak metinleri gömülüyoruz
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") #HuggingFace gömme modelini kullanıyoruz

vectordb = FAISS.from_documents(docs, embedding) #metinleri gömülüyoruz ve FAISS vektör deposu oluşturuyoruz

# vektör veri tabanını locale kaydediyoruz
vectordb.save_local("sss_vectorstore") #FAISS vektör deposunu yerel diske kaydediyoruz

print("Embedding ve Vektör veri tabanı başarıyla oluşturuldu ve kaydedildi.") #başarılı mesajı yazdırıyoruz