import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import build_chain

# Carrega .env se existir
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Classificador Jurídico RAG")

# Permitir CORS do frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou coloque a URL do frontend para segurança
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    document: str

# Inicializa o chain (carrega embeddings, FAISS, LLM)
try:
    chain = build_chain()
except Exception as e:
    print("Erro ao construir chain:", e)
    chain = None

@app.post("/classify")
async def classify(req: ClassifyRequest):
    if chain is None:
        raise HTTPException(status_code=500, detail="Chain não inicializado.")
    doc = req.document
    try:
        # invoke diretamente com o texto do usuário
        result = chain.invoke(doc)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
