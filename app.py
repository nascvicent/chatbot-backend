import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_chain import build_chain

load_dotenv()

app = FastAPI(title="Classificador Jurídico RAG")

# 🔒 Permitir apenas o domínio do seu front
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://classificadorgjur.vercel.app",
        "http://localhost:5173"  # opcional para testar local
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    document: str

# ⚙️ Lazy loading — só cria o chain quando for necessário
chain = None

def get_chain():
    global chain
    if chain is None:
        print("🔄 Carregando modelo RAG pela primeira vez...")
        chain = build_chain()
        print("✅ Modelo carregado com sucesso!")
    return chain

@app.post("/classify")
async def classify(req: ClassifyRequest):
    try:
        rag_chain = get_chain()
        result = rag_chain.invoke(req.document)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
