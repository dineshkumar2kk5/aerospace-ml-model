import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from semantic_explainer import SemanticExplainer

app = FastAPI(
    title="Aviation Meteorology Semantic ML API",
    description="Dense Neural Vector Search + Cross-Encoder Reranker for IC Joshi Aviation Meteorology",
    version="1.0.0"
)

# Global model instance
explainer: Optional[SemanticExplainer] = None

@app.on_event("startup")
def load_model():
    global explainer
    print("[*] Starting Aviation Semantic ML Engine...")
    explainer = SemanticExplainer()
    print("[+] Model loaded and ready to serve requests!")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Aviation Meteorology Semantic ML",
        "endpoints": ["/api/explain", "/api/search", "/docs"]
    }

@app.post("/api/explain")
def explain_endpoint(request: QueryRequest):
    if not explainer:
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    result = explainer.explain(request.query, rerank_top_k=request.top_k or 3)
    return result

@app.get("/api/search")
def search_endpoint(q: str = Query(..., description="Natural language search query")):
    if not explainer:
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    hits = explainer.embedder.search_semantic(q, top_k=5)
    return {"query": q, "results": hits}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
