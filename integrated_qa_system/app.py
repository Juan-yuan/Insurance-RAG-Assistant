import asyncio
import json
import os
import re
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from main_history_stream_version import IntegratedQASystem

STATIC_DIR = os.path.join(_HERE, "static")

app = FastAPI(
    title="Insurance RAG Assistant API",
    description="HTTP and WebSocket Q&A for health-insurance claims review, backed by MySQL FAQ search and RAG over clauses, medical records, and similar cases.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)

qa_system = None

GREETING_PATTERNS = [
    {
        "pattern": r"^(hi|hello|hey)\b",
        "response": "Hello. I am the Insurance RAG Assistant. I can help review coverage against policy clauses, medical records, and similar cases.",
    },
    {
        "pattern": r"^(who are you|what('s| is) your name|your name)\b",
        "response": "I am the Insurance RAG Assistant, a claims-review helper for policy clauses, medical records, and similar cases.",
    },
    {
        "pattern": r"^(are you (there|here)|anyone there)\b",
        "response": "I am here. Ask about deductibles, waiting periods, exclusions, or whether a bill is payable under the policy.",
    },
    {
        "pattern": r"^(what are you doing|what('re| are) you up to)\b",
        "response": "Standing by to review coverage questions against policy clauses, medical records, and similar cases. How can I help?",
    },
]


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


def get_qa_system():
    global qa_system
    if qa_system is None:
        try:
            qa_system = IntegratedQASystem()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Q&A backend is unavailable: {e}")
    return qa_system


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    system = get_qa_system()
    try:
        history = system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load conversation history: {str(e)}")


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    system = get_qa_system()
    success = system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "Conversation history cleared"}
    raise HTTPException(status_code=500, detail="Failed to clear conversation history")


def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


def _normalize_source_filter(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }
    system = get_qa_system()
    answer, need_rag = system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        return {
            "answer": "Please use the WebSocket endpoint for a streamed response",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
    }


def _websocket_connected(websocket: WebSocket) -> bool:
    return websocket.client_state == WebSocketState.CONNECTED


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data.get("query")
            source_filter = _normalize_source_filter(request_data.get("source_filter"))
            session_id = request_data.get("session_id") or str(uuid.uuid4())
            start_time = time.time()
            if not query or not str(query).strip():
                if _websocket_connected(websocket):
                    await websocket.send_json({"type": "error", "error": "Missing query"})
                continue
            if _websocket_connected(websocket):
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id,
                })
            greeting_response = check_greeting(str(query))
            if greeting_response:
                if _websocket_connected(websocket):
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id,
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time,
                    })
                continue
            system = get_qa_system()
            collected_answer = ""
            for token, is_complete in system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token
                if is_complete and not collected_answer:
                    if _websocket_connected(websocket):
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                if token and _websocket_connected(websocket):
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id,
                    })
                if is_complete:
                    if _websocket_connected(websocket):
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                await asyncio.sleep(0.01)
    except HTTPException as e:
        if _websocket_connected(websocket):
            await websocket.send_json({
                "type": "error",
                "error": e.detail,
            })
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if _websocket_connected(websocket):
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
    finally:
        try:
            if _websocket_connected(websocket):
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/sources")
async def get_sources():
    return {"sources": get_qa_system().config.VALID_SOURCES}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host=host, port=port, reload=False)
