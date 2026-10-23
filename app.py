"""Web arayüzü.

Kullanım: .venv\\Scripts\\python app.py   (ya da baslat.bat'a çift tıkla)
Tarayıcıda http://127.0.0.1:8000 açılır.
"""
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

import engine
import fetch_data

URL = "http://127.0.0.1:8000"
app = FastAPI(title="BTC Pattern")
engines: dict[str, engine.Engine] = {}
lock = threading.Lock()   # numba'nın paralel katmanı aynı anda iki aramayı kaldırmaz


def json_response(data) -> Response:
    """Büyük yanıtlar (50 sonuç x yüzlerce mum) FastAPI'nin varsayılan kodlayıcısından çok daha hızlı."""
    return Response(json.dumps(data, separators=(",", ":")), media_type="application/json")


def get_engine(interval: str) -> engine.Engine:
    """Zaman dilimi motoru; ilk istendiğinde yüklenir (verisi yoksa önce indirilir)."""
    if interval not in engine.INTERVALS:
        raise HTTPException(400, f"Zaman dilimi şunlardan biri olmalı: {', '.join(engine.INTERVALS)}")
    if interval not in engines:
        if not (fetch_data.DATA_DIR / f"{fetch_data.SYMBOL.lower()}_{interval}.csv").exists():
            fetch_data.fetch(interval)
        engines[interval] = engine.Engine(interval)
    return engines[interval]


@app.get("/api/status")
def status(interval: str = "1h"):
    with lock:
        return get_engine(interval).status()


@app.post("/api/update")
def update(interval: str = "1h"):
    with lock:
        eng = get_engine(interval)
        try:
            fetch_data.fetch(interval)
        except Exception as e:
            raise HTTPException(502, f"Binance'ten veri alınamadı: {e}")
        eng.reload()
        return eng.status()


@app.get("/api/search")
def search(interval: str = "1h", method: str = "dtw", now: int | None = None,
           start: int | None = None, end: int | None = None):
    """now: 'şu an' (unix saniye, boşsa son mum). start/end: kullanıcının seçtiği aralık."""
    if method not in ("dtw", "euclid"):
        raise HTTPException(400, "method 'dtw' ya da 'euclid' olmalı")
    if (start is None) != (end is None):
        raise HTTPException(400, "Seçim için start ve end birlikte verilmeli")
    with lock:
        try:
            return json_response(get_engine(interval).find(
                now, method == "dtw", (start, end) if start is not None else None))
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.get("/api/detail")
def detail(start: int, length: int, interval: str = "1h", now: int | None = None):
    """Bir eşleşmenin karşılaştırma penceresi için ayrıntılı mumları (sonrası uzun)."""
    with lock:
        try:
            return json_response({"bars": get_engine(interval).detail(start, length, now)})
        except ValueError as e:
            raise HTTPException(400, str(e))


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")

if __name__ == "__main__":
    get_engine("1h")      # varsayılan zaman dilimi açılışta hazır olsun
    threading.Timer(1.0, lambda: webbrowser.open(URL)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
