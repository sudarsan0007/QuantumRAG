"""Start the QuantumRAG dashboard:  python run.py  ->  http://127.0.0.1:8000"""
import argparse
import logging

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"QuantumRAG dashboard: http://{args.host}:{args.port}")
    uvicorn.run("api.server:app", host=args.host, port=args.port, log_level="info")
