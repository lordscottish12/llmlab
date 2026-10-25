import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("LLMLAB_HOST", "127.0.0.1")
    port = int(os.environ.get("LLMLAB_PORT", "8000"))
    print(f"[llmlab] open http://{'127.0.0.1' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run("server.app:app", host=host, port=port)
