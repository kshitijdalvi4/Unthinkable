import sys
import os
import asyncio

# Ensure backend directory is in path for relative and absolute imports
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

if sys.platform == 'win32':
    # Explicitly set ProactorEventLoopPolicy for Windows to support subprocesses in asyncio
    # This MUST be done before any event loop is created or uvicorn starts.
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("[Launcher] Windows ProactorEventLoopPolicy set.")

import uvicorn

if __name__ == "__main__":
    print("[Launcher] Starting uvicorn...")
    # We import 'app' here to ensure the logic above runs first
    from main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
