"""Run the API server:  python -m t2_assistant"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("t2_assistant.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
