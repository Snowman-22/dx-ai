"""
서버 실행 진입점.
`python main.py` 또는 `uvicorn app:app --reload` 로 실행.
"""
import os

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
