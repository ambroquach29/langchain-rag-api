from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from backend import generate_response

app = FastAPI()

# Optional: CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate-response")
async def generate_response_api(
    file: UploadFile,
    google_api_key: str = Form(...),
    query_text: str = Form(...)
):
    file_content = (await file.read()).decode()
    response = generate_response(file_content, google_api_key, query_text)
    return JSONResponse(content={"result": response})
