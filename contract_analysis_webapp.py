from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import docx
from typing import Dict, Any
from contract_analysis_logic import analyzer, initialize_models

app = FastAPI(title="劳动合同风险分析系统")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

analysis_results_store: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    print("服务启动中...")
    
    initialize_models()
    print("模型初始化完成")

def read_file_content(file: UploadFile) -> str:
    filename = file.filename
    if filename.endswith(".docx"):
        try:
            doc = docx.Document(file.file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading docx file: {e}")
            return ""
    elif filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    else:
        return ""

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.svg")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_and_analyze(request: Request, contract_file: UploadFile = File(...)):
    if not contract_file:
        return JSONResponse(status_code=400, content={"message": "No file uploaded."})

    print(f"收到文件: {contract_file.filename}")
    contract_text = read_file_content(contract_file)

    if not contract_text:
        return JSONResponse(status_code=400, content={"message": "Could not read file content or unsupported file type."})

    file_id = contract_file.filename

    print(f"分析合同: {file_id}")
    analysis_result = analyzer.analyze_contract(contract_text, auto_extract=True, performance_mode='balanced')
    
    analysis_results_store[file_id] = analysis_result

    return templates.TemplateResponse("report.html", {
        "request": request,
        "file_id": file_id,
        "filename": contract_file.filename,
        "report": analysis_result
    })

@app.get("/api/report/{file_id}", response_class=JSONResponse)
async def get_report_data(file_id: str):
    report = analysis_results_store.get(file_id)
    if not report:
        return JSONResponse(status_code=404, content={"message": "Report not found."})
    return JSONResponse(content=report)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 