# 劳动合同风险分析系统（FastAPI + LoRA + RAG）

一个可交付的本地/服务化劳动合同分析工具。后端基于 FastAPI，分析侧采用 LoRA 微调大模型 + RAG 检索增强。支持：
- 网页上传 .txt/.docx 合同，生成分析报告（Jinja2 模板展示）
- 命令行分析（本地/服务模式），输出 JSON 和可选 HTML 报告

## 目录结构
- `contract_analysis_webapp.py`: FastAPI 服务（启动即加载模型，一次加载，多次复用）
- `contract_analysis_logic.py`: 模型与分析核心逻辑（幂等初始化、RAG + LoRA 推理）
- `analyze_contract_cli.py`: 命令行分析工具（可通过 API 服务避免本地重复加载）
- `templates/`, `static/`: 前端模板与静态资源
- `legal_lora_model/`: LoRA 适配器目录（必须存在）
- `legal_rag_index_*.{npy,index,pkl}`: RAG 索引三件套（必须存在）
- `run_webapp.py`: 本地开发时的服务启动器
- `webapp_requirements.txt`: 运行依赖
- `train.ipynb`: 用于LoRA微调和RAG检索的训练文件

## 环境准备
建议使用 Conda 新环境（示例环境名：`new`）
```bash
conda create -n new python=3.12 -y
conda activate new
pip install -U -r webapp_requirements.txt
```

可选：若使用命令行工具的“服务模式”，需要安装 `requests`：
```bash
pip install requests
```

## Web 服务方式（推荐避免重复加载）
启动服务（首启将加载模型与索引，后续请求直接复用）：
```bash
python run_webapp.py
# 浏览器访问 http://127.0.0.1:8000

## 注意事项
- docker拉取镜像 -docker pull yorunokami/contract-analyzer:code
- 本项目RAG索引中embedding模型为sentence-transformers/all-MiniLM-L6-v2
- LoRA微调的基座模型为ShengbinYue/LawLLM-7B
- 本项目随仓库包含 LoRA 适配器与 RAG 索引，但不包含基础 7B 模型权重。生产运行建议提供本地基础模型路径，或允许在线拉取。
- GPU 环境下默认启用 4-bit 量化（可在 `contract_analysis_logic.py` 配置）。
- 首次加载模型耗时较长，建议以 Web 服务常驻进程方式使用，以避免重复加载。 