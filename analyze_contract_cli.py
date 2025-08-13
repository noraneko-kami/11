
import os
import sys
import json
import argparse
import time
from typing import Any, Dict

try:
    import docx 
except Exception:
    docx = None

try:
    import requests
except Exception:
    requests = None

from contract_analysis_logic import analyzer, initialize_models


def _resolve_runtime_base_path() -> str:
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path and os.path.isdir(base_path):
        return base_path
    return os.path.dirname(os.path.abspath(__file__))


def _chdir_to_base(base_path: str) -> None:
    """Change working directory to base path so relative resource paths work."""
    try:
        os.chdir(base_path)
    except Exception:
        pass


def _read_file_content(input_path: str) -> str:
    """Read contract content from .txt or .docx."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    lower = input_path.lower()
    if lower.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif lower.endswith(".docx"):
        if docx is None:
            raise RuntimeError("python-docx 未安装，无法读取 .docx 文件")
        try:
            document = docx.Document(input_path)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            raise RuntimeError(f"读取 .docx 失败: {e}")
    else:
        raise ValueError("仅支持 .txt 与 .docx 文件")


def _write_json(data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_html(report: Dict[str, Any], output_path: str) -> None:
    """Write a simple self-contained HTML report (no template dependency)."""
    summary = report.get("contract_summary", "")
    risk = report.get("risk_assessment", {})
    clause_count = report.get("clause_count", 0)
    processing_time = report.get("processing_time", 0)

    high = risk.get("risk_distribution", {}).get("high_risk", 0)
    medium = risk.get("risk_distribution", {}).get("medium_risk", 0)
    low = risk.get("risk_distribution", {}).get("low_risk", 0)
    overall = risk.get("overall_risk_level", "未知")

    html = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>劳动合同风险分析报告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif; margin: 24px; color: #1f2937; }}
  h1 {{ font-size: 20px; margin-bottom: 8px; }}
  .meta {{ color: #6b7280; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
  .high {{ background: #fee2e2; color: #b91c1c; }}
  .medium {{ background: #ffedd5; color: #c2410c; }}
  .low {{ background: #dcfce7; color: #166534; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; border-radius: 6px; }}
</style>
</head>
<body>
  <h1>劳动合同风险分析报告</h1>
  <div class=\"meta\">生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} · 条款数：{clause_count} · 耗时：{processing_time:.2f}s</div>
  <div class=\"grid\">
    <div class=\"card\"><div>总体风险等级</div><div style=\"margin-top:6px\"><span class=\"badge {'high' if overall=='高风险' else 'medium' if overall=='中等风险' else 'low'}\">{overall}</span></div></div>
    <div class=\"card\"><div>高风险条款</div><div style=\"margin-top:6px\">{high}</div></div>
    <div class=\"card\"><div>中等风险条款</div><div style=\"margin-top:6px\">{medium}</div></div>
    <div class=\"card\"><div>低风险条款</div><div style=\"margin-top:6px\">{low}</div></div>
  </div>
  <h2 style=\"margin-top:16px; font-size:16px\">AI分析摘要</h2>
  <pre>{summary}</pre>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _call_api(api_url: str, text: str, mode: str, auto_extract: bool) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests 不可用，无法使用 API 模式")
    url = api_url.rstrip('/') + "/api/analyze_text"
    resp = requests.post(url, json={
        "text": text,
        "performance_mode": mode,
        "auto_extract": auto_extract
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="劳动合同风险分析（CLI）——基于 LoRA + RAG 的本地分析器"
    )
    parser.add_argument("input", help="输入合同文件路径（.txt 或 .docx）")
    parser.add_argument("--mode", choices=["fast", "balanced", "detailed"], default="balanced", help="性能模式")
    parser.add_argument("--no-auto-extract", action="store_true", help="禁用自动条款提取，整文分析")
    parser.add_argument("--json-out", default=None, help="JSON 报告输出路径（默认：<输入名>.report.json）")
    parser.add_argument("--html-out", default=None, help="可选：HTML 报告输出路径")
    parser.add_argument("--api-url", default=None, help="可选：分析服务地址，例如 http://127.0.0.1:8000，用于避免本地重复加载模型")
    args = parser.parse_args()

    base_path = _resolve_runtime_base_path()
    _chdir_to_base(base_path)

    input_path = os.path.abspath(args.input)
    print(f"读取文件: {input_path}")
    content = _read_file_content(input_path)

    api_url = args.api_url or os.environ.get("ANALYZER_API_URL")
    report: Dict[str, Any]

    if api_url:
        try:
            print(f"使用服务进行分析：{api_url}")
            t_api = time.time()
            report = _call_api(api_url, content, args.mode, not args.no_auto_extract)
            print(f"服务分析完成，用时 {time.time() - t_api:.2f}s")
        except Exception as e:
            print(f"服务分析失败，改用本地模式：{e}")
            api_url = None
    
    if not api_url:
        print("初始化模型与索引...")
        t0 = time.time()
        initialize_models()
        print(f"初始化完成，用时 {time.time() - t0:.2f}s")

        print(f"开始分析（mode={args.mode}）...")
        t1 = time.time()
        report = analyzer.analyze_contract(
            content,
            auto_extract=(not args.no_auto_extract),
            performance_mode=args.mode
        )
        print(f"分析完成，用时 {time.time() - t1:.2f}s")

    json_out = args.json_out or (os.path.splitext(input_path)[0] + ".report.json")
    _write_json(report, json_out)
    print(f"已保存 JSON 报告: {json_out}")

    if args.html_out:
        _write_html(report, args.html_out)
        print(f"已保存 HTML 报告: {args.html_out}")


if __name__ == "__main__":
    main() 