#!/usr/bin/env python3
"""
劳动合同分析Web应用启动器
快速启动本地网页服务
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

def check_dependencies():
    """检查依赖库"""
    print("检查依赖库...")
    
    required_packages = {
        'fastapi': 'FastAPI web框架',
        'uvicorn': 'ASGI服务器',
        'jinja2': '模板引擎',
        'python-multipart': '文件上传支持',
        'pandas': '数据处理'
    }
    
    optional_packages = {
        'reportlab': 'PDF导出功能',
        'python-docx': 'Word导出功能'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, description in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   {package} - {description}")
        except ImportError:
            missing_required.append(package)
            print(f"   {package} - {description} (未安装)")
    
    for package, description in optional_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   {package} - {description}")
        except ImportError:
            missing_optional.append(package)
            print(f"   {package} - {description} (可选，未安装)")
    
    if missing_required:
        print(f"\n缺少必需依赖: {', '.join(missing_required)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n可选依赖未安装: {', '.join(missing_optional)}")
        print("安装可选依赖以启用完整功能:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True

def main():
    """主函数"""
    print("劳动合同分析Web应用")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装必需的依赖库")
        sys.exit(1)
    
    print("\n📋 应用功能:")
    print("合同文件上传 (TXT, DOC, DOCX, PDF)")
    print("智能风险分析")
    print("条款级别评估")
    print("修改建议生成")
    print("多格式报告导出")
    print("打印功能支持")
    
    print("\n启动信息:")
    print("   访问地址: http://localhost:8000")
    print("   API文档: http://localhost:8000/docs")
    print("   快速模式: python run_webapp.py --fast")
    
    # 检查主应用文件
    webapp_file = Path("contract_analysis_webapp.py")
    if not webapp_file.exists():
        print(f"\n找不到主应用文件: {webapp_file}")
        print("请确保 contract_analysis_webapp.py 文件存在")
        sys.exit(1)
    
    print(f"\n准备就绪！启动Web服务...")
    print("按 Ctrl+C 可停止服务")
    
    # 启动应用
    try:
        os.system(f"{sys.executable} contract_analysis_webapp.py")
    except KeyboardInterrupt:
        print("\n服务已停止")

if __name__ == "__main__":
    main() 