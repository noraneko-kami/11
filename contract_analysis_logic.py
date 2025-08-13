import os
import sys
import warnings
import subprocess
import multiprocessing
from pathlib import Path
import psutil
import torch
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import re
from typing import List, Dict, Any, Tuple
import pickle
from tqdm import tqdm
import gc

# 预设置 HuggingFace 镜像端点（需在相关库导入前生效）
if os.environ.get('HF_ENDPOINT') is None:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)

warnings.filterwarnings('ignore')

def configure_hf(mirror: str = None, offline: bool = False) -> None:
    try:
        if mirror:
            os.environ.setdefault('HF_ENDPOINT', mirror)
        if offline:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
        print(f"HuggingFace 端点: {endpoint} | 离线: {os.environ.get('HF_HUB_OFFLINE', '0')}")
    except Exception as e:
        print(f"配置 HuggingFace 镜像失败: {e}")

configure_hf(mirror="https://hf-mirror.com")

CPU_COUNT = multiprocessing.cpu_count()
GPU_AVAILABLE = torch.cuda.is_available()
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3) if GPU_AVAILABLE else 0

if GPU_AVAILABLE:
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    print("GPU 不可用")

LINUX_CONFIG = {
    "use_gpu": GPU_AVAILABLE,
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "num_workers": min(CPU_COUNT, 8),
    "prefetch_factor": 2,
    "pin_memory": GPU_AVAILABLE,
    "thread_pool_size": CPU_COUNT // 2,
    "process_pool_size": min(CPU_COUNT // 2, 4),
    "model": {"model_name": "ShengbinYue/LawLLM-7B", "max_length": 1024, "batch_size": 2, "use_4bit": True, "cpu_offload": False},
    "rag_config": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "top_k": 5, "similarity_threshold": 0.7, "chunk_size": 256, "chunk_overlap": 50, "index_type": "sklearn"},
    "lora_config": {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["q_proj", "v_proj"], "bias": "none", "task_type": "CAUSAL_LM"},
    "training_config": {"learning_rate": 1e-4, "num_epochs": 3, "warmup_steps": 100, "gradient_accumulation_steps": 2, "dataloader_num_workers": min(CPU_COUNT, 4), "fp16": GPU_AVAILABLE, "gradient_checkpointing": True}
}

from sklearn.metrics.pairwise import cosine_similarity

class LegalRAGSystem:
    def __init__(self, config=LINUX_CONFIG):
        self.config = config
        self.embedding_model = None
        self.knowledge_base = []
        self.embeddings = None
        self.index = None

    def load_embedding_model(self):
        print("正在加载嵌入模型...")
        model_name = self.config['rag_config']['embedding_model']

        self.embedding_model = SentenceTransformer(
            model_name,
            device=self.config['device']
        )
        print(f"嵌入模型加载成功: {model_name}")

    def load_index(self, path="legal_rag_index"):
        print(f"加载RAG索引: {path}...")
        try:
            with open(f"{path}_knowledge.pkl", 'rb') as f:
                data = pickle.load(f)
                self.knowledge_base = data['knowledge_base']
                saved_faiss_available = data.get('faiss_available', True)
            
            self.embeddings = np.load(f"{path}_embeddings.npy")
            
            print("使用sklearn进行向量搜索")
            self.index = None
            
            self.load_embedding_model()
            print("RAG索引加载完成")
            return True
        except Exception as e:
            print(f"无法加载索引: {e}")
            return False

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        if top_k is None:
            top_k = self.config['rag_config']['top_k']
        
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, device=self.config['device']
        )
        
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        score_idx_pairs = list(zip(top_scores, top_indices))
        
        results = []
        for score, idx in score_idx_pairs:
            if idx >= len(self.knowledge_base):
                continue
            if score > self.config['rag_config']['similarity_threshold']:
                results.append({
                    'content': self.knowledge_base[idx]['content'],
                    'metadata': self.knowledge_base[idx]['metadata'],
                    'score': float(score),
                    'type': self.knowledge_base[idx]['type']
                })
        return results

class LegalLoRATrainer:
    def __init__(self, config=LINUX_CONFIG):
        self.config = config
        model_config = config.get('model')
        if not model_config:
            raise KeyError("未找到可用的模型配置，请在配置中提供 'model' 字段")
        self.model_config = model_config
        self.lora_config = config['lora_config']
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def load_model_and_tokenizer(self, model_dir="./legal_lora_model"):
        print(f"加载模型和tokenizer: {model_dir}...")
        device = self.config['device']

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                'trust_remote_code': True,
                'device_map': 'auto' if device == 'cuda' else None,
                'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
            }

            if self.model_config['use_4bit'] and device == 'cuda':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs['quantization_config'] = bnb_config
                print("4-bit量化已启用")
            
            base_model_name = self.model_config['model_name']
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
            
            self.peft_model = get_peft_model(base_model, LoraConfig.from_pretrained(model_dir))
            
            print(f"PEFT模型加载成功: {model_dir}")

        except Exception as e:
            print(f"无法加载模型: {e}")
            raise RuntimeError(f"Failed to load LoRA model: {e}")

class IntegratedLegalAnalyzer:
    def __init__(self, rag_system, lora_trainer, config=LINUX_CONFIG):
        self.rag_system = rag_system
        self.lora_trainer = lora_trainer
        self.config = config

    def analyze_clause(self, clause_text: str, detailed: bool = True) -> Dict[str, Any]:
        print(f"分析条款: {clause_text[:50]}...")
        start_time = time.time()
        rag_results = self.rag_system.search(clause_text, top_k=5)
        enhanced_context = self._build_enhanced_context(clause_text, rag_results)
        analysis = self._generate_analysis(enhanced_context)
        result = self._integrate_results(clause_text, rag_results, analysis, detailed)
        result['processing_time'] = time.time() - start_time
        print(f"分析完成. 耗时: {result['processing_time']:.2f}s")
        return result

    def _build_enhanced_context(self, clause_text: str, rag_results: List[Dict]) -> str:
        context = f"分析条款：{clause_text}\\n\\n参考：\\n"
        for i, item in enumerate(rag_results[:2], 1):
            content = item['content'][:100]
            context += f"{i}. {content}...\\n"
        context += "\\n请简要分析风险和建议："
        return context

    def _generate_analysis(self, enhanced_context: str) -> str:
        try:
            inputs = self.lora_trainer.tokenizer(
                enhanced_context,
                return_tensors="pt",
                max_length=self.lora_trainer.model_config['max_length'],
                truncation=True,
                padding=True
            ).to(self.config['device'])

            with torch.no_grad():
                outputs = self.lora_trainer.peft_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.lora_trainer.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            analysis = self.lora_trainer.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            return analysis.strip()
        except Exception as e:
            print(f"LoRA生成分析失败: {e}")
            raise RuntimeError(f"LoRA模型生成分析失败: {e}")

    def _integrate_results(self, clause_text: str, rag_results: List[Dict], analysis: str, detailed: bool) -> Dict[str, Any]:
        return {
            'input_clause': clause_text,
            'risk_level': self._assess_risk_level(clause_text, analysis),
            'analysis': analysis,
            'rag_sources': len(rag_results) if detailed else [],
        }

    def _assess_risk_level(self, clause_text: str, analysis: str) -> str:
        text_lower = (clause_text + analysis).lower()
        if any(keyword in text_lower for keyword in ['高风险', '严重', '无效', '违法']):
            return "高风险"
        if any(keyword in text_lower for keyword in ['中等风险', '建议修改', '不明确', '风险']):
            return "中等风险"
        return "低风险"

class OptimizedContractAnalyzer:
    def __init__(self, legal_analyzer):
        self.legal_analyzer = legal_analyzer
        self.rag_system = legal_analyzer.rag_system
        self.lora_trainer = legal_analyzer.lora_trainer
        self.config = legal_analyzer.config
        self.performance_config = {
            'max_input_length': 600,
            'max_tokenizer_length': 400,
            'max_new_tokens': 128, 
            'rag_top_k': 3,
            'context_truncate_length': 100,
            'clause_min_length': 20,
            'paragraph_min_length': 30,
            'max_parallel_workers': 4
        }
        print("高性能合同分析器初始化完成")

    def analyze_contract(self, contract_text: str, auto_extract: bool = True, performance_mode: str = 'balanced') -> Dict[str, Any]:
        print(f"分析合同 (模式: {performance_mode})...")
        print(f"合同长度: {len(contract_text)} 字符")
        start_time = time.time()
        
        try:
            clauses = self._extract_contract_clauses(contract_text, auto_extract, performance_mode)
            print(f"发现 {len(clauses)} 个条款.")
            
            if not clauses:
                return self._create_fallback_result("No clauses found.", start_time, "Clause extraction failed.")
            clause_results = self._batch_analyze_optimized(clauses, performance_mode)
            contract_analysis = self._generate_contract_summary_fast(clauses, clause_results)
            risk_assessment = self._assess_contract_risks_enhanced(clause_results)
            modification_suggestions = self._generate_smart_suggestions(clause_results)
            processing_time = time.time() - start_time
            result = self._build_comprehensive_result(
                contract_text, clauses, clause_results,
                contract_analysis, risk_assessment,
                modification_suggestions, processing_time,
                performance_mode
            )
            
            print(f"合同分析完成，耗时 {processing_time:.2f}s.")
            print(f"分析 {len(clauses)} 个条款，发现 {len(modification_suggestions)} 个改进点")
            
            return result
            
        except Exception as e:
            print(f"合同分析失败: {e}")
            return self._create_fallback_result(contract_text, start_time, str(e))

    def _extract_contract_clauses(self, contract_text: str, auto_extract: bool = True, performance_mode: str = 'balanced') -> List[str]:
        
        if not auto_extract:
            return [contract_text]
        
        clause_patterns = [
            r'第[一二三四五六七八九十\d]+条[：:]?',
            r'[一二三四五六七八九十\d]+[、．\.]',
            r'\d+[\.、]',
            r'[（(][一二三四五六七八九十\d]+[）)]',
            r'第[一二三四五六七八九十\d]+章',
            r'[一二三四五六七八九十\d]+条',
        ]
        
        
        clauses = []
        current_clause = ""
        lines = contract_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_new_clause = any(re.match(pattern, line) for pattern in clause_patterns)
            
            if is_new_clause and current_clause:
                clauses.append(current_clause.strip())
                current_clause = line
            else:
                if current_clause:
                    current_clause += '\n' + line
                else:
                    current_clause = line
        
        if current_clause:
            clauses.append(current_clause.strip())
        
        min_length = self.performance_config['clause_min_length']
            
        valid_clauses = [clause for clause in clauses if len(clause) >= min_length]
        
        if not valid_clauses or len(valid_clauses) <= 1:
            print("使用智能分割模式...")
            valid_clauses = self._smart_split_contract(contract_text)
        
        return valid_clauses

    def _smart_split_contract(self, contract_text: str) -> List[str]:
        
        keyword_patterns = [
            r'(工作时间|工作内容|劳动报酬|工资|薪酬|试用期|合同期限|违约|解除|终止|保密|竞业|社会保险|福利|休假|加班|奖惩|培训|争议|纠纷|其他|附则)',
            r'([一二三四五六七八九十]\s*[、．\.])',
            r'(\d+\s*[、．\.])',
            r'([（(]\s*[一二三四五六七八九十\d]+\s*[）)])',
        ]
        
        clauses = []
        
        for pattern in keyword_patterns:
            parts = re.split(pattern, contract_text, flags=re.IGNORECASE)
            if len(parts) > 3:
                current_clause = ""
                for i, part in enumerate(parts):
                    if re.match(pattern, part, re.IGNORECASE):
                        if current_clause.strip():
                            clauses.append(current_clause.strip())
                        current_clause = part
                    else:
                        current_clause += part
                
                if current_clause.strip():
                    clauses.append(current_clause.strip())
                break
        
        if not clauses or len(clauses) <= 1:
            sentences = contract_text.split('。')
            current_clause = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                current_clause += sentence + '。'
                
                if len(current_clause) >= 100:
                    clauses.append(current_clause.strip())
                    current_clause = ""
            
            if current_clause.strip():
                clauses.append(current_clause.strip())
        
        if not clauses or len(clauses) <= 1:
            clauses = self._split_by_paragraphs_optimized(contract_text)
        
        min_length = 30 
        valid_clauses = [clause for clause in clauses if len(clause) >= min_length]
        
        return valid_clauses

    def _force_split_long_text(self, contract_text: str) -> List[str]:
        
        chunk_size = 300
        clauses = []
        
        words = contract_text.split()
        current_clause = ""
        
        for word in words:
            if len(current_clause + word) > chunk_size and current_clause:
                clauses.append(current_clause.strip())
                current_clause = word
            else:
                current_clause += " " + word if current_clause else word
        
        if current_clause.strip():
            clauses.append(current_clause.strip())
        
        return clauses if len(clauses) > 1 else [contract_text]

    def _split_by_paragraphs_optimized(self, contract_text: str) -> List[str]:
        
        paragraphs = contract_text.split('\n\n')
        min_length = self.performance_config['paragraph_min_length']
        valid_paragraphs = [
            para.strip() for para in paragraphs 
            if para.strip() and len(para.strip()) >= min_length
        ]
        
        if len(valid_paragraphs) < 3:
            lines = contract_text.split('\n')
            current_para = ""
            valid_paragraphs = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_para = current_para + '\n' + line if current_para else line
                    if len(current_para) > 200:
                        valid_paragraphs.append(current_para)
                        current_para = ""
            
            if current_para:
                valid_paragraphs.append(current_para)
        
        return valid_paragraphs

    def _batch_analyze_optimized(self, clauses: List[str], performance_mode: str) -> List[Dict]:
        results = []
        for i, clause in enumerate(clauses):
            try:
                result = self._analyze_single_clause_optimized(clause)
                result['clause_index'] = i
                results.append(result)
            except Exception as e:
                print(f"条款 {i} 分析失败: {e}")
                results.append(self._create_fallback_clause_result(clause, i))
        return results
    
    def _analyze_single_clause_optimized(self, clause_text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        rag_results = self.rag_system.search(clause_text, top_k=self.performance_config['rag_top_k'])
        
        enhanced_context = self._build_optimized_context(clause_text, rag_results)
        
        analysis = self._generate_analysis_optimized(enhanced_context)
        
        result = self._integrate_results_fast(clause_text, rag_results, analysis)
        result['processing_time'] = time.time() - start_time
        
        return result

    def _build_optimized_context(self, clause_text: str, rag_results: List[Dict]) -> str:
        context = f"请分析以下劳动合同条款的合规风险，并给出修改建议：\n条款原文：'{clause_text}'\n\n"
        if rag_results:
            context += "参考信息：\n"
            for item in rag_results:
                context += f"- {item['content']}\n"
        return context

    def _generate_analysis_optimized(self, enhanced_context: str) -> str:
        """LoRA模型专业分析生成 - 纯AI驱动，无备用机制"""
        
        if not hasattr(self.lora_trainer, 'peft_model') or self.lora_trainer.peft_model is None:
            raise RuntimeError("LoRA模型未加载，无法进行专业分析")

        structured_prompt = f"""
请对以下劳动合同条款进行专业的法律风险分析，按以下格式输出：

{enhanced_context}

请按以下结构进行分析：
条款缺陷分析：[详细说明条款存在的法律风险和问题]
相关案例：[提供相关的法律案例和判例]
判决结果：[说明法院对类似案例的判决倾向]
法律依据：[引用相关的法律条文]
修改建议：[提供具体的修改建议]
风险等级：[高风险/中等风险/低风险]
"""

        inputs = self.lora_trainer.tokenizer(
            structured_prompt,
            return_tensors="pt", 
            truncation=True, 
            max_length=self.performance_config['max_tokenizer_length']
        )
        
        device = self.config['device']
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.lora_trainer.peft_model.generate(
                **inputs, 
                max_new_tokens=self.performance_config['max_new_tokens'],
                min_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.lora_trainer.tokenizer.pad_token_id or self.lora_trainer.tokenizer.eos_token_id,
                eos_token_id=self.lora_trainer.tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        analysis = self.lora_trainer.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        if len(analysis) < 50:
            raise RuntimeError(f"LoRA模型生成内容过短（{len(analysis)}字符），分析质量不符合要求")
        
        if not any(keyword in analysis for keyword in ['分析', '建议', '风险', '法律']):
            raise RuntimeError("LoRA模型生成内容不包含法律分析关键信息，分析失败")
        
        print(f"LoRA模型成功生成专业分析，长度: {len(analysis)} 字符")
        return analysis

    def _integrate_results_fast(self, clause_text: str, rag_results: List[Dict], analysis: str) -> Dict[str, Any]:
        risk_level = self._assess_risk_level_fast(clause_text + analysis)
        parsed_info = self._parse_analysis_fast(analysis)
        result = {
            'input_clause': clause_text,
            'analysis': analysis,
            'rag_sources': len(rag_results),
            'risk_level': risk_level
        }
        result.update(parsed_info)
        return result

    def _assess_risk_level_fast(self, text: str) -> str:
        """基于AI分析结果的智能风险评级"""
        text_lower = text.lower()
        
        if '风险等级：高风险' in text or '风险等级：高' in text:
            return "高风险"
        elif '风险等级：中等风险' in text or '风险等级：中' in text:
            return "中等风险"
        elif '风险等级：低风险' in text or '风险等级：低' in text:
            return "低风险"
        
        high_risk_indicators = [
            '严重违法', '无效条款', '显失公平', '违反法律', '败诉', '仲裁败诉',
            '超出法定', '不合法', '违约金过高', '试用期过长', '随时解除',
            '不予补偿', '单方面决定', '承担责任', '赔偿损失'
        ]
        
        medium_risk_indicators = [
            '建议修改', '需要完善', '表述不清', '可能存在风险', '注意合规',
            '建议调整', '需要明确', '应当规范', '可能争议', '建议优化'
        ]
        
        low_risk_indicators = [
            '符合法律', '表述规范', '基本合规', '建议保持', '内容合理',
            '无重大问题', '基本符合', '较为规范'
        ]
        
        high_count = sum(1 for indicator in high_risk_indicators if indicator in text_lower)
        medium_count = sum(1 for indicator in medium_risk_indicators if indicator in text_lower)
        low_count = sum(1 for indicator in low_risk_indicators if indicator in text_lower)
        
        if high_count >= 2 or any(phrase in text_lower for phrase in ['严重违法', '无效条款', '败诉']):
            return "高风险"
        elif high_count >= 1 or medium_count >= 2:
            return "中等风险"
        elif medium_count >= 1 or (low_count == 0 and high_count == 0):
            return "中等风险"
        else:
            return "低风险"

    def _parse_analysis_fast(self, analysis: str) -> Dict[str, str]:
        result = {
            'defect_analysis': '',
            'related_cases': '',
            'judgment_result': '',
            'legal_basis': '',
            'modification_suggestion': ''
        }
        
        lines = analysis.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if '条款缺陷分析' in line or '缺陷分析' in line:
                current_section = 'defect_analysis'
                content = line.split('：', 1)[-1] if '：' in line else ''
                if content:
                    result[current_section] = content
            elif '相关案例' in line or '案例' in line:
                current_section = 'related_cases'
                content = line.split('：', 1)[-1] if '：' in line else ''
                if content:
                    result[current_section] = content
            elif '判决结果' in line or '判决' in line:
                current_section = 'judgment_result'
                content = line.split('：', 1)[-1] if '：' in line else ''
                if content:
                    result[current_section] = content
            elif '法律依据' in line or '法条' in line:
                current_section = 'legal_basis'
                content = line.split('：', 1)[-1] if '：' in line else ''
                if content:
                    result[current_section] = content
            elif '修改建议' in line or '建议' in line:
                current_section = 'modification_suggestion'
                content = line.split('：', 1)[-1] if '：' in line else ''
                if content:
                    result[current_section] = content
            elif line and current_section and not line.startswith(('条款', '相关', '判决', '法律', '修改')):
                if result[current_section]:
                    result[current_section] += '\n' + line
                else:
                    result[current_section] = line
        
        if not any(result.values()):
            result['defect_analysis'] = analysis
            result['modification_suggestion'] = '请根据分析结果进行相应修改。'
        
        return result

    def _generate_smart_suggestions(self, clause_results: List[Dict]) -> List[Dict]:
        suggestions = []
        
        for i, result in enumerate(clause_results):
            risk_level = result.get('risk_level', '低风险')
            original_clause = result.get('input_clause', '')
            analysis = result.get('analysis', '')
            
            ai_suggestion = self._generate_ai_modification_suggestion(original_clause, analysis, risk_level)
            
            suggestion = {
                'clause_number': i + 1,
                'risk_level': risk_level,
                'original_clause': original_clause,
                'main_issues': result.get('defect_analysis', ''),
                'suggested_action': ai_suggestion.get('modification_action', result.get('modification_suggestion', '')),
                'related_cases': result.get('related_cases', ''),
                'judgment_result': result.get('judgment_result', ''),
                'legal_basis': result.get('legal_basis', ''),
                'priority': 1 if risk_level == '高风险' else (2 if risk_level == '中等风险' else 3),
                'urgency': '紧急' if risk_level == '高风险' else ('重要' if risk_level == '中等风险' else '一般'),
                'ai_confidence': ai_suggestion.get('confidence', 'high'),
                'modification_type': ai_suggestion.get('modification_type', 'optimization'),
                'implementation_steps': ai_suggestion.get('implementation_steps', [])
            }
            
            suggestions.append(suggestion)
        
        suggestions = self._ai_prioritize_suggestions(suggestions)
        return suggestions

    def _generate_ai_modification_suggestion(self, clause_text: str, analysis: str, risk_level: str) -> Dict[str, Any]:
        
        if not hasattr(self.lora_trainer, 'peft_model') or self.lora_trainer.peft_model is None:
            raise RuntimeError("LoRA模型未加载，无法生成AI修改建议")
        
        suggestion_prompt = f"""
基于以下合同条款分析结果，请生成具体的修改建议：

原始条款：{clause_text}

分析结果：{analysis}

风险等级：{risk_level}

请按以下格式提供修改建议：
修改类型：[删除/修改/补充/重写]
具体建议：[详细的修改建议和理由]
修改后条款：[提供修改后的具体条款文本]
实施步骤：[1. 步骤一 2. 步骤二 3. 步骤三]
预期效果：[修改后预期达到的效果]
置信度：[高/中/低]
"""

        try:
            inputs = self.lora_trainer.tokenizer(
                suggestion_prompt,
                return_tensors="pt", 
                truncation=True, 
                max_length=self.performance_config['max_tokenizer_length']
            )
            
            # 移动到正确的设备
            device = self.config['device']
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.lora_trainer.peft_model.generate(
                    **inputs, 
                    max_new_tokens=self.performance_config['max_new_tokens'],
                    min_new_tokens=80,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.85,
                    repetition_penalty=1.2,
                    pad_token_id=self.lora_trainer.tokenizer.pad_token_id or self.lora_trainer.tokenizer.eos_token_id,
                    eos_token_id=self.lora_trainer.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            suggestion_text = self.lora_trainer.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            parsed_suggestion = self._parse_ai_suggestion(suggestion_text)
            
            print(f"AI成功生成修改建议，长度: {len(suggestion_text)} 字符")
            return parsed_suggestion
            
        except Exception as e:
            print(f"AI建议生成失败: {e}")
            raise RuntimeError(f"大模型修改建议生成失败: {e}")

    def _parse_ai_suggestion(self, suggestion_text: str) -> Dict[str, Any]:
        
        result = {
            'modification_type': 'optimization',
            'modification_action': '',
            'revised_clause': '',
            'implementation_steps': [],
            'expected_effect': '',
            'confidence': 'medium'
        }
        
        lines = suggestion_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if '修改类型：' in line or '修改类型:' in line:
                mod_type = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                result['modification_type'] = mod_type.strip()
            elif '具体建议：' in line or '具体建议:' in line:
                current_section = 'modification_action'
                content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                if content.strip():
                    result[current_section] = content.strip()
            elif '修改后条款：' in line or '修改后条款:' in line:
                current_section = 'revised_clause'
                content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                if content.strip():
                    result[current_section] = content.strip()
            elif '实施步骤：' in line or '实施步骤:' in line:
                current_section = 'implementation_steps'
                content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                if content.strip():
                    result[current_section] = [content.strip()]
            elif '预期效果：' in line or '预期效果:' in line:
                current_section = 'expected_effect'
                content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                if content.strip():
                    result[current_section] = content.strip()
            elif '置信度：' in line or '置信度:' in line:
                confidence = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                result['confidence'] = confidence.strip().lower()
            elif line and current_section:
                if current_section == 'implementation_steps':
                    if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                        result[current_section].append(line)
                    elif result[current_section]:
                        result[current_section][-1] += '\n' + line
                else:
                    if result[current_section]:
                        result[current_section] += '\n' + line
                    else:
                        result[current_section] = line
        
        if not result['modification_action']:
            result['modification_action'] = suggestion_text[:200] + ('...' if len(suggestion_text) > 200 else '')
        
        return result

    def _ai_prioritize_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        
        if not hasattr(self.lora_trainer, 'peft_model') or self.lora_trainer.peft_model is None:
            return sorted(suggestions, key=lambda x: (x['priority'], x['clause_number']))
        
        try:
            suggestion_summary = []
            for i, sug in enumerate(suggestions):
                summary = f"条款{sug['clause_number']}: {sug['risk_level']}, 问题: {sug['main_issues'][:50]}..."
                suggestion_summary.append(summary)
            
            priority_prompt = f"""
请对以下合同条款修改建议进行优先级排序，考虑法律风险、紧急程度、实施难度等因素：

{chr(10).join(suggestion_summary)}

请按优先级从高到低排序，输出格式：
优先级排序：条款X, 条款Y, 条款Z...
排序理由：[简要说明排序依据]
"""

            inputs = self.lora_trainer.tokenizer(
                priority_prompt,
                return_tensors="pt", 
                truncation=True, 
                max_length=400
            )
            
            device = self.config['device']
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.lora_trainer.peft_model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8
                )
            
            input_length = inputs['input_ids'].shape[1]
            priority_result = self.lora_trainer.tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            ).strip()
            
            ai_order = self._parse_ai_priority_order(priority_result, len(suggestions))
            
            if ai_order:
                reordered_suggestions = []
                for clause_num in ai_order:
                    for sug in suggestions:
                        if sug['clause_number'] == clause_num:
                            reordered_suggestions.append(sug)
                            break
                
                for sug in suggestions:
                    if sug not in reordered_suggestions:
                        reordered_suggestions.append(sug)
                
                print(f"AI智能排序完成，优先级: {ai_order}")
                return reordered_suggestions
            
        except Exception as e:
            print(f"AI优先级排序失败: {e}")
        
        return sorted(suggestions, key=lambda x: (x['priority'], x['clause_number']))

    def _parse_ai_priority_order(self, priority_text: str, total_suggestions: int) -> List[int]:
        
        import re
        
        for line in priority_text.split('\n'):
            if '优先级排序' in line or '排序' in line:
                numbers = re.findall(r'条款(\d+)', line)
                if numbers:
                    try:
                        return [int(num) for num in numbers if 1 <= int(num) <= total_suggestions]
                    except ValueError:
                        continue
        
        return []

    def _generate_contract_summary_fast(self, clauses: List[str], clause_results: List[Dict]) -> str:
        risk_counts = pd.Series([r['risk_level'] for r in clause_results]).value_counts().to_dict()
        
        high_risk = risk_counts.get('高风险', 0)
        medium_risk = risk_counts.get('中等风险', 0)
        low_risk = risk_counts.get('低风险', 0)
        
        main_issues = []
        for i, result in enumerate(clause_results, 1):
            if result.get('risk_level') in ['高风险', '中等风险']:
                defect = result.get('defect_analysis', '').split('\n')[0][:80]
                if defect:
                    main_issues.append(f"条款{i}: {defect}")
        
        summary = f"""
【AI智能分析摘要】
✅ 条款总数: {len(clauses)} 个
🔴 高风险条款: {high_risk} 个
🟡 中等风险条款: {medium_risk} 个  
🟢 低风险条款: {low_risk} 个

【主要风险点】
{chr(10).join(main_issues[:5]) if main_issues else '✅ 未发现重大风险点'}

【AI分析建议】
基于大模型专业分析，建议重点关注高风险条款，及时修订以降低法律风险。
        """.strip()
        
        return summary

    def _assess_contract_risks_enhanced(self, clause_results: List[Dict]) -> Dict[str, Any]:
        
        total_clauses = len(clause_results)
        high_risk = len([r for r in clause_results if r.get('risk_level') == '高风险'])
        medium_risk = len([r for r in clause_results if r.get('risk_level') == '中等风险'])
        low_risk = len([r for r in clause_results if r.get('risk_level') == '低风险'])
        
        risk_score = (high_risk * 25 + medium_risk * 10 + low_risk * 2) / total_clauses if total_clauses > 0 else 0
        
        if risk_score >= 20:
            overall_risk, risk_color = "高风险", "#d32f2f"
        elif risk_score >= 10:
            overall_risk, risk_color = "中等风险", "#f57c00"
        else:
            overall_risk, risk_color = "低风险", "#388e3c"
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': round(risk_score, 2),
            'risk_color': risk_color,
            'risk_distribution': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk,
                'total': total_clauses
            },
            'risk_percentage': {
                'high': round(high_risk / total_clauses * 100, 1) if total_clauses > 0 else 0,
                'medium': round(medium_risk / total_clauses * 100, 1) if total_clauses > 0 else 0,
                'low': round(low_risk / total_clauses * 100, 1) if total_clauses > 0 else 0
            },
            'risk_intensity': 'severe' if risk_score >= 20 else 'moderate' if risk_score >= 10 else 'mild',
            'ai_confidence': 'high'
        }

    def _build_comprehensive_result(self, contract_text, clauses, clause_results, summary, risk, suggestions, time, mode):
        return {
            'contract_text': contract_text,
            'clause_count': len(clauses),
            'contract_summary': summary,
            'risk_assessment': risk,
            'modification_suggestions': suggestions,
            'processing_time': time,
            'extracted_clauses': clauses,
            'clause_results': clause_results,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_mode': mode
        }
        
    def _create_fallback_result(self, contract_text, start_time, error_msg):
        return {
            'error': error_msg,
            'contract_text': contract_text,
            'clause_count': 0,
            'processing_time': time.time() - start_time,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed'
        }
        
    def _create_fallback_clause_result(self, clause_text, index):
        return {
            'input_clause': clause_text,
            'risk_level': '分析失败',
            'analysis': f'条款分析失败，错误索引: {index}',
            'rag_sources': 0,
            'clause_index': index,
            'error': 'analysis_failed'
        }

rag_system = LegalRAGSystem()
lora_trainer = LegalLoRATrainer()
integrated_analyzer = IntegratedLegalAnalyzer(rag_system, lora_trainer)
analyzer = OptimizedContractAnalyzer(integrated_analyzer)




def check_model_cache_status() -> Dict[str, Any]:
    try:
        status = {
            'mkl_threading_layer': os.environ.get('MKL_THREADING_LAYER', ''),
            'vector_backend': 'sklearn',
            'gpu_available': GPU_AVAILABLE,
            'gpu_memory_gb': round(GPU_MEMORY_GB, 2) if GPU_AVAILABLE else 0,
            'embedding_model': LINUX_CONFIG['rag_config'].get('embedding_model'),
            'model_dir_exists': os.path.isdir('./legal_lora_model'),
            'rag_index': {
                'knowledge': os.path.exists('legal_rag_index_knowledge.pkl'),
                'embeddings': os.path.exists('legal_rag_index_embeddings.npy'),
                # 'faiss_index' removed; sklearn backend does not use an index
            }
        }
        return status
    except Exception as e:
        return {'error': str(e)}


def initialize_models():
    if not rag_system.load_index():
        print("RAG系统初始化失败")
    lora_trainer.load_model_and_tokenizer() 