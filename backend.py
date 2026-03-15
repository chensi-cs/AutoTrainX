'''
Author: chensi-cs 
Date: 2026-01-15 16:12:33
LastEditors: chensi-cs 
LastEditTime: 2026-02-13 00:30:18
FilePath: \流程网页\backend.py
Description: 
'''
# backend.py
# ============================================
# 后端：FastAPI + SiliconFlow + AutoDL
# - 本地：上传数据 -> 解析/修复 -> 调 LLM 生成 train.py（只生成不运行）
# - 远程 AutoDL：上传 train.py + requirements + data.csv，创建 .venv 并装依赖
# - 远程训练：运行 train.py，若报错则自动把（报错+代码+数据摘要）回传 LLM 修复，
#           覆盖远端 train.py，并可选自动重跑一次
# ============================================

from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import time
import tempfile
import zipfile
import uuid
from pathlib import Path
from typing import Dict, Optional, List,Union,Tuple

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import paramiko


# ================== 依赖缺失提示（用于报错信息友好） ==================

def detect_missing_dependency(error: Exception) -> Optional[str]:
    """
    检查 ModuleNotFoundError 并返回安装提示（备用）。
    """
    missing_packages = {
        "xgboost": "pip install xgboost",
        "catboost": "pip install catboost",
        "lightgbm": "pip install lightgbm",
        "torch": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "tensorflow": "pip install tensorflow",
        "matplotlib": "pip install matplotlib",
        "seaborn": "pip install seaborn",
        "sklearn": "pip install scikit-learn",
        "scipy": "pip install scipy",
        "rarfile": "pip install rarfile",
    }

    if isinstance(error, ModuleNotFoundError):
        pkg = str(error).split("'")[1]
        cmd = missing_packages.get(pkg)
        if cmd:
            return f"你的环境缺少依赖：{pkg}\n请执行： {cmd}"
        else:
            return f"缺少依赖包：{pkg}，请自行安装。"
    return None


# ================== 环境变量 & AutoDL 配置 ==================
print("🚀 正在启动后端服务...")
print(f"📁 当前工作目录: {os.getcwd()}")

# 新增：全局连接追踪（用于退出时统一关闭）
GLOBAL_CONNECTIONS = {
    "ssh": [],
    "sftp": [],
    "transport": []
}

SILICON_API_KEY = os.getenv("SILICON_API_KEY")
if not SILICON_API_KEY:
    print("⚠️ 警告：未检测到 SILICON_API_KEY 环境变量")
else:
    print(f"✅ SILICON_API_KEY 已配置")

# SiliconFlow 模型名称（可用环境变量覆盖）
MODEL_NAME = os.getenv("SILICON_MODEL", "deepseek-ai/DeepSeek-V3.2")
print(f"🤖 使用的模型: {MODEL_NAME}")


# 生成代码的 max_tokens：默认 8192（可自行调整）
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "8192"))

# 生成被截断时，最多续写几轮
LLM_CONTINUE_MAX_ROUNDS = int(os.getenv("LLM_CONTINUE_MAX_ROUNDS", "3"))

# 训练报错后是否自动修复 + 是否重跑一次
AUTO_FIX_ON_ERROR = os.getenv("AUTO_FIX_ON_ERROR", "1") == "1"
AUTO_FIX_RERUN = os.getenv("AUTO_FIX_RERUN", "1") == "1"

# 结束标记：用于判断是否输出完整
LLM_END_MARKER = os.getenv("LLM_END_MARKER", "<<END_OF_TRAIN_PY>>")

# AutoDL 远程服务器配置
REMOTE_HOST = os.getenv("REMOTE_HOST")                    # 例如 connect.westb.seetacloud.com
REMOTE_USER = os.getenv("REMOTE_USER") or "root"          # 一般是 root
REMOTE_PASSWORD = os.getenv("REMOTE_PASSWORD")            # SSH 密码
REMOTE_PORT = int(os.getenv("REMOTE_PORT", "22"))         # 例如 41942
REMOTE_BASE_DIR = os.getenv("REMOTE_BASE_DIR", "/root/autodl_tmp")
REMOTE_RECOMMEND_DIR = os.getenv("REMOTE_RECOMMEND_DIR", "/root/autodl_recommend")
REMOTE_RAG_DIR = os.getenv("REMOTE_RECOMMEND_DIR", "/root/autodl_recommend/rag")

print(f"🌐 远程服务器配置 - 主机: {REMOTE_HOST}, 端口: {REMOTE_PORT}, 用户: {REMOTE_USER}")
print(f"📁 远程基础目录: {REMOTE_BASE_DIR}")
print(f"📁 远程推荐目录: {REMOTE_RECOMMEND_DIR}")
print(f"📁 远程RAG推荐目录: {REMOTE_RAG_DIR}")

app = FastAPI(title="LLM Code Gen Backend (SiliconFlow + AutoDL)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# dataset_id -> 本地 CSV 路径
DATASETS: Dict[str, str] = {}


# ================== RAR 支持 ==================

try:
    import rarfile
    HAS_RAR = True
except Exception:
    HAS_RAR = False


# ================== 数据集解析：csv/data/arff/zip/rar ==================

def prepare_dataset_from_path(src_path: Path) -> str:
    """
    根据文件后缀（csv / data / arff / zip / rar），返回真正的 csv 文件路径。
    abalone.data -> 自动转成 9 列的 csv
    """
    suffix = src_path.suffix.lower()

    def _parse_text_file_to_df(text_path: Path, expected_cols: Optional[int] = None) -> pd.DataFrame:
        import re
        rows = []
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"[,\s]+", line)
                parts = [p for p in parts if p != ""]
                rows.append(parts)

        if not rows:
            raise ValueError(f"文件为空或无法解析：{text_path}")

        if expected_cols is not None:
            fixed_rows = []
            for r in rows:
                if len(r) == expected_cols:
                    fixed_rows.append(r)
                elif len(r) > expected_cols:
                    fixed_rows.append(r[:expected_cols])
                else:
                    fixed_rows.append(r + [None] * (expected_cols - len(r)))
            rows = fixed_rows

        max_cols = max(len(r) for r in rows)
        df = pd.DataFrame(rows, columns=[f"col_{i}" for i in range(max_cols)])
        return df

    # 1) 直接 CSV
    if suffix == ".csv":
        return str(src_path)

    # 2) .data（UCI，如 abalone.data）
    if suffix == ".data":
        tmp_dir = Path(tempfile.mkdtemp(prefix="dataset_data_"))
        csv_path = tmp_dir / (src_path.stem + ".csv")

        df = _parse_text_file_to_df(src_path, expected_cols=9)
        df.columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight",
            "Rings",
        ]
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    # 3) ARFF
    if suffix == ".arff":
        try:
            from scipy.io import arff
        except ImportError:
            raise RuntimeError("需要安装 scipy 来解析 ARFF： pip install scipy")

        data, meta = arff.loadarff(src_path)
        df = pd.DataFrame(data)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)

        tmp_dir = Path(tempfile.mkdtemp(prefix="dataset_arff_"))
        csv_path = tmp_dir / (src_path.stem + ".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    # 4) ZIP
    if suffix == ".zip":
        with zipfile.ZipFile(src_path, "r") as zf:
            names = zf.namelist()
            for ext in [".csv", ".data", ".arff"]:
                match = [n for n in names if n.lower().endswith(ext)]
                if match:
                    target = match[0]
                    tmp_dir = Path(tempfile.mkdtemp(prefix="dataset_zip_"))
                    zf.extract(target, path=tmp_dir)
                    return prepare_dataset_from_path(tmp_dir / target)
        raise ValueError("ZIP 内未找到 csv/data/arff 文件")

    # 5) RAR
    if suffix == ".rar":
        if not HAS_RAR:
            raise RuntimeError("需要安装 rarfile 库来解析 RAR： pip install rarfile")
        rf = rarfile.RarFile(src_path)
        names = rf.namelist()
        for ext in [".csv", ".data", ".arff"]:
            match = [n for n in names if n.lower().endswith(ext)]
            if match:
                target = match[0]
                tmp_dir = Path(tempfile.mkdtemp(prefix="dataset_rar_"))
                rf.extract(target, path=tmp_dir)
                return prepare_dataset_from_path(tmp_dir / target)
        raise ValueError("RAR 内未找到 csv/data/arff 文件")

    raise ValueError(f"暂不支持文件类型：{suffix}")


# ================== 智能修复 Abalone 等异常 CSV ==================

def auto_fix_dataset(csv_path: str) -> str:
    """
    兜底修复：
    - 尝试自动识别分隔符 sep=None
    - 若只读出一列，则用正则再次拆分
    - 避免“所有数据塞进一列”
    """
    import re

    try:
        df = pd.read_csv(csv_path, engine="python", sep=None)
    except Exception:
        df = pd.read_csv(csv_path)

    if df.shape[1] > 1:
        df.to_csv(csv_path, index=False)
        print(f"✔ auto_fix_dataset: 检测到多列数据 ({df.shape[1]} 列)，保持不变。")
        return csv_path

    one_col = df.iloc[:, 0].astype(str)
    rows = []
    for line in one_col:
        parts = re.split(r"[,\s]+", line.strip())
        parts = [p for p in parts if p != ""]
        rows.append(parts)

    max_cols = max(len(r) for r in rows)
    df_fixed = pd.DataFrame(rows, columns=[f"col_{i}" for i in range(max_cols)])

    if df_fixed.shape[1] == 9:
        df_fixed.columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight",
            "Rings",
        ]

    df_fixed.to_csv(csv_path, index=False)
    print(f"✔ auto_fix_dataset: 原始只有 1 列，已拆分为 {df_fixed.shape[1]} 列。")
    return csv_path


# ================== 生成 requirements.txt ==================

BASE_DEPENDENCIES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "xgboost",
    "tqdm",
]

def create_requirements_file_for_dataset(dataset_path: str) -> str:
    project_dir = Path(dataset_path).parent
    env_path = project_dir / "requirements.txt"
    # 强制覆盖为固定版本，不保留旧文件
    with open(env_path, "w", encoding="utf-8") as f:
        for pkg in BASE_DEPENDENCIES:
            f.write(pkg + "\n")
    print(f"✔ 已生成固定版本的环境文件: {env_path}")
    return str(env_path)


# ================== LLM 调用：完整输出 + 自动续写 ==================

def _siliconflow_chat(messages: List[dict], temperature: float = 0.2, max_tokens: Optional[int] = None, top_p: float = 0.95) -> str:
    """
    单次调用 SiliconFlow chat/completions，返回 assistant content。
    """
    if not SILICON_API_KEY:
        raise RuntimeError("未配置 SILICON_API_KEY 环境变量。")

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICON_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens or MODEL_MAX_TOKENS),
        "top_p": float(top_p),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"SiliconFlow 返回不是合法 JSON：{resp.text[:500]}")

    if resp.status_code != 200:
        raise RuntimeError(f"调用 SiliconFlow 失败：status={resp.status_code}, body={data}")

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"SiliconFlow 返回内容异常：{data}")

    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise RuntimeError("SiliconFlow 未生成任何内容。")
    return content


def _strip_code_fences(text: str) -> str:
    """
    防止模型偶尔输出 ```python ... ```。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _generate_with_continuation(base_messages: List[dict], end_marker: str = LLM_END_MARKER, max_rounds: int = LLM_CONTINUE_MAX_ROUNDS) -> str:
    """
    要求模型最终输出 end_marker；若未出现，则自动续写最多 max_rounds 轮。
    返回：去掉 end_marker 后的完整代码字符串。
    """
    messages = list(base_messages)
    full = ""

    for _ in range(max_rounds):
        chunk = _siliconflow_chat(messages, temperature=0.2, max_tokens=MODEL_MAX_TOKENS, top_p=0.95)
        chunk = _strip_code_fences(chunk)
        full += chunk + "\n"

        messages.append({"role": "assistant", "content": chunk})

        if end_marker in full:
            break

        messages.append({
            "role": "user",
            "content": (
                "继续输出剩余内容。要求：\n"
                "1) 必须从上一次停止处继续；\n"
                "2) 严禁重复已输出内容；\n"
                f"3) 最终必须以 {end_marker} 结束。\n"
            )
        })

    if end_marker not in full:
        print("⚠️ 警告：LLM 输出可能仍被截断（未检测到结束标记）。")
    code = full.split(end_marker)[0].rstrip() + "\n"
    return code

# ================== 生成 train.py（完整输出） ==================
def call_llm_to_generate_code(algorithm_name: str, dataset_path: str) -> str:
    if not algorithm_name:
        algorithm_name = "XGBRegressor"
    if not SILICON_API_KEY:
        raise RuntimeError("未配置 SILICON_API_KEY 环境变量。")

    prompt = f"""
        你是一名资深 Python 数据科学工程师，现在需要为一个监督学习任务生成完整的训练脚本 train.py。

        【数据集说明】
        - 数据集 CSV 路径：DATASET_PATH ="data.csv"  # 必须使用这个固定相对路径
        - 默认约定：最后一列为标签 y，其余列为特征 X
        - 可能存在 "Sex" 列，取值 "M"、"F"、"I"

        【输出要求（非常重要）】
        - 你必须输出“完整可运行”的 train.py 单文件源码
        - 严禁输出 Markdown 代码块标记（不要 ```）
        - 你必须在源码最后一行单独输出结束标记：{LLM_END_MARKER}
        - 除代码与必要日志 print 外，不要输出解释性自然语言

        【关键约束】
        - 顶部写全 import
        - 必须有 main()，并且：
            if __name__ == "__main__":
                result = main()
        main() return result，并且全局存在 result 变量
        - 若存在 "Sex"：{{"M":0,"F":1,"I":2}}
        - 除标签列外特征列：pd.to_numeric(errors="coerce")，清理 NaN
        - EDA corr 只对数值列 X.select_dtypes(include="number")
        - XGBoost tree_method 必须通过 detect_xgboost_tree_method() 自动检测 gpu_hist/hist
        - 分类：accuracy + classification_report + metrics.json
        - 回归：mse/rmse/mae/r2 + metrics.json
        - 不允许 input()，不写死本地绝对路径，只使用 DATASET_PATH 或相对路径
        - 日志简洁，不要输出大段解释文本

        【算法优先】
        尽量使用算法名：{algorithm_name}
        识别不了则：
        - 分类：XGBClassifier
        - 回归：XGBRegressor

        请直接输出 train.py 完整源码，并n以 {LLM_END_MARKER} 结束。
        """.strip()

    base_messages = [
        {"role": "system", "content": "你是专业的 Python 机器学习工程师。"},
        {"role": "user", "content": prompt},
    ]

    return _generate_with_continuation(base_messages, end_marker=LLM_END_MARKER, max_rounds=LLM_CONTINUE_MAX_ROUNDS)

# ================== 大模型根据数据推荐算法 ==================
def call_llm_to_recommend_algorithm(dataset_path: str) -> str:
    """
    调用LLM分析数据集，推荐最合适的算法
    返回算法名称字符串
    """
    try:
        # 读取数据集进行分析
        df = pd.read_csv(dataset_path, nrows=200)  # 读取前200行
        n_samples, n_features = df.shape
        target_col = df.columns[-1]
        
        # 分析目标列类型
        target_dtype = str(df[target_col].dtype)
        unique_values = df[target_col].nunique()
        
        # 判断问题类型
        if target_dtype in ['object', 'str', 'bool']:
            problem_type = "classification"
            default_algo = "XGBClassifier"
        elif unique_values <= 10:  # 少于10个唯一值，可能是分类
            problem_type = "classification"
            default_algo = "XGBClassifier"
        else:
            problem_type = "regression"
            default_algo = "XGBRegressor"
        
        # 构建数据集摘要
        dataset_summary = f"""
            数据集分析报告：
            - 数据规模: {n_samples} 个样本, {n_features} 个特征
            - 目标列: '{target_col}' (数据类型: {target_dtype})
            - 问题类型: {problem_type} (唯一值数量: {unique_values})
            - 特征类型: 混合类型
            - 数据大小: {os.path.getsize(dataset_path) / 1024:.1f} KB
            """
                    
        prompt = f"""
            你是一名资深的数据科学家，请分析以下数据集并推荐最适合的机器学习算法。

            【数据集分析报告】
            {dataset_summary}

            【推荐要求】
            1. 基于数据集规模、特征类型、问题类型推荐
            2. 优先考虑准确性和训练效率
            3. 如果是小数据集，推荐较简单的算法
            4. 如果是大数据集，推荐可扩展的算法

            【输出格式】
            只输出一个算法名称，例如：
            RandomForestClassifier
            或
            XGBRegressor

            请推荐最适合的算法：
            """
        
        # 调用LLM
        response = _siliconflow_chat(
            messages=[
                {"role": "system", "content": "你是专业的机器学习工程师，擅长算法推荐。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=30
        )
        
        # 清理响应
        algorithm = response.strip()
        algorithm = algorithm.replace("```", "").strip()
        algorithm = algorithm.strip('"').strip("'")
        
        # 验证算法名称
        valid_algorithms = [
            "XGBClassifier", "XGBRegressor",
            "RandomForestClassifier", "RandomForestRegressor",
            "LogisticRegression", "LinearRegression",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "SVC", "SVR",
            "KNeighborsClassifier", "KNeighborsRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "CatBoostClassifier", "CatBoostRegressor"
        ]
        
        # 检查是否是有效的算法名称
        for valid_algo in valid_algorithms:
            if valid_algo.lower() in algorithm.lower():
                print(f"✅ LLM推荐的算法: {valid_algo}")
                return valid_algo
        
        # 如果无法识别，使用基于问题类型的默认值
        print(f"⚠️ LLM推荐 '{algorithm}' 无法识别，使用默认值: {default_algo}")
        return default_algo
        
    except Exception as e:
        print(f"❌ 算法推荐失败: {e}")
        # 返回基于文件大小的默认值
        try:
            file_size = os.path.getsize(dataset_path)
            if file_size < 1024 * 100:  # 小于100KB
                return "RandomForestClassifier"
            else:
                return "XGBClassifier"
        except:
            return "XGBClassifier"
        
# ================== 后端模型根据数据推荐算法 ==================
def call_auto_recommendation_backend(dataset_id: str, dataset_path: str) -> str:
    """
    自动推荐算法后端实现
    1. 上传数据集到远程推荐服务器
    2. 运行run.sh脚本进行分析
    3. 解析推荐结果并返回算法名称
    """
    try:
        print(f"🔬 开始自动算法推荐流程...")
        print(f"📤 dataset_id: {dataset_id}")
        print(f"📁 dataset_path: {dataset_path}")
        
        # 1. 准备上传文件
        # 将数据集上传到远程推荐服务器的 dataset/data 目录
        remote_dir = f"{REMOTE_RECOMMEND_DIR}/dataset/data"
        
        dataset_name = Path(dataset_path).stem  # 原始文件名（不带扩展名）
        
        # 生成远程文件名（使用dataset_id避免冲突）
        remote_filename = f"dataset_{dataset_name}.csv"
        
        local_files = [
            (dataset_path, remote_filename)  # 本地路径, 远程相对路径
        ]
        
        print(f"📤 准备上传数据集到远程推荐服务器...")
        print(f"📁 本地文件: {dataset_path}")
        print(f"🌐 远程目录: {remote_dir}")
        print(f"📄 远程文件名: {remote_filename}")
        
        # 2. 上传文件到远程推荐服务器
        success_files, failed_files = upload_files_to_remote(local_files, remote_dir)
        
        if failed_files:
            error_msg = f"上传数据集到推荐服务器失败: {failed_files}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        print(f"✅ 数据集上传成功: {success_files[0][1] if success_files else '未知'}")
        
        # 3. 运行远程推荐脚本
        print(f"🚀 开始运行远程推荐脚本...")
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # 连接远程服务器
            print(f"🔗 正在连接远程服务器 {REMOTE_HOST}:{REMOTE_PORT}...")
            ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
            print(f"✅ SSH连接成功")
            
            # 切换到推荐目录
            print(f"📁 切换到推荐目录: {REMOTE_RECOMMEND_DIR}")
            
             # 在执行脚本前，先检查当前目录和文件
            print(f"🔍 检查远程目录和文件...")
            
            # 检查当前目录
            check_pwd_cmd = "pwd"
            stdin, stdout, stderr = ssh.exec_command(check_pwd_cmd)
            current_dir = stdout.read().decode('utf-8').strip()
            print(f"📁 当前目录: {current_dir}")

            # 运行 run.sh 脚本
            run_script_path = f"{REMOTE_RECOMMEND_DIR}/run.sh"
            command = f"cd {REMOTE_RECOMMEND_DIR} && sh {run_script_path}  2>&1 | tee run.log"
            
            print(f"▶️ 执行命令: {command}")
            
            # 执行命令并获取输出
            stdin, stdout, stderr = ssh.exec_command(command, timeout=600)  # 10分钟超时
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            print(f"📊 脚本输出长度: {len(output)} 字符")
            print(f"⚠️ 脚本错误输出长度: {len(error)} 字符")
            
            if error:
                print(f"⚠️ 脚本执行有错误信息（前500字符）:\n{error[:500]}...")
            
            # 打印部分输出用于调试
            if output:
                print(f"📄 脚本输出预览（前500字符）:\n{output[:500]}...")
            
            # 4. 解析输出结果
            print(f"🔍 开始解析推荐结果...")
            
            # 尝试从输出中提取算法家族名称
            recommended_algorithm = None
            
            # 查找包含"算法家族名称:"的行
            lines = output.split('\n')
            for line in lines:
                if "算法家族名称:" in line:
                    print(f"📄 找到算法家族行: {line.strip()}")
                    # 提取算法家族名称
                    parts = line.split("算法家族名称:")
                    if len(parts) > 1:
                        algorithm_family = parts[1].strip()
                        algorithm_family = algorithm_family.split('|')[0].strip() if '|' in algorithm_family else algorithm_family
                        recommended_algorithm = algorithm_family
                        print(f"🎯 提取到算法家族: {algorithm_family}")
                        break
            print(f"🎯 自动推荐算法结果: {recommended_algorithm}")

            # 6. 清理远程临时文件
            try:
                print(f"🧹 清理远程临时文件...")
                cleanup_cmd = f"rm -f {remote_dir}/{remote_filename}"
                ssh.exec_command(cleanup_cmd)
                # cleanup_cmd2 = f"rm -f {REMOTE_RECOMMEND_DIR}/run.log"
                # ssh.exec_command(cleanup_cmd2)
                print(f"✅ 已清理远程临时文件")
            except Exception as cleanup_error:
                print(f"⚠️ 清理临时文件失败: {cleanup_error}")

            # if recommended_algorithm == "Other Methods (OM)" or not recommended_algorithm:
            #     print(f"⚠️ 自动推荐结果不合法，改为调用大模型推荐算法")
            #     # 调用LLM推荐算法
            #     try:
            #         print(f"🤖 正在调用大模型推荐算法...")
            #         llm_recommended_algorithm = call_llm_to_recommend_algorithm(dataset_path)
            #         print(f"✅ 大模型推荐算法: {llm_recommended_algorithm}")
                    
            #     except Exception as llm_error:
            #         print(f"❌ 大模型推荐失败: {llm_error}")
            #         # 如果LLM推荐也失败，返回默认算法
            #         recommended_algorithm="XGBClassifier"
            
            print(f"🎯 最终推荐算法结果: {recommended_algorithm}")
            ssh.close()
            print(f"🔌 已关闭SSH连接")
            return recommended_algorithm
            
        except paramiko.SSHException as ssh_error:
            print(f"❌ SSH连接错误: {ssh_error}")
            raise Exception(f"SSH连接失败: {ssh_error}")
        except Exception as e:
            print(f"❌ 运行推荐脚本时出错: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"运行推荐脚本失败: {e}")
        finally:
            if ssh:
                ssh.close()
                print(f"🔌 已关闭SSH连接")
    
    except Exception as e:
        print(f"❌ 自动推荐流程失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 失败时返回默认算法
        return "XGBClassifier"


def _get_dataset_brief(dataset_path: str) -> str:
    """
    给 LLM 一个轻量 dataset 摘要，帮助修复 dtype / NaN / 列名等问题（不喂全量数据）。
    """
    try:
        df = pd.read_csv(dataset_path, nrows=50)
        parts = []
        parts.append(f"columns={list(df.columns)}")
        parts.append(f"dtypes={df.dtypes.astype(str).to_dict()}")
        parts.append("head=" + df.head(5).to_csv(index=False))
        return "\n".join(parts)
    except Exception as e:
        return f"(无法读取数据集摘要：{e!r})"


# ================== LLM 修复 train.py ==================
def call_llm_to_fix_code(algorithm_name: str, dataset_path: str, broken_code: str, error_log: str) -> str:
    """
    把报错日志 + 原代码 + 数据集摘要发回 LLM，让它输出“完整修复版 train.py”。
    """
    if not algorithm_name:
        algorithm_name = "XGBRegressor"

    dataset_brief = _get_dataset_brief(dataset_path)

    err = (error_log or "").strip()
    if len(err) > 8000:
        err = err[-8000:]  # 通常保留 Traceback 尾部更关键

    code = (broken_code or "").strip()
    if len(code) > 20000:
        # 避免上下文爆掉：保留头尾（一般语法错误在尾部更关键，但也可能 import 缺失在头部）
        code = code[:12000] + "\n\n# ...(truncated)...\n\n" + code[-8000:]

    prompt = f"""
你是一名资深 Python 机器学习工程师。下面的 train.py 在运行时发生错误。
请你基于“错误日志 + 数据集摘要 + 原代码”给出“完整修复后的 train.py”。

【输出要求（非常重要）】
- 必须输出完整 train.py 单文件源码（不是 diff）
- 严禁输出 Markdown 代码块标记（不要 ```）
- 必须在源码最后一行单独输出结束标记：{LLM_END_MARKER}

【数据集摘要】
{dataset_brief}

【错误日志】
{err}

【原 train.py】
{code}

【修复要求】
- 修复导致报错的根因（语法、缺失 import、类型/列处理、EDA corr 报错、xgboost tree_method 等）
- 保持结构要求：main()、result 全局变量、metrics.json、清理 NaN、EDA 仅数值列
- 尽量使用算法名：{algorithm_name}

请直接输出修复后的 train.py 完整源码，并以 {LLM_END_MARKER} 结束。
""".strip()

    base_messages = [
        {"role": "system", "content": "你是专业的 Python 机器学习工程师。"},
        {"role": "user", "content": prompt},
    ]
    return _generate_with_continuation(base_messages, end_marker=LLM_END_MARKER, max_rounds=LLM_CONTINUE_MAX_ROUNDS)


# ================== Pydantic 请求体 ==================

class RunCodeRequest(BaseModel):
    dataset_id: str
    code: str
    algorithm_name: Optional[str] = None

class TrainRequest(BaseModel):
    dataset_id: str
    algorithm_name: Optional[str] = None

class FixCodeRequest(BaseModel):
    dataset_id: str
    algorithm_name: Optional[str] = None
    code: str
    error_log: str


# ================== 通用函数：上传本地文件到远程服务器 ==================
def upload_files_to_remote(local_files: List[Tuple[str, str]], remote_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    通用函数：上传多个本地文件到远程服务器
    
    参数:
        local_files: [(本地路径, 远程相对路径), ...]
        remote_dir: 远程基础目录
        
    返回:
        (成功列表, 失败列表)
    """
    print(f"📤 开始上传文件到远程服务器: {remote_dir}")
    
    ssh = None
    sftp = None
    success_files = []
    failed_files = []
    
    try:
        # 建立连接
        print("🔗 正在建立 SSH 连接...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        print("✅ SSH 连接建立成功")
        
        # 创建SFTP连接
        print("🔗 正在建立 SFTP 连接...")
        transport = ssh.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("✅ SFTP 连接建立成功")
        
        # 检查远程目录是否存在
        try:
            sftp.stat(remote_dir)
            print(f"✅ 远程目录已存在: {remote_dir}")
        except FileNotFoundError:
            print(f"⚠️ 远程目录不存在，尝试创建: {remote_dir}")
            ssh.exec_command(f"mkdir -p {remote_dir}")
        
        # 上传每个文件
        for local_path, remote_rel_path in local_files:
            remote_path = f"{remote_dir}/{remote_rel_path}"
            print(f"  📤 准备上传: {local_path} -> {remote_path}")
            
            if not os.path.exists(local_path):
                error_msg = f"本地文件不存在: {local_path}"
                print(f"    ❌ {error_msg}")
                failed_files.append((local_path, error_msg))
                continue
            
            print(f"    📊 本地文件大小: {os.path.getsize(local_path)} 字节")
            
            try:
                # 确保父目录存在
                remote_dir = os.path.dirname(remote_path)
                try:
                    sftp.stat(remote_dir)
                    print(f"    ✅ 远程目录已存在: {remote_dir}")
                except FileNotFoundError:
                    print(f"    ⚠️ 远程目录不存在，尝试创建: {remote_dir}")
                    ssh.exec_command(f"mkdir -p {remote_dir}")
                
                # 上传文件
                print(f"    ⏳ 开始上传文件...")
                sftp.put(local_path, remote_path)
                
                # 验证上传
                remote_stat = sftp.stat(remote_path)
                local_stat = os.stat(local_path)
                print(f"    ✅ 上传完成")
                print(f"    📊 验证: 本地={local_stat.st_size}字节, 远程={remote_stat.st_size}字节")
                
                if local_stat.st_size == remote_stat.st_size:
                    success_files.append((local_path, remote_path))
                    print(f"    ✅ 验证通过: 大小一致")
                else:
                    warning_msg = f"大小不一致: 本地{local_stat.st_size}字节, 远程{remote_stat.st_size}字节"
                    print(f"    ⚠️ {warning_msg}")
                    failed_files.append((local_path, warning_msg))
                    
            except Exception as e:
                error_msg = f"上传文件失败: {e!r}"
                print(f"    ❌ {error_msg}")
                import traceback
                traceback.print_exc()
                failed_files.append((local_path, error_msg))
    
    except Exception as e:
        error_msg = f"连接远程服务器失败: {e!r}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        failed_files.extend([(local_path, error_msg) for local_path, _ in local_files if (local_path, _) not in success_files])
    finally:
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()
    
    print(f"📋 上传总结: 成功 {len(success_files)} 个文件, 失败 {len(failed_files)} 个文件")
    return success_files, failed_files


# ================== 预处理数据  ==================
@app.post("/preprocess_data")
async def preprocess_data(
    file: UploadFile = File(...),
):
    print(f"\n========== /preprocess_data 开始处理 ==========")
    print(f"📤 收到请求 - 文件名: {file.filename}")

    # 1. 保存并预处理数据
    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    file_path = tmp_dir / file.filename
    print(f"📂 临时文件路径: {file_path}")

    with open(file_path, "wb") as f:
        f.write(await file.read())
        print(f"💾 临时文件已保存")
    
    try:
        print("🔄 开始处理数据集...")
        dataset_path = prepare_dataset_from_path(file_path)
        print(f"✅ 数据集已准备好，原始路径: {dataset_path}")
        
        dataset_path = auto_fix_dataset(dataset_path)
        print(f"✅ 数据集已修复（如有必要），最终路径: {dataset_path}")

        # 显示数据集基本信息
        try:
            df = pd.read_csv(dataset_path, nrows=5)
            print(f"📊 数据集预览 - 形状: {df.shape}, 列名: {list(df.columns)}")
            print(f"📊 数据集前5行:\n{df.head()}")
        except Exception as e:
            print(f"⚠️ 无法读取数据集预览: {e}")

    except Exception as e:
        print(f"❌ 处理数据集失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"处理数据集失败：{e}")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = dataset_path
    print(f"🆔 生成的 dataset_id: {dataset_id}")
    print(f"🗂️ DATASETS 字典当前大小: {len(DATASETS)}")
    
    return {
        "dataset_id": dataset_id,
        "dataset_path": dataset_path
    }


from fastapi.responses import StreamingResponse
import json

# ================== 配置远程环境（流式响应版本） ==================
@app.post("/setup_remote_env_stream")
async def setup_remote_env_stream(
    dataset_id: str = Form(...),
):
    """
    流式响应版本，实时推送日志
    """
    print(f"\n========== /setup_remote_env_stream 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {dataset_id}")
    
    async def generate():
        if not (REMOTE_HOST and REMOTE_USER and REMOTE_PASSWORD):
            yield f"data: ❌ 未配置 REMOTE_HOST / REMOTE_USER / REMOTE_PASSWORD\n\n"
            return
        
        dataset_path = DATASETS.get(dataset_id)
        if not dataset_path or not os.path.exists(dataset_path):
            yield f"data: ❌ 无效的 dataset_id 或数据文件不存在\n\n"
            return
        
        yield f"data: 🔍 找到数据集路径: {dataset_path}\n\n"
        
        # 1. 生成 requirements.txt
        try:
            requirements_path = create_requirements_file_for_dataset(dataset_path)
            yield f"data: ✅ 生成 requirements.txt: {requirements_path}\n\n"
        except Exception as e:
            yield f"data: ❌ 生成 requirements.txt 失败: {e}\n\n"
            return
        
        # 2. 上传文件
        remote_dir = f"{REMOTE_BASE_DIR}/{dataset_id}"
        local_files = [(str(requirements_path), "requirements.txt")]
        
        yield f"data: 📤 开始上传文件到远程服务器...\n\n"
        
        success_files, failed_files = upload_files_to_remote(local_files, remote_dir)
        
        if failed_files:
            error_details = "\n".join([f"{path}: {reason}" for path, reason in failed_files])
            yield f"data: ❌ 文件上传失败:\n{error_details}\n\n"
            return
        
        for local_path, remote_path in success_files:
            yield f"data: ✅ {os.path.basename(local_path)} -> {remote_path}\n\n"
        
        # 3. 远程安装环境（实时输出）
        yield f"data: 🔧 正在远程创建虚拟环境并安装依赖...\n\n"
        
        ssh = None
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
            
            remote_project_dir = f"{REMOTE_BASE_DIR}/{dataset_id}"
            CONDA_PYTHON_PATH = "/opt/miniconda3/bin/python"
            
            commands = [
                # 0. 检查 Python 环境
                f"echo '=== 1. 检查 Python 环境 ===' && "
                f"{CONDA_PYTHON_PATH} --version && "
                f"echo 'Python 路径: {CONDA_PYTHON_PATH}'",
                
                # 1. 清理旧环境
                f"echo '=== 2. 清理旧虚拟环境 ===' && cd {remote_project_dir} && pwd && rm -rf .venv",
                
                # 2. 创建虚拟环境（使用 conda 的 Python）
                f"echo '=== 3. 创建虚拟环境 ===' && cd {remote_project_dir} && "
                f"{CONDA_PYTHON_PATH} -m venv .venv",
                
                # 3. 检查虚拟环境
                f"echo '=== 4. 检查虚拟环境 ===' && cd {remote_project_dir} && "
                f"[ -f .venv/bin/python ] && echo '✅ 虚拟环境创建成功' && .venv/bin/python --version || "
                f"echo '❌ 虚拟环境创建失败'",
                
                # 4. 升级pip
                f"echo '=== 5. 升级pip ==='  && cd {remote_project_dir} && "
                f".venv/bin/python -m pip install --upgrade pip==23.3.1 "
                f"-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com",
                
                # 5. 安装依赖
                f"echo '=== 6. 安装依赖 ==='   && cd {remote_project_dir} && "
                f".venv/bin/python -m pip install  --only-binary :all: -r requirements.txt "
                f"-i https://pypi.tuna.tsinghua.edu.cn/simple "
                f"--extra-index-url https://download.pytorch.org/whl/cu118 "
                f"--trusted-host pypi.tuna.tsinghua.edu.cn "
                f"--trusted-host download.pytorch.org",
                
                # 5. 验证
                f"echo '=== 7. 验证依赖 ===' && cd {remote_project_dir} && "
                f".venv/bin/python -c \""
                f"import numpy; print('✅ numpy:', numpy.__version__);"
                f"import pandas; print('✅ pandas:', pandas.__version__);"
                f"import scipy; print('✅ scipy:', scipy.__version__);"
                f"import sklearn; print('✅ sklearn:', sklearn.__version__);"
                f"\""
            ]
            
            for i, cmd in enumerate(commands, 1):
                yield f"data: 🔧 执行步骤 {i}...\n\n"
                yield f"data: ▶️ 命令: {cmd}\n\n"
                
                stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
                
                # 实时读取输出
                while True:
                    if stdout.channel.recv_ready():
                        chunk = stdout.channel.recv(1024).decode('utf-8', errors='ignore')
                        if chunk:
                            yield f"data: {chunk}\n\n"
                    
                    if stderr.channel.recv_stderr_ready():
                        chunk = stderr.channel.recv_stderr(1024).decode('utf-8', errors='ignore')
                        if chunk:
                            yield f"data: ⚠️ {chunk}\n\n"
                    
                    if not stdout.channel.recv_ready() and not stderr.channel.recv_stderr_ready():
                        if stdout.channel.exit_status_ready():
                            break
                    
                    await asyncio.sleep(0.1)
            
            yield f"data: 🎉 环境配置完成！\n\n"
            yield f"data: ✅ 远程项目目录：{remote_project_dir}\n\n"
            yield f"data: ✅ requirements_path: {requirements_path}\n\n"
            
        except Exception as e:
            yield f"data: ❌ 远程环境设置失败：{e}\n\n"
            import traceback
            traceback.print_exc()
        finally:
            if ssh:
                ssh.close()
        
        yield f"data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
        }
    )


# ================== 配置远程环境（普通版本，保持兼容） ==================
@app.post("/setup_remote_env")
async def setup_remote_env(
    dataset_id: str = Form(...),
):
    """
    原有同步版本，保持兼容
    """
    print(f"\n========== /setup_remote_env 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {dataset_id}")

    
    # 1. 验证配置
    if not (REMOTE_HOST and REMOTE_USER and REMOTE_PASSWORD):
        error_msg = "未配置 REMOTE_HOST / REMOTE_USER / REMOTE_PASSWORD"
        raise HTTPException(status_code=500, detail=error_msg)

    # 2. 获取数据集路径
    dataset_path = DATASETS.get(dataset_id)
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="无效的 dataset_id 或数据文件不存在")
    
    print(f"📂 找到数据集路径: {dataset_path}")
    
    # 3. 生成固定版本的 requirements.txt
    requirements_path = create_requirements_file_for_dataset(dataset_path)
    print(f"✅ 生成 requirements.txt: {requirements_path}")

     # 4. 定义要上传的文件
    local_files = [
        (str(requirements_path), "requirements.txt"),
    ]
    
    remote_dir = f"{REMOTE_BASE_DIR}/{dataset_id}"

    # 5. 上传文件
    success_files, failed_files = upload_files_to_remote(local_files, remote_dir)

    if failed_files:
        error_details = "\n".join([f"{path}: {reason}" for path, reason in failed_files])
        raise HTTPException(
            status_code=500, 
            detail=f"部分文件上传失败:\n{error_details}"
        )
    
    log = "=== 文件上传完成 ===\n"
    for local_path, remote_path in success_files:
        log += f"✅ {os.path.basename(local_path)} -> {remote_path}\n"
    
    # 6. 创建虚拟环境并安装依赖
    print("🔧 正在远程创建虚拟环境并安装依赖...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        remote_project_dir = f"{REMOTE_BASE_DIR}/{dataset_id}"
        # 定义正确的 Python 路径
        CONDA_PYTHON_PATH = "/usr/local/conda/bin/python"

        commands = [
            # 0. 检查 Python 环境
            f"echo '=== 1. 检查 Python 环境 ===' && "
            f"{CONDA_PYTHON_PATH} --version && "
            f"echo 'Python 路径: {CONDA_PYTHON_PATH}'",
            
            # 1. 清理旧环境
            f"echo '=== 2. 清理旧虚拟环境 ===' && cd {remote_project_dir} && pwd && rm -rf .venv",
            
            # 2. 创建虚拟环境（使用 conda 的 Python）
            f"echo '=== 3. 创建虚拟环境 ===' && cd {remote_project_dir} && "
            f"{CONDA_PYTHON_PATH} -m venv .venv",
            
            # 3. 检查虚拟环境
            f"echo '=== 4. 检查虚拟环境 ===' && cd {remote_project_dir} && "
            f"[ -f .venv/bin/python ] && echo '✅ 虚拟环境创建成功' && .venv/bin/python --version || "
            f"echo '❌ 虚拟环境创建失败'",
            
            # 4. 升级pip
            f"echo '=== 5. 升级pip ==='  && cd {remote_project_dir} && "
            f".venv/bin/python -m pip install --upgrade pip==23.3.1 "
            f"-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com",
            
            # 5. 安装依赖
            f"echo '=== 6. 安装依赖 ==='   && cd {remote_project_dir} && "
            f".venv/bin/python -m pip install  --only-binary :all: -r requirements.txt "
            f"-i https://pypi.tuna.tsinghua.edu.cn/simple "
            f"--extra-index-url https://download.pytorch.org/whl/cu118 "
            f"--trusted-host pypi.tuna.tsinghua.edu.cn "
            f"--trusted-host download.pytorch.org",
            
            # 5. 验证
            f"echo '=== 7. 验证依赖 ===' && cd {remote_project_dir} && "
            f".venv/bin/python -c \""
            f"import numpy; print('✅ numpy:', numpy.__version__);"
            f"import pandas; print('✅ pandas:', pandas.__version__);"
            f"import scipy; print('✅ scipy:', scipy.__version__);"
            f"import sklearn; print('✅ sklearn:', sklearn.__version__);"
            f"\""
        ]
            
        
        all_output = ""
        for i, cmd in enumerate(commands, 1):
            print(f"🔧 执行步骤 {i}...")      
            print(f"    🖥️ 命令: {cmd}")      
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            out = ""
            err = "" 
            start_time = time.time()
            while True:
                if stdout.channel.recv_ready():
                    chunk = stdout.channel.recv(1024).decode('utf-8', errors='ignore')
                    out += chunk
                    if "Successfully installed" in chunk or "ERROR" in chunk:
                        print(f"📦 {chunk.strip()}")
                
                if stderr.channel.recv_stderr_ready():
                    chunk = stderr.channel.recv_stderr(1024).decode('utf-8', errors='ignore')
                    err += chunk
                    if chunk.strip():
                        print(f"⚠️ {chunk.strip()}")
                
                if not stdout.channel.recv_ready() and not stderr.channel.recv_stderr_ready():
                    if stdout.channel.exit_status_ready():
                        break
                
                # 每30秒显示一次进度
                if time.time() - start_time > 30:
                    print("⏳ 仍在安装中，请耐心等待...")
                    start_time = time.time()
                
                time.sleep(0.5)

            all_output += f"\n=== 步骤 {i} ===\n"
            if out.strip():
                all_output += f"输出: {out}\n"
            if err.strip():
                all_output += f"错误: {err}\n"
            
            print(f"✅ 步骤 {i} 完成")
        
        print("🎉 环境配置完成！")
        
        log += "\n=== 环境配置日志 ===\n" + all_output
        
        log += (
            "\n\n=== 环境已配置完成 ===\n"
            f"远程项目目录：{remote_project_dir}\n"
            "激活命令: source .venv/bin/activate\n"
            "运行训练: .venv/bin/python train.py\n"
        )
        
    except Exception as e:
        error_msg = f"远程环境设置失败：{e}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        if ssh:
            ssh.close()
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if ssh:
            ssh.close()

    print("========== /setup_remote_env 处理完成 ==========\n")

    return {
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "requirements_path": str(requirements_path),  
        "remote_project_dir": remote_project_dir,
        "result": log  # 返回日志给前端
    }

# ================== 路由：LLM推荐算法接口 ==================
@app.post("/llm_recommend_algorithm")
async def llm_recommend_algorithm(req: dict):
    """
    LLM推荐算法接口
    """
    dataset_id = req.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="缺少 dataset_id")
    
    dataset_path = DATASETS.get(dataset_id)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="无效的 dataset_id")
    
    try:
        print(f"🤖 正在为 dataset_id={dataset_id} 推荐算法...")
        algorithm = call_llm_to_recommend_algorithm(dataset_path)
        print(f"✅ 算法推荐完成: {algorithm}")
        return {
            "recommended_algorithm": algorithm,
            "source": "llm"
        }
    except Exception as e:
        print(f"❌ 算法推荐失败: {e}")
        # 返回默认算法
        import traceback
        traceback.print_exc()
        return {"recommended_algorithm": "XGBClassifier"}


@app.post("/auto_recommend_algorithm")
async def auto_recommend_algorithm(req: dict):
    """
    自动推荐算法接口
    1. 接收dataset_id
    2. 从DATASETS获取数据路径
    3. 上传到内部服务器并运行run.sh
    4. 返回推荐的算法
    """
    dataset_id = req.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="缺少 dataset_id")
    
    dataset_path = DATASETS.get(dataset_id)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="无效的 dataset_id")
    
    try:
        print(f"🔬 正在为 dataset_id={dataset_id} 进行自动算法推荐...")
        recommended_algorithm = call_auto_recommendation_backend(dataset_id, dataset_path)
        print(f"✅ 自动算法推荐完成: {recommended_algorithm}")
        
        return {
            "recommended_algorithm": recommended_algorithm,
            "source": "auto"
        }
    except Exception as e:
        print(f"❌ 自动算法推荐失败: {e}")
        import traceback
        traceback.print_exc()
        return {"recommended_algorithm": "XGBClassifier"}
        
    
# ================== 路由：生成代码 ==================
@app.post("/generate_code_and_upload")
async def generate_code_and_upload(
    dataset_id: str = Form(...),  # 从预处理接口获取的 dataset_id
    algorithm_option: str = Form("manual"),
    algorithm_name: str = Form(...),
):
    print(f"\n========== /generate_code 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {dataset_id}")
    print(f"📊 算法选项: {algorithm_option}")
    print(f"🤖 算法名称: {algorithm_name}")

    # 1. 校验 dataset_id
    dataset_path = DATASETS.get(dataset_id)
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="无效的 dataset_id 或数据文件不存在")

    # 2. 确定最终算法名称（修复这里）
    # 注意：前端已经处理了算法推荐，这里直接使用前端传入的算法名称
    final_algorithm_name = algorithm_name.strip() if algorithm_name else "XGBClassifier"
    
    # 如果算法名称为空，使用默认值
    if not final_algorithm_name:
        final_algorithm_name = "XGBClassifier"
    
    print(f"🎯 最终使用的算法: {final_algorithm_name}")

    # 3. 调用 LLM 生成代码
    try:
        print("🤖 正在调用大模型生成代码...")
        code = call_llm_to_generate_code(final_algorithm_name, dataset_path)  # 使用 final_algorithm_name
        print(f"✅ 代码生成成功，代码长度: {len(code)} 字符")

    except Exception as e:
        print(f"❌ 调用大模型失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调用大模型失败：{e}")

    # 4. 将生成的代码写入本地文件
    project_dir = Path(dataset_path).parent
    train_path = project_dir / "train.py"
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"💾 本地写入 train.py: {train_path}")

    # 5. 上传代码和数据到远程服务器
    remote_dir = f"{REMOTE_BASE_DIR}/{dataset_id}"
    local_files = [
        (str(dataset_path), "data.csv"),
        (str(train_path), "train.py")
    ]
    success_files, failed_files = upload_files_to_remote(local_files, remote_dir)
    
    if failed_files:
        error_details = "\n".join([f"{path}: {reason}" for path, reason in failed_files])
        raise HTTPException(
            status_code=500, 
            detail=f"代码上传失败:\n{error_details}"
        )

    print("========== /generate_code_and_upload 处理完成 ==========\n")

    return {
        "dataset_id": dataset_id,
        "generated_code": code,
        "algorithm_used": final_algorithm_name,
        "recommendation_source": algorithm_option  # 返回推荐来源
    }

# ================== 路由：仅修复代码（给前端“手动修复”按钮） ==================

@app.post("/fix_code")
async def fix_code(req: FixCodeRequest):
    print(f"\n========== /fix_code 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {req.dataset_id}, 算法名: {req.algorithm_name}")
    print(f"📝 原代码长度: {len(req.code)} 字符")
    print(f"❌ 错误日志长度: {len(req.error_log)} 字符")
    if req.error_log:
        print(f"❌ 错误日志预览（前300字符）:\n{req.error_log[:300]}...")

    dataset_path = DATASETS.get(req.dataset_id)
    if not dataset_path:
        error_msg = f"无效的 dataset_id: {req.dataset_id}"
        print(f"❌ {error_msg}")
        print(f"🗂️ 当前 DATASETS 中的 keys: {list(DATASETS.keys())}")
        raise HTTPException(status_code=404, detail="无效的 dataset_id")

    try:
        print("🤖 正在调用大模型修复代码...")
        fixed = call_llm_to_fix_code(req.algorithm_name or "XGBRegressor", dataset_path, req.code, req.error_log)
        print(f"✅ 代码修复成功，新代码长度: {len(fixed)} 字符")
        print(f"📝 修复后代码预览（前500字符）:\n{fixed[:500]}...")
    except Exception as e:
        print(f"❌ 调用大模型修复失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调用大模型修复失败：{e}")

    print("========== /fix_code 处理完成 ==========\n")
    
    return {"fixed_code": fixed}


# ================== 路由：在 AutoDL 上运行训练（自动修复可选） ==================

@app.post("/run_remote_train")
async def run_remote_train(req: TrainRequest):
    """
    在远程 AutoDL 上执行：
        cd <remote_project_dir> && .venv/bin/python train.py
    捕获 stdout + stderr 返回给前端。
    """
    print(f"\n========== /run_remote_train 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {req.dataset_id}, 算法名: {req.algorithm_name}")
    
    if not (REMOTE_HOST and REMOTE_USER and REMOTE_PASSWORD):
        error_msg = "未配置 REMOTE_HOST / REMOTE_USER / REMOTE_PASSWORD"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    dataset_path = DATASETS.get(req.dataset_id)
    if not dataset_path:
        error_msg = f"无效的 dataset_id: {req.dataset_id}"
        print(f"❌ {error_msg}")
        print(f"🗂️ 当前 DATASETS 中的 keys: {list(DATASETS.keys())}")
        raise HTTPException(status_code=404, detail="无效的 dataset_id，请重新生成代码并配置环境。")

    remote_project_dir = f"{REMOTE_BASE_DIR}/{req.dataset_id}"
    print(f"📁 远程项目目录: {remote_project_dir}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print("🔗 正在建立 SSH 连接...")
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        print("✅ SSH 连接建立成功")
    except Exception as e:
        error_msg = f"SSH 连接远程服务器失败：{e!r}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

    def _exec(cmd: str):
        print(f"💻 执行远程命令: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
        print(f"📤 命令输出长度 - stdout: {len(out)} 字符, stderr: {len(err)} 字符")
        return out, err

    # 先检查远程文件是否存在
    print("🔍 检查远程文件...")
    check_cmds = [
        f"cd {remote_project_dir} && ls -la",
        f"cd {remote_project_dir} && ls -la .venv/bin/python 2>/dev/null || echo '未找到 .venv'",
        f"cd {remote_project_dir} && head -20 train.py"
    ]
    
    for cmd in check_cmds:
        out, err = _exec(cmd)
        if out:
            print(f"📁 检查结果 ({cmd}):\n{out[:300]}...")

    train_cmd = f"cd {remote_project_dir} && .venv/bin/python train.py"
    print(f"🚀 开始执行训练命令: {train_cmd}")
    
    import time
    start_time = time.time()
    
    out, err = _exec(train_cmd)

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️ 训练耗时: {duration:.2f} 秒")
    print(f"📤 训练输出长度 - stdout: {len(out)} 字符, stderr: {len(err)} 字符")

    # 构建完整的日志输出
    log_lines = []
    
    # 添加标题和时间信息
    log_lines.append(f"========== 训练开始 ==========")
    log_lines.append(f"数据集ID: {req.dataset_id}")
    log_lines.append(f"算法名称: {req.algorithm_name or '未指定'}")
    log_lines.append(f"训练时长: {duration:.2f} 秒")
    log_lines.append("")
    
    # 添加标准输出
    if out and out.strip():
        log_lines.append("=== 标准输出 (stdout) ===")
        log_lines.append(out.strip())
    else:
        log_lines.append("=== 标准输出 (stdout) ===")
        log_lines.append("无输出")
    
    log_lines.append("")
    
    # 添加错误输出
    if err and err.strip():
        log_lines.append("=== 错误输出 (stderr) ===")
        log_lines.append(err.strip())
    else:
        log_lines.append("=== 错误输出 (stderr) ===")
        log_lines.append("无错误输出")
    
    log_lines.append("")
    log_lines.append("========== 训练结束 ==========")
    
    # 合并所有日志行
    full_log = "\n".join(log_lines)

    print("✅ 训练完成")
    ssh.close()
    print("========== /run_remote_train 处理完成 ==========\n")
    
    return {"result": full_log}

# ================== 根路由 & main ==================

@app.get("/")
async def root():
    return {"msg": "LLM Code Gen Backend is running"}
if __name__ == "__main__":
    import sys
    import signal
    import os
    import asyncio
    import uvicorn
    from uvicorn.config import Config
    from uvicorn.server import Server
    import psutil  # 需提前安装：pip install psutil
    print(f"🐍 Python 版本: {sys.version}")
    print(f"📦 FastAPI 后端启动中...")
    print(f"🌐 服务地址: http://0.0.0.0:8000")
    print(f"📚 API 文档: http://0.0.0.0:8000/docs")
    
    # 获取当前进程 PID（用于强制杀死）
    CURRENT_PID = os.getpid()
    print(f"🔍 当前进程 PID: {CURRENT_PID}")

    # 定义强制退出函数（仅适配 Windows 支持的信号）
    def force_exit(signum, frame):
        print("\n🛑 收到终止信号（Ctrl+C），执行强制退出流程...")
        
        # 第一步：清理关键资源（SSH/SFTP 连接）
        print("🔌 关闭所有 SSH/SFTP 连接...")
        for conn_type in ["sftp", "ssh", "transport"]:
            for conn in GLOBAL_CONNECTIONS[conn_type]:
                try:
                    conn.close()
                except:
                    pass
            GLOBAL_CONNECTIONS[conn_type].clear()
        
        # 第二步：杀死当前进程及其所有子进程（Windows 终极手段）
        print(f"🔫 强制终止进程 {CURRENT_PID} 及其子进程...")
        try:
            # 获取当前进程对象
            current_process = psutil.Process(CURRENT_PID)
            # 获取所有子进程（递归）
            children = current_process.children(recursive=True)
            # 先终止子进程
            for child in children:
                try:
                    print(f"🔪 终止子进程 {child.pid}")
                    child.terminate()  # Windows 下 terminate = 强制杀死
                except:
                    try:
                        child.kill()
                    except:
                        pass
            # 等待子进程退出
            psutil.wait_procs(children, timeout=2)
            # 最后终止主进程（Windows 下最直接的方式）
            print(f"🔚 终止主进程 {CURRENT_PID}")
            os._exit(0)  # 立即退出，不执行任何后续代码
        except Exception as e:
            print(f"⚠️ 进程清理失败，直接退出: {e}")
            os._exit(1)

    # ========== 仅注册 Windows 支持的信号 ==========
    # SIGINT：对应 Ctrl+C（Windows 唯一常用的终止信号）
    signal.signal(signal.SIGINT, force_exit)
    # 可选：注册 SIGBREAK（对应 Ctrl+Break，备用终止方式）
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, force_exit)

    # 禁用 Python 默认信号处理（避免干扰）
    if hasattr(signal, "set_wakeup_fd"):
        signal.set_wakeup_fd(-1)

    # 启动 Uvicorn（Windows 适配版）
    config = Config(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Windows 下必须关闭 reload
        workers=1,     # 单进程避免残留
        loop="asyncio"
    )
    server = Server(config=config)

    # 后台线程启动 Uvicorn（避免阻塞主线程信号处理）
    import threading
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    print("✅ Uvicorn 服务器已在后台线程启动（Windows 适配版）")

    # 主线程保持运行（等待终止信号）
    try:
        while server_thread.is_alive():
            server_thread.join(1)  # 每 1 秒检查一次
    except Exception as e:
        print(f"⚠️ 主线程异常: {e}")
        force_exit(None, None)
    import sys
    import signal
    import asyncio
    import uvicorn
    from uvicorn.config import Config
    from uvicorn.server import Server
    print(f"🐍 Python 版本: {sys.version}")
    print(f"📦 FastAPI 后端启动中...")
    print(f"🌐 服务地址: http://0.0.0.0:8000")
    print(f"📚 API 文档: http://0.0.0.0:8000/docs")
    
    # 全局变量：用于控制 Uvicorn 服务器
    server = None
    loop = None
    
    # 定义优雅退出的信号处理函数（强化版）
    def handle_exit(signum, frame):
        print("\n🛑 收到终止信号，正在优雅退出...")
        
        # 第一步：关闭所有 SSH/SFTP 连接（核心！）
        print("🔌 正在关闭所有 SSH/SFTP 连接...")
        for conn_type in ["sftp", "ssh", "transport"]:
            for conn in GLOBAL_CONNECTIONS[conn_type]:
                try:
                    conn.close()
                    print(f"✅ 已关闭 {conn_type} 连接")
                except Exception as e:
                    print(f"⚠️ 关闭 {conn_type} 连接失败: {e}")
            GLOBAL_CONNECTIONS[conn_type].clear()
        
        # 第二步：清理临时文件（可选，避免磁盘残留）
        print("🗑️ 正在清理临时文件...")
        try:
            import shutil
            # 清理所有以 upload_ / dataset_ 开头的临时目录
            for tmp_dir in Path(tempfile.gettempdir()).glob("upload_*"):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            for tmp_dir in Path(tempfile.gettempdir()).glob("dataset_*"):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            print("✅ 临时文件清理完成")
        except Exception as e:
            print(f"⚠️ 清理临时文件失败: {e}")
        
        # 第三步：终止所有子进程（如果有）
        print("🔪 正在终止子进程...")
        try:
            import psutil
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                child.terminate()
                print(f"✅ 已终止子进程 {child.pid}")
            # 等待子进程退出
            gone, alive = psutil.wait_procs(current_process.children(), timeout=5)
            for p in alive:
                p.kill()
                print(f"🔫 强制杀死未退出的子进程 {p.pid}")
        except ImportError:
            print("⚠️ 未安装 psutil，跳过子进程清理（可执行 pip install psutil）")
        except Exception as e:
            print(f"⚠️ 清理子进程失败: {e}")
        
        # 第四步：强制停止 Uvicorn 服务器和 asyncio 事件循环（关键！）
        print("🔌 正在停止 Uvicorn 服务器...")
        if server:
            # 停止 Uvicorn 服务器
            asyncio.run_coroutine_threadsafe(server.shutdown(), loop)
            print("✅ Uvicorn 服务器已停止")
        
        if loop:
            # 强制停止事件循环
            loop.stop()
            # 关闭事件循环
            if not loop.is_closed():
                loop.close()
                print("✅ asyncio 事件循环已关闭")
        
        print("✅ 优雅退出完成")
        sys.exit(0)

        # 终极兜底：强制杀死当前进程
        print("🔫 终极兜底：强制终止当前进程...")

        os._exit(0)  # 区别于 sys.exit，os._exit 会立即终止进程，不执行任何清理钩子
        
    
    # 注册所有退出信号
    signal.signal(signal.SIGINT, handle_exit)    # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)   # kill 命令
    signal.signal(signal.SIGHUP, handle_exit)    # 终端关闭
    signal.signal(signal.SIGQUIT, handle_exit)   # Ctrl+\
    
    # 手动配置并启动 Uvicorn（替代直接调用 uvicorn.run）
    config = Config(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop="asyncio",
        limit_concurrency=10,
        limit_max_requests=100,
        timeout_keep_alive=5
    )
    server = Server(config=config)
    
    # 获取并保存事件循环（用于退出时终止）
    loop = asyncio.get_event_loop()
    
    try:
        # 运行服务器（阻塞直到停止）
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        # 捕获 Ctrl+C，触发退出逻辑
        handle_exit(None, None)
    finally:
        # 兜底：确保事件循环关闭
        if not loop.is_closed():
            loop.close()