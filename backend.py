'''
Author: chensi-cs 
Date: 2026-01-15 16:12:33
LastEditors: chensi-cs 
LastEditTime: 2026-01-16 15:38:38
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

import os
import tempfile
import zipfile
import uuid
from pathlib import Path
from typing import Dict, Optional, List,Union

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

# print(f"【服务器信息】 REMOTE_HOST={REMOTE_HOST}, REMOTE_PORT={REMOTE_PORT}, REMOTE_USER={REMOTE_USER}")
# print(f"【服务器信息】 REMOTE_BASE_DIR={REMOTE_BASE_DIR}")
print(f"🌐 远程服务器配置 - 主机: {REMOTE_HOST}, 端口: {REMOTE_PORT}, 用户: {REMOTE_USER}")
print(f"📁 远程基础目录: {REMOTE_BASE_DIR}")

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

def create_env_file_for_dataset(dataset_path: str) -> str:
    project_dir = Path(dataset_path).parent
    env_path = project_dir / "requirements.txt"
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            for pkg in BASE_DEPENDENCIES:
                f.write(pkg + "\n")
        print(f"✔ 已生成环境文件: {env_path}")
    else:
        print(f"✔ 已存在环境文件: {env_path}")
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
- 数据集 CSV 路径：DATASET_PATH = "{dataset_path}"
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
        

        
# ================== LLM 修复 train.py ==================

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


# ================== 路由：生成代码（只生成，不运行） ==================

""" 
1. 用户上传数据集（支持多种格式）
2. 后端解析数据 → 修复异常 → 转为CSV
3. 调用LLM生成完整的train.py
4. 生成requirements.txt依赖文件
5. 返回dataset_id和生成的代码 
"""

@app.post("/generate_code")
async def generate_code(
    algorithm_option: str = Form("manual"),  # "manual" 或 "auto"
    algorithm_name: str = Form(...),
    file: UploadFile = File(...),
):
    print(f"\n========== /generate_code 开始处理 ==========")
    print(f"📤 收到请求 - 算法名: {algorithm_name}, 文件名: {file.filename}")

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

    env_file = create_env_file_for_dataset(dataset_path)
    # ===== 新增：算法选择逻辑 =====
    final_algorithm_name = algorithm_name

    if algorithm_option == "auto":
        print("🤖 正在调用大模型推荐算法...")
        try:
            recommended_algorithm = call_llm_to_recommend_algorithm(dataset_path)
            final_algorithm_name = recommended_algorithm
            print(f"✅ 大模型推荐的算法: {final_algorithm_name}")
        except Exception as e:
            print(f"❌ 算法推荐失败，使用默认算法: {e}")
            final_algorithm_name = "XGBClassifier"  # 默认值
    elif not algorithm_name or not algorithm_name.strip():
        print("⚠️ 未提供算法名称，使用默认算法")
        final_algorithm_name = "XGBClassifier"  # 默认值
    
    final_algorithm_name = final_algorithm_name or "XGBClassifier"
    print(f"🎯 最终使用的算法: {final_algorithm_name}")

    try:
        print("🤖 正在调用大模型生成代码...")
        code = call_llm_to_generate_code(algorithm_name, dataset_path)
        print(f"✅ 代码生成成功，代码长度: {len(code)} 字符")

    except Exception as e:
        print(f"❌ 调用大模型失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"调用大模型失败：{e}")
    
    print("========== /generate_code 处理完成 ==========\n")
    
    return {
        "dataset_id": dataset_id,
        "generated_code": code,
        "env_file": env_file,
        "algorithm_used": final_algorithm_name,  # 新增：返回实际使用的算法
        "algorithm_recommended": algorithm_option == "auto",  # 新增：是否是推荐的
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


# ================== 路由：上传到 AutoDL 并配置环境（只创建 .venv，不运行 train.py） ==================

@app.post("/setup_remote_env")
async def setup_remote_env(req: RunCodeRequest):
    """
    1) 在本地写入 train.py
    2) 上传 train.py + requirements.txt + data.csv 到 AutoDL
    3) 在 AutoDL 上创建 .venv 并安装依赖（只用 .venv，不动全局）
    """
    print(f"\n========== /setup_remote_env 开始处理 ==========")
    print(f"📤 收到请求 - dataset_id: {req.dataset_id}, 算法名: {req.algorithm_name}")
    print(f"📝 代码长度: {len(req.code)} 字符")

    if not (REMOTE_HOST and REMOTE_USER and REMOTE_PASSWORD):
        error_msg = "未配置 REMOTE_HOST / REMOTE_USER / REMOTE_PASSWORD"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    dataset_path = DATASETS.get(req.dataset_id)
    if not dataset_path:
        error_msg = f"无效的 dataset_id: {req.dataset_id}"
        print(f"❌ {error_msg}")
        print(f"🗂️ 当前 DATASETS 中的 keys: {list(DATASETS.keys())}")
        raise HTTPException(status_code=404, detail="无效的 dataset_id，请重新生成代码。")

    print(f"📂 找到数据集路径: {dataset_path}")

    # 检查 dataset_path 是否存在
    if not os.path.exists(dataset_path):
        error_msg = f"本地数据集文件不存在：{dataset_path}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    project_dir = Path(dataset_path).parent
    print(f"📁 项目目录: {project_dir}")

    # 1. 写入 train.py
    train_path = project_dir / "train.py"
    print(f"📝 train.py 路径: {train_path}")
    print(f"✅ train.py 存在: {train_path.exists()}")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(req.code)
    print(f"💾 已写入 train.py，文件大小: {os.path.getsize(train_path)} 字节")

    # 2. 生成/检查 requirements.txt
    env_path = Path(create_env_file_for_dataset(dataset_path))
    print(f"📦 requirements.txt 路径: {env_path}")
    print(f"✅ requirements.txt 存在: {env_path.exists()}")
    if env_path.exists():
        print(f"📦 requirements.txt 内容预览:")
        with open(env_path, 'r') as f:
            content = f.read()
            print(content[:200] + "..." if len(content) > 200 else content)
    
    # 3. 检查 data.csv
    print(f"📊 data.csv 路径: {dataset_path}")
    print(f"✅ data.csv 存在: {os.path.exists(dataset_path)}")
    if os.path.exists(dataset_path):
        print(f"📊 data.csv 大小: {os.path.getsize(dataset_path)} 字节")
    
    if not os.path.exists(dataset_path):
        error_msg = f"data.csv 文件不存在: {dataset_path}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    # 连接并检查远程服务器
    print(f"🌐 正在连接远程服务器 {REMOTE_HOST}:{REMOTE_PORT} 进行 SFTP 上传...")

    ssh = None
    sftp = None
    transport = None
    
    try:
        print("🔗 正在建立 SSH 连接...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        print("✅ SSH 连接建立成功")
        
        # 检查远程目录是否存在
        remote_project_dir = f"{REMOTE_BASE_DIR}/{req.dataset_id}"
        print(f"📁 远程项目目录: {remote_project_dir}")
        
        # 测试远程目录访问权限
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {REMOTE_BASE_DIR} | grep {req.dataset_id}")
        dir_list = stdout.read().decode().strip()
        print(f"🔍 远程目录检查结果: {dir_list}")
        
        # 如果目录不存在，创建它
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_project_dir}")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print(f"✅ 远程目录创建/确认成功: {remote_project_dir}")
        else:
            error_msg = f"远程目录创建失败: {stderr.read().decode()}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 检查远程目录权限
        stdin, stdout, stderr = ssh.exec_command(f"ls -ld {remote_project_dir}")
        perm_info = stdout.read().decode().strip()
        print(f"🔐 远程目录权限: {perm_info}")
        
        # 创建SFTP连接
        print("🔗 正在建立 SFTP 连接...")
        transport = ssh.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("✅ SFTP 连接建立成功")
        
        # 测试SFTP连接
        try:
            sftp.listdir(REMOTE_BASE_DIR)
            print(f"✅ SFTP 可以访问基础目录: {REMOTE_BASE_DIR}")
        except Exception as e:
            print(f"❌ SFTP 无法访问基础目录: {e}")
            raise

    except Exception as e:
        error_msg = f"连接远程服务器失败：{e!r}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        if ssh:
            ssh.close()
        raise HTTPException(status_code=500, detail=error_msg)

    # 准备上传的文件列表
    upload_files = [
        (str(train_path), f"{remote_project_dir}/train.py"),
        (str(env_path), f"{remote_project_dir}/requirements.txt"),
        (str(dataset_path), f"{remote_project_dir}/data.csv")
    ]
    
    print("📤 开始上传文件到远程服务器...")
    success_files = []
    failed_files = []
    
    for local_path, remote_path in upload_files:
        print(f"  📤 准备上传: {local_path} -> {remote_path}")
        
        if not os.path.exists(local_path):
            error_msg = f"本地文件不存在，无法上传: {local_path}"
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
            error_msg = f"上传文件失败：{e!r}"
            print(f"    ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            failed_files.append((local_path, error_msg))

    # 关闭连接
    if sftp:
        sftp.close()
    if ssh:
        ssh.close()
    
    print(f"📋 上传总结: 成功 {len(success_files)} 个文件, 失败 {len(failed_files)} 个文件")
    
    if failed_files:
        error_details = "\n".join([f"{path}: {reason}" for path, reason in failed_files])
        raise HTTPException(
            status_code=500, 
            detail=f"部分文件上传失败:\n{error_details}"
        )

    # 创建虚拟环境并安装依赖
    print("🔧 正在远程创建虚拟环境并安装依赖...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        
        # 检查远程文件
        check_cmd = f"cd {remote_project_dir} && ls -la"
        stdin, stdout, stderr = ssh.exec_command(check_cmd)
        check_output = stdout.read().decode()
        check_error = stderr.read().decode()
        print(f"🔍 远程文件列表:\n{check_output}")
        if check_error:
            print(f"⚠️ 检查文件时出错: {check_error}")

        # 创建虚拟环境并安装依赖
        # python -m venv .venv: 在远程项目目录创建名为 .venv 的虚拟环境
        setup_cmd = f"""
            cd {remote_project_dir} && \
            echo "===== 环境前置检查 =====" && \
            # 打印核心环境信息（便于排查）
            python --version && \
            pip --version 2>/dev/null || echo "系统pip未安装" && \
            echo "===== 清理旧虚拟环境 =====" && \
            rm -rf .venv && \
            echo "===== 创建虚拟环境 =====" && \
            python -m venv .venv && \
            echo "===== 升级pip到兼容版本 =====" && \
            # 锁定pip版本为23.3.1（兼容Python3.11+numpy/torch）
            .venv/bin/python -m pip install --upgrade pip==23.3.1 \
            -i https://mirrors.aliyun.com/pypi/simple \
            --trusted-host mirrors.aliyun.com && \
            echo "===== 预安装兼容版numpy（避免编译错误） =====" && \
            # 安装无需编译的numpy稳定版（1.26.4适配Python3.11）
            .venv/bin/python -m pip install numpy==1.26.4 \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --trusted-host pypi.tuna.tsinghua.edu.cn && \
            echo "===== 安装PyTorch（兼容CUDA 11.8） =====" && \
            # 核心修改：用extra-index-url替代index-url，避免源限制
            # 优先清华源下通用依赖，PyTorch专属包从官方源下
            .venv/bin/python -m pip install torch torchvision torchaudio \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --extra-index-url https://download.pytorch.org/whl/cu118 \
            --trusted-host pypi.tuna.tsinghua.edu.cn \
            --trusted-host download.pytorch.org || (
                echo "⚠️ CUDA版本不兼容，降级安装CPU版PyTorch" && \
                .venv/bin/python -m pip install torch torchvision torchaudio \
                -i https://pypi.tuna.tsinghua.edu.cn/simple \
                --trusted-host pypi.tuna.tsinghua.edu.cn
            ) && \
            echo "===== 安装业务依赖 =====" && \
            .venv/bin/python -m pip install -r requirements.txt \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --trusted-host pypi.tuna.tsinghua.edu.cn && \
            echo "===== 依赖版本验证 =====" && \
            # 验证核心依赖是否安装成功（关键：失败则整个命令返回非0）
            .venv/bin/python -c "
            import numpy
            import torch
            import pandas  # 确保pandas安装（业务核心依赖）
            print('✅ numpy版本:', numpy.__version__)
            print('✅ torch版本:', torch.__version__)
            print('✅ CUDA可用:', torch.cuda.is_available())
            print('✅ pandas版本:', pandas.__version__)
            " && \
            echo "===== 环境安装完成 ====="
            """
        
        print("🔧 正在设置远程环境...")
        stdin, stdout, stderr = ssh.exec_command(setup_cmd, get_pty=True)
        
        # 远程命令实时输出日志
        import time
        output_lines = []
        error_lines = []
        
        while True:
            if stdout.channel.recv_ready():
                line = stdout.channel.recv(1024).decode('utf-8', errors='ignore')
                if line:
                    output_lines.append(line)
                    print(f"📤 [远程输出] {line.rstrip()}")
            
            if stderr.channel.recv_stderr_ready():
                line = stderr.channel.recv_stderr(1024).decode('utf-8', errors='ignore')
                if line:
                    error_lines.append(line)
                    print(f"❌ [远程错误] {line.rstrip()}")
            
            if not stdout.channel.recv_ready() and not stderr.channel.recv_stderr_ready():
                if stdout.channel.exit_status_ready():
                    break
            
            time.sleep(0.1)
        
        exit_status = stdout.channel.recv_exit_status()
        output = "".join(output_lines)
        error = "".join(error_lines)
        
        print(f"📤 远程环境设置完成，退出状态: {exit_status}")
        
    except Exception as e:
        error_msg = f"远程环境设置失败：{e!r}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        if ssh:
            ssh.close()
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if ssh:
            ssh.close()

    log = "=== 文件上传完成 ===\n"
    for local_path, remote_path in success_files:
        log += f"✅ {os.path.basename(local_path)} -> {remote_path}\n"
    
    log += "\n=== 环境配置日志 ===\n" + (output or "[无输出]")
    if error.strip():
        log += "\n=== 错误输出（可能包含 pip 警告，可忽略） ===\n" + error

    log += (
        "\n\n=== 环境已配置完成，可在 Gradio 中点击「在 AutoDL 上运行训练」按钮启动训练 ===\n"
        f"远程项目目录：{remote_project_dir}\n"
        "（如需手动检查，可 SSH 登录该目录并查看 train.py / .venv ）\n"
    )

    print("========== /setup_remote_env 处理完成 ==========\n")
    
    return {"result": log}

# ================== 路由：在 AutoDL 上运行训练（自动修复可选） ==================

@app.post("/run_remote_train")
async def run_remote_train(req: TrainRequest):
    """
    在远程 AutoDL 上执行：
        cd <remote_project_dir> && .venv/bin/python train.py
    捕获 stdout + stderr 返回给前端。
    若检测到报错：自动调用 LLM 修复 train.py，覆盖远端，并可选重跑一次。
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
        f"cd {remote_project_dir} && ls -la .venv/bin/python 2>/dev/null || echo '未找到 .venv'", #列出虚拟环境中的Python解释器的详细信息
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

    log = "=== 训练输出（stdout） ===\n" + (out or "[无输出]")
    if (err or "").strip():
        log += "\n=== 错误输出（stderr） ===\n" + err

    log += "\n\n=== 提示 ===\n若训练脚本中使用 tqdm 等进度条，其文本输出也会出现在上述日志中。"

    err_text = (err or "").strip()
    is_error = False
    if err_text:
        print(f"⚠️ 检测到 stderr 输出，长度: {len(err_text)} 字符")
        if ("Traceback" in err_text) or ("SyntaxError" in err_text) or ("Exception" in err_text) or ("Error" in err_text):
            is_error = True
            print("❌ 检测到错误关键词，标记为错误")
            print(f"❌ 错误内容（前500字符）:\n{err_text[:500]}...")

    if (not AUTO_FIX_ON_ERROR) or (not is_error):
        print("✅ 训练完成（无错误或自动修复未启用）")
        ssh.close()
        print("========== /run_remote_train 处理完成 ==========\n")
        return {"result": log, "auto_fixed": False}

    print("🔄 检测到错误，开始自动修复流程...")

    # 读取远端当前 train.py
    code_out, code_err = _exec(f"cd {remote_project_dir} && cat train.py")
    broken_code = code_out if code_out.strip() else ""
    print(f"📝 读取远程 train.py，长度: {len(broken_code)} 字符")

    # 调 LLM 修复
    algo = req.algorithm_name or "XGBRegressor"
    try:
        print("🤖 正在调用大模型修复代码...")
        fixed_code = call_llm_to_fix_code(algo, dataset_path, broken_code, err_text)
        print(f"✅ 代码修复成功，新代码长度: {len(fixed_code)} 字符")
    except Exception as e:
        print(f"❌ 调用大模型修复失败: {e}")
        import traceback
        traceback.print_exc()
        ssh.close()
        return {"result": log + f"\n\n⚠️ 自动修复失败：{e}", "auto_fixed": False}

    # 覆盖写回远端 train.py（heredoc）
    heredoc_tag = "PYCODE_EOF_9f3a"
    payload = fixed_code.replace("\r\n", "\n")
    write_cmd = (
        f"cd {remote_project_dir} && "
        f"cat > train.py <<'{heredoc_tag}'\n"
        f"{payload}\n"
        f"{heredoc_tag}\n"
    )
    _exec(write_cmd)

    log += "\n\n=== 自动修复 ===\n检测到训练报错，已将（报错信息+代码+数据摘要）回传大模型并覆盖更新 train.py。"

    if not AUTO_FIX_RERUN:
        print("✅ 自动修复完成，不重跑训练")
        ssh.close()
        print("========== /run_remote_train 处理完成 ==========\n")
        return {"result": log, "auto_fixed": True, "fixed_code": fixed_code}

    print("🔄 自动重跑训练...")

    # 自动重跑一次
    out2, err2 = _exec(train_cmd)
    log += "\n\n=== 自动重跑输出（stdout） ===\n" + (out2 or "[无输出]")
    if (err2 or "").strip():
        log += "\n=== 自动重跑错误（stderr） ===\n" + err2

    ssh.close()
    print("========== /run_remote_train 处理完成 ==========\n")
    return {"result": log, "auto_fixed": True, "fixed_code": fixed_code}


# ================== 根路由 & main ==================

@app.get("/")
async def root():
    return {"msg": "LLM Code Gen Backend is running"}


if __name__ == "__main__":
    import sys
    print(f"🐍 Python 版本: {sys.version}")
    print(f"📦 FastAPI 后端启动中...")
    print(f"🌐 服务地址: http://0.0.0.0:8000")
    print(f"📚 API 文档: http://0.0.0.0:8000/docs")
    
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
