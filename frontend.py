# frontend.py
import requests
import gradio as gr
import os
import sys
import threading

# 启用文件更改检测
if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
    # 在调试模式下运行
    os.environ['GRADIO_WATCH_MODULES'] = 'true'

BACKEND_URL = "http://127.0.0.1:8000"

STATE = {
    "dataset_id": None,
    "algorithm_name": None,
    "last_train_log": "",
    "requirements_path": None,
    "llm_recommend_algorithm": None,
    "llm_recommendation_in_progress": False,
    "auto_recommend_algorithm": None,  
    "auto_recommendation_in_progress": False, 
    "current_mode": "自动推荐算法",  # 添加当前模式
}

# ---------------------------
# Helpers
# ---------------------------

def _looks_like_error(log_text: str) -> bool:
    if not log_text:
        return False
    s = log_text.lower()
    keywords = [
        "traceback (most recent call last)",
        "syntaxerror",
        "modulenotfounderror",
        "importerror",
        "valueerror",
        "typeerror",
        "keyerror",
        "attributeerror",
        "xgboosterror",
        "error",
        "exception",
    ]
    return any(k in s for k in keywords)

def _post_json(url: str, payload: dict, timeout: int):
    return requests.post(url, json=payload, timeout=timeout)


def call_llm_recommendation(dataset_id: str):
    """调用后端推荐算法"""
    if not dataset_id:
        return "请先进行数据预处理"
    
    try:
        print(f"🔍 正在调用LLM推荐算法... dataset_id={dataset_id}")
        resp = requests.post(
            f"{BACKEND_URL}/llm_recommend_algorithm",
            json={"dataset_id": dataset_id},
            timeout=60
        )
        
        if resp.status_code == 200:
            j = resp.json()
            llm_recommend_algorithm = j.get("llm_recommend_algorithm", "")
            print(f"✅ LLM推荐算法: {llm_recommend_algorithm}")
            return llm_recommend_algorithm
        else:
            print(f"❌ 算法推荐失败: {resp.status_code}")
            return "XGBClassifier"
            
    except Exception as e:
        print(f"❌ 算法推荐请求失败: {e}")
        return "XGBClassifier"
    
def call_auto_recommendation(dataset_id: str) -> str:
    """
    调用自动推荐算法（内部模型）
    """
    if not dataset_id :
        print("❌ 缺少数据集信息")
        return "XGBClassifier"
    
    try:
        print(f"🔍 正在调用自动推荐算法... dataset_id={dataset_id}")
        
        # 调用后端自动推荐接口
        resp = requests.post(
            f"{BACKEND_URL}/auto_recommend_algorithm",
            json={"dataset_id": dataset_id},
            timeout=300  # 5分钟超时
        )
        
        if resp.status_code == 200:
            j = resp.json()
            recommended_algorithm = j.get("recommended_algorithm", "")
            print(f"✅ 自动推荐算法: {recommended_algorithm}")
            return recommended_algorithm
        else:
            print(f"❌ 自动算法推荐失败: {resp.status_code}")
            return "XGBClassifier"
            
    except Exception as e:
        print(f"❌ 自动算法推荐请求失败: {e}")
        import traceback
        traceback.print_exc()
        return "XGBClassifier"


def trigger_background_llm_recommendation(dataset_id: str):
    """在后台触发LLM推荐算法"""
    if STATE.get("llm_recommendation_in_progress"):
        return
    
    STATE["llm_recommendation_in_progress"] = True
    STATE["llm_recommended_algorithm"] = None
    
    def recommend_and_update():
        try:
            print(f"🤖 开始后台LLM算法推荐...")
            recommended = call_llm_recommendation(dataset_id)
            STATE["llm_recommended_algorithm"] = recommended
            print(f"✅ 后台LLM推荐完成: {recommended}")
        except Exception as e:
            print(f"❌ 后台LLM推荐失败: {e}")
            STATE["llm_recommended_algorithm"] = "XGBClassifier"
        finally:
            STATE["llm_recommendation_in_progress"] = False
    
    thread = threading.Thread(target=recommend_and_update, daemon=True)
    thread.start()

def trigger_background_auto_recommendation(dataset_id: str):
    """在后台触发自动推荐算法（内部模型）"""
    if STATE.get("auto_recommendation_in_progress"):
        return
    
    STATE["auto_recommendation_in_progress"] = True
    STATE["auto_recommended_algorithm"] = None
    
    def recommend_and_update():
        try:
            print(f"🔬 开始后台自动推荐算法...")
            recommended = call_auto_recommendation(dataset_id)
            STATE["auto_recommended_algorithm"] = recommended
            print(f"✅ 后台自动推荐完成: {recommended}")
        except Exception as e:
            print(f"❌ 后台自动推荐失败: {e}")
            STATE["auto_recommended_algorithm"] = "XGBClassifier"
        finally:
            STATE["auto_recommendation_in_progress"] = False
    
    thread = threading.Thread(target=recommend_and_update, daemon=True)
    thread.start()


def update_algorithm_input(algorithm_option):
    """
    根据算法选择模式更新算法输入框的显示状态
    """
    if algorithm_option == "输入算法名":
        return gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=False)

def handle_llm_recommend_click():
    """处理大模型推荐按钮点击"""
    dataset_id = STATE.get("dataset_id")
    
    if not dataset_id:
        return "请先进行数据预处理", "大模型推荐算法", "🤖 等待数据预处理后，LLM将自动推荐算法",gr.Button(variant="primary"), gr.Button(variant="secondary")
    
    print("🔄 用户点击大模型推荐按钮...")
    
    # 更新当前模式
    STATE["current_mode"] = "大模型推荐算法"
    
    # 触发后台推荐
    trigger_background_llm_recommendation(dataset_id)
    
    # 更新按钮样式
    return (
        "正在调用大模型推荐算法...",  # status_box
        "大模型推荐算法",  # current_mode
        "🤖 LLM正在分析数据集，请稍候...",  # algorithm_info
        gr.Button(variant="primary"),  # llm_recommend_btn样式
        gr.Button(variant="secondary")  # auto_recommend_btn样式
    )

def handle_auto_recommend_click():
    """处理自动推荐按钮点击"""
    dataset_id = STATE.get("dataset_id")
    
    if not dataset_id:
        return "请先进行数据预处理", "自动推荐算法", "🔬等待数据预处理后，后台将自动推荐算法", gr.Button(variant="secondary"), gr.Button(variant="primary")
    
    print("🔄 用户点击自动推荐按钮...")
    
    # 更新当前模式
    STATE["current_mode"] = "自动推荐算法"
    
    print(f"🔍 调用后端自动推荐接口，dataset_id={dataset_id}")
    
    try:
        # 触发后台自动推荐
        # 这里需要传递dataset_id，后端会自己查找数据集路径
        trigger_background_auto_recommendation(dataset_id)
        status_msg = "正在调用自动推荐算法..."
        info_msg = "🔬 自动推荐模型正在分析数据集，请稍候..."
        
    except Exception as e:
        print(f"❌ 自动推荐初始化失败: {e}")
        status_msg = f"❌ 自动推荐初始化失败: {str(e)[:100]}"
        info_msg = "🔬 自动推荐服务异常"
    
    # 更新按钮样式
    return (
        status_msg,
        "自动推荐算法",
        info_msg,
        gr.Button(variant="secondary"),
        gr.Button(variant="primary")
    )

def ui_generate_code():
    """
    第三步：生成代码
    调用后端的 /generate_code_and_upload
    """
    dataset_id = STATE.get("dataset_id")
    current_mode = STATE.get("current_mode", "自动推荐算法")
    
    if not dataset_id:
        return "", "请先进行数据预处理", "请先进行数据预处理"  # 多返回一个参数
    
    print(f"🤖 开始生成代码: dataset_id={dataset_id}, mode={current_mode}")
    
    final_algorithm = ""  # 初始化

    # 确定要使用的算法
    if current_mode == "大模型推荐算法":
        # 使用LLM推荐的算法
        final_algorithm = STATE.get("llm_recommended_algorithm", "")
        if not final_algorithm:
            print("⚠️ 未找到LLM推荐的算法，重新推荐...")
            final_algorithm = call_llm_recommendation(dataset_id)
            STATE["llm_recommended_algorithm"] = final_algorithm  # 更新状态
    
    elif current_mode == "自动推荐算法":
        # 使用自动推荐的算法
        final_algorithm = STATE.get("auto_recommended_algorithm", "")
        if not final_algorithm:
            print("⚠️ 未找到自动推荐的算法，尝试获取数据集路径并推荐...")
            try:
                from backend import DATASETS
                dataset_path = DATASETS.get(dataset_id)
                if dataset_path:
                    final_algorithm = call_auto_recommendation(dataset_id, dataset_path)
                else:
                    return "", "❌ 未找到数据集路径，请重新预处理", "❌ 未找到数据集路径"
            except ImportError:
                return "", "❌ 无法连接自动推荐服务，请稍后再试", "❌ 无法连接自动推荐服务"
        # 确保状态更新
        if final_algorithm and not STATE.get("auto_recommended_algorithm"):
            STATE["auto_recommended_algorithm"] = final_algorithm
    
    # 确保算法名称不为空
    if not final_algorithm:
        final_algorithm = "XGBClassifier"
    
    print(f"🎯 最终使用的算法: {final_algorithm}")
    
    # 根据当前模式生成算法信息显示
    if current_mode == "大模型推荐算法":
        algo_info = f"🤖 LLM推荐算法: {final_algorithm}\n点击生成代码使用此算法"
    else:
        algo_info = f"🔬 自动推荐算法: {final_algorithm}\n点击生成代码使用此算法"

    data = {
        "dataset_id": dataset_id,
        "algorithm_option": current_mode,  # 发送模式信息
        "algorithm_name": final_algorithm      # 发送实际使用的算法
    }
    
    try:
        resp = requests.post(
            f"{BACKEND_URL}/generate_code_and_upload",
            data=data,
            timeout=600
        )
        
        if resp.status_code != 200:
            return "", f"生成代码失败\n{resp.text}", algo_info
        
        j = resp.json()
        code = j.get("generated_code", "")
        algorithm_used = j.get("algorithm_used", "")
        
        if not code:
            return "", "代码生成失败", algo_info
        
        STATE["algorithm_name"] = algorithm_used or final_algorithm
        
        if current_mode == "大模型推荐算法":
            status = f"✅ 代码生成完成 (LLM推荐: {algorithm_used or final_algorithm})"
        else:
            status = f"✅ 代码生成完成 (自动推荐: {algorithm_used or final_algorithm})"
        
        return code, status, algo_info  # 返回三个值
        
    except Exception as e:
        return "", f"生成代码请求失败: {e}", algo_info
    
def ui_preprocess_data(file_path: str):
    """
    第一步：数据预处理
    调用后端的 /preprocess_data
    """
    if not file_path:
        return (
            gr.DataFrame(
                value=None, 
                visible=False, 
                headers=[f"列{i+1}" for i in range(5)],
                row_count=5,
                col_count=5
            ),
            "❌ 请先上传数据集文件"
        )
    
    print(f"📤 开始预处理数据: {file_path}")
    
    try:
        files = {"file": open(file_path, "rb")}
        try:
            resp = requests.post(
                f"{BACKEND_URL}/preprocess_data",
                files=files,
                timeout=300
            )
        finally:
            files["file"].close()
        
        if resp.status_code != 200:
            return (
                gr.DataFrame(
                    value=None, 
                    visible=False, 
                    headers=[f"列{i+1}" for i in range(5)],
                    row_count=5,
                    col_count=5
                ),
                f"❌ 预处理失败: {resp.status_code}\n{resp.text}"
            )
        
        j = resp.json()
        dataset_id = j.get("dataset_id")
        dataset_path = j.get("dataset_path", "")
        
        if not dataset_id:
            return (
                gr.DataFrame(
                    value=None, 
                    visible=False, 
                    headers=[f"列{i+1}" for i in range(5)],
                    row_count=5,
                    col_count=5
                ),
                "❌ 预处理失败：后端未返回 dataset_id"
            )
        
        # 保存 dataset_id 到全局状态
        STATE["dataset_id"] = dataset_id
        
        # 数据预览 - 显示所有列，只显示5行
        preview_data = []
        preview_headers = []
        status_msgs = [f"✅ 数据预处理完成 (dataset_id={dataset_id})"]
        
        try:
            import pandas as pd
            # 读取数据
            df = pd.read_csv(dataset_path)
            
            # 取前5行（显示所有列）
            df_preview = df.head(5)
            
            # 转换为 Gradio DataFrame 兼容格式
            preview_data = df_preview.values.tolist()
            preview_headers = df_preview.columns.tolist()
            
            status_msgs.append(f"📊 数据集: {len(df)}行 × {len(df.columns)}列")
            
        except Exception as e:
            preview_data = []
            preview_headers = []
            status_msgs.append(f"⚠️ 无法预览数据: {str(e)[:100]}")
        
        status = " | ".join(status_msgs)
        
        # 如果成功读取到数据，显示预览表格
        if preview_data and preview_headers:
            return (
                gr.DataFrame(
                    value=preview_data,
                    visible=True,
                    headers=preview_headers,
                    row_count=5,
                    col_count=len(preview_headers),  # 显示所有列
                    wrap=True,  # 允许文本换行
                    height=350  # Gradio 3.x 使用 height 参数
                ),
                status
            )
        else:
            return (
                gr.DataFrame(
                    value=None,
                    visible=False,
                    headers=[f"列{i+1}" for i in range(5)],
                    row_count=5,
                    col_count=5
                ),
                status
            )
        
    except Exception as e:
        error_msg = f"❌ 预处理请求失败: {e!r}"
        print(error_msg)
        return (
            gr.DataFrame(
                value=None, 
                visible=False, 
                headers=[f"列{i+1}" for i in range(5)],
                row_count=5,
                col_count=5
            ),
            error_msg
        )
    
def ui_setup_remote_env():
    """
    第二步：配置远程环境
    调用后端的 /setup_remote_env_stream
    """
    dataset_id = STATE.get("dataset_id")
    
    if not dataset_id:
        yield "# 请先进行数据预处理。", "❌ 未找到 dataset_id"
        return
    
    print(f"🔧 开始配置远程环境: dataset_id={dataset_id}")
    
    try:
        data = {"dataset_id": dataset_id}
        
        # 使用流式请求到新的端点
        with requests.post(
            f"{BACKEND_URL}/setup_remote_env_stream",  # 注意：改为新端点
            data=data,
            stream=True,
            timeout=1800
        ) as resp:
            
            if resp.status_code != 200:
                yield f"# 环境配置失败\n{resp.text}", f"❌ 环境配置失败: {resp.status_code}"
                return
            
            # 累积日志
            full_log = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        # SSE 格式解析
                        if line.startswith(b'data: '):
                            content = line.decode('utf-8')[6:]  # 去掉 "data: "
                            if content == "[DONE]":
                                break
                            
                            full_log += content
                            # 实时更新到前端
                            yield full_log, "⏳ 环境配置中..."
                            
                    except Exception as e:
                        print(f"解析日志行失败: {e}")
            
            # 解析最终的 requirements_path
            import re
            req_path_match = re.search(r'requirements_path: (.+)', full_log)
            if req_path_match:
                STATE["requirements_path"] = req_path_match.group(1)
            
            status = f"✅ 远程环境配置完成"
            yield full_log, status
            
    except Exception as e:
        error_msg = f"❌ 环境配置请求失败: {e!r}"
        print(error_msg)
        yield error_msg, error_msg

def ui_run_remote_train(code: str):
    """
    第四步：运行训练
    调用后端的 /run_remote_train
    """
    dataset_id = STATE.get("dataset_id")
    if not dataset_id:
        return code, "", "当前没有 dataset_id，请先进行数据预处理。"
    
    algorithm_name = STATE.get("algorithm_name")
    if not algorithm_name:
        return code, "", "当前没有算法名称，请先生成代码。"
    
    payload = {
        "dataset_id": dataset_id,
        "algorithm_name": algorithm_name,
    }

    print(f"🚀 发送训练请求到后端...")
    print(f"📤 dataset_id: {dataset_id}")
    print(f"🤖 algorithm_name: {algorithm_name}")

    try:
        # 使用 requests 而不是 _post_json 以便调试
        resp = requests.post(
            f"{BACKEND_URL}/run_remote_train",
            json=payload,
            timeout=3600
        )
    except requests.exceptions.Timeout:
        error_msg = "❌ 训练请求超时（60分钟），请检查远程服务器状态。"
        print(error_msg)
        return code, "", error_msg
    except Exception as e:
        error_msg = f"❌ 远程训练请求失败：{e!r}"
        print(error_msg)
        return code, "", error_msg

    # 检查响应状态码
    if resp.status_code != 200:
        error_detail = resp.text[:500] if resp.text else "无错误信息"
        error_msg = f"❌ 远程训练失败：{resp.status_code}\n错误详情：{error_detail}"
        print(error_msg)
        print(f"响应内容: {resp.text}")
        return code, "", error_msg

    # 尝试解析JSON响应
    try:
        j = resp.json()
        print(f"📊 后端返回JSON keys: {list(j.keys())}")
        
        # 关键：正确获取result字段
        train_log = j.get("result", "")
        
        if not train_log:
            train_log = "⚠️ 后端返回了空日志。可能训练脚本没有输出。"
            print("⚠️ 后端返回的result字段为空")
        else:
            print(f"📝 收到训练日志，长度: {len(train_log)} 字符")
            print(f"📝 日志前200字符: {train_log[:200]}")
        
        # 直接返回训练日志到train_log_box
        return code, train_log, "✅ 训练已完成"
        
    except ValueError as e:
        # JSON解析失败
        error_msg = f"❌ 无法解析后端响应为JSON：{e!r}"
        print(error_msg)
        print(f"原始响应: {resp.text[:500]}")
        return code, "", error_msg
    except Exception as e:
        error_msg = f"❌ 处理响应时出错：{e!r}"
        print(error_msg)
        return code, "", error_msg

def ui_fix_code(code: str, error_log: str):
    """
    第五步：修复代码
    调用后端的 /fix_code
    """
    dataset_id = STATE.get("dataset_id")
    if not dataset_id:
        return code, "❌ 当前没有 dataset_id，请先进行数据预处理。"

    algorithm_name = STATE.get("algorithm_name")
    if not algorithm_name:
        return code, "❌ 当前没有算法名称，请先生成代码。"

    err = (error_log or "").strip()
    if not err:
        err = (STATE.get("last_train_log") or "").strip()

    if not err:
        return code, "❌ 没有可用的报错信息：请粘贴 Traceback 或先运行训练一次。"

    payload = {
        "dataset_id": dataset_id,
        "algorithm_name": algorithm_name,
        "code": code or "",
        "error_log": err,
    }

    try:
        resp = _post_json(f"{BACKEND_URL}/fix_code", payload, timeout=600)
    except Exception as e:
        return code, f"❌ 修复请求失败：{e!r}"

    if resp.status_code != 200:
        return code, f"❌ 修复失败：{resp.status_code}\n{resp.text}"

    j = resp.json()
    fixed_code = j.get("fixed_code", "")
    if not fixed_code.strip():
        return code, "❌ 后端未返回 fixed_code。"

    return fixed_code, "✅ 已修复代码并回填（请再次点击运行训练）。"

def ui_oneclick_execute(file_path: str):
    """
    一键执行全流程：预处理 → 配置环境 → 生成代码 → 运行训练
    """
    # 1. 数据预处理
    data_preview, preprocess_status = ui_preprocess_data(file_path)
    
    if "❌" in preprocess_status:
        return data_preview, preprocess_status, "", "", "", "", ""
    
    yield data_preview, "✅ 数据预处理完成，正在配置环境...", "", "", "", "", ""
    
    # 2. 配置环境
    env_log = ""
    for env_log_part, env_status in ui_setup_remote_env():
        env_log = env_log_part
        yield data_preview, env_status, "", env_log, "", "", ""
    
    yield data_preview, "✅ 环境配置完成，正在生成代码...", "", env_log, "", "", ""
    
    # 3. 生成代码（使用自动推荐模式）
    STATE["current_mode"] = "自动推荐算法"
    code, gen_status, algo_info = ui_generate_code()
    
    yield data_preview, gen_status, algo_info, env_log, code, "", ""
    
    if "❌" in gen_status:
        return
    
    yield data_preview, "✅ 代码生成完成，正在运行训练...", algo_info, env_log, code, "", ""
    
    # 4. 运行训练
    code_final, train_log, train_status = ui_run_remote_train(code)
    
    yield data_preview, train_status, algo_info, env_log, code_final, train_log, ""

# ---------------------------
# UI
# ---------------------------
def build_demo():
    # 读取CSS文件
    try:
        with open("frontend.css", "r", encoding="utf-8") as f:
            css_content = f.read()
    except FileNotFoundError:
        print("⚠️ CSS文件未找到，使用默认样式")
        css_content = ""
    
    with gr.Blocks(
        title="LLM 生成训练代码 + 远程训练",
        theme=gr.themes.Soft(),
        css=css_content
    ) as demo:
        
        # 1. 使用说明的可折叠面板
        with gr.Accordion("📘 使用指南（点击展开）", open=False):
            gr.Markdown("""
            ### 两种使用方式：
            
            **方式一：分步执行（推荐新手）**
            1. **📁 上传数据集** → **📊 数据预处理**
            2. **🤖 选择算法模式**：
            - 大模型推荐：让AI分析并推荐算法
            - 自动推荐：使用内置模型推荐算法
            3. **⚙️ 配置环境** → **🤖 生成代码** → **⚡ 运行训练**
            4. 如需调试：使用**🔧 修复代码**功能
            
            **方式二：一键执行（快速开始）**
            - 上传数据集后，直接点击 **🚀 一键执行全流程**
            - 系统将自动完成：预处理 → 配置环境 → 生成代码 → 运行训练
            
            ---
            **💡 提示**：
            - 支持格式：csv/zip/rar/data/arff
            - 大模型推荐需要联网
            - 训练结果在右侧日志区查看
            """)
        
        # 2. 优化第一行布局：文件上传和算法选择
        with gr.Row(equal_height=True):
            # 左侧：文件上传（固定高度）
            with gr.Column(min_width=300):
                file_input = gr.File(
                    label="📁 上传数据集文件",
                    file_types=[".csv", ".zip", ".rar", ".data", ".arff"],
                    type="filepath",
                    height=80,
                    elem_id="file_upload"
                )
            
            # 右侧：算法选择（与左侧高度匹配）
            with gr.Column(min_width=300):
                with gr.Group():
                    # 标题区域
                    gr.Markdown("🔍 根据数据集自动推荐算法")
                    
                    # 按钮行
                    with gr.Row():
                        llm_recommend_btn = gr.Button(
                            "🤖 大模型推荐",
                            variant="secondary",
                            size="sm",
                            min_width=140,
                            elem_id="llm_recommend_btn"
                        )
                        
                        auto_recommend_btn = gr.Button(
                            "🔬 后台自动推荐",
                            variant="primary",  # 默认选择自动推荐
                            size="sm",
                            min_width=140,
                            elem_id="auto_recommend_btn"
                        )
                    
                    # 算法信息显示框
                    algorithm_info = gr.Textbox(
                        label="🎯 算法推荐结果",
                        lines=2,
                        placeholder="请选择推荐模式并等待结果...",
                        interactive=False,
                        visible=True,
                        elem_id="algo_info"
                    )
                current_mode = gr.State(value="自动推荐算法") 
                
        # 4. 数据预览区域（默认隐藏）
        with gr.Row():
            data_preview = gr.DataFrame(
                label="📊 数据集预览（前5行）",
                headers=[f"列{i+1}" for i in range(5)],
                row_count=5,
                col_count=5,
                visible=False,
                elem_id="data_preview",
                interactive=False
            )
        
        # 5. 操作按钮行
        with gr.Row():
            preprocess_btn = gr.Button(
                "📊 ① 数据预处理",
                variant="primary",
                scale=1,
                min_width=120,
                elem_id="preprocess_btn"
            )
            setup_env_btn = gr.Button(
                "⚙️ ② 配置环境",
                variant="primary",
                scale=1,
                min_width=120,
                elem_id="setup_btn"
            )
            generate_btn = gr.Button(
                "🤖 ③ 生成代码",
                variant="primary",
                scale=1,
                min_width=120,
                elem_id="generate_btn"
            )
            run_btn = gr.Button(
                "⚡ ④ 运行训练",
                variant="secondary",
                scale=1,
                min_width=120,
                elem_id="train_btn"
            )
            fix_btn = gr.Button(
                "🔧 ⑤ 修复代码",
                variant="secondary",
                scale=1,
                min_width=120,
                elem_id="fix_btn"
            )
            oneclick_btn = gr.Button(
                "🚀 一键执行",
                variant="primary",
                scale=1,
                min_width=120,
                elem_id="oneclick_btn"
            )
        
        # 6. 状态和信息显示
        with gr.Row():
            status_box = gr.Textbox(
                label="📊 系统状态",
                lines=3,
                interactive=False,
                elem_id="status_box"
            )
        

        
        # 7. 代码编辑区
        code_box = gr.Code(
            label="🐍 train.py（可编辑代码）",
            language="python",
            lines=28,
            interactive=True,
            elem_id="code_box"
        )
        
        # 8. 日志显示区
        with gr.Row():
            with gr.Column(scale=1):
                env_log_box = gr.Textbox(
                    label="🔧 环境配置日志",
                    lines=14,
                    interactive=False,
                    elem_id="env_log"
                )
            
            with gr.Column(scale=1):
                train_log_box = gr.Textbox(
                    label="📈 训练日志（stdout/stderr）",
                    lines=14,
                    interactive=False,
                    elem_id="train_log"
                )
        
        # 9. 错误信息粘贴区
        with gr.Row():
            error_input = gr.Textbox(
                label="📋 错误信息粘贴区（可选）",
                lines=6,
                placeholder="可以粘贴报错信息（Traceback/stderr）\n留空则使用最近一次训练日志",
                interactive=True,
                elem_id="error_input"
            )
        
        # ================== 交互逻辑函数 ==================
        def update_algorithm_info():
            """
            根据当前模式更新算法信息显示
            """
            current_mode = STATE.get("current_mode", "自动推荐算法")
            
            if current_mode == "大模型推荐算法":
                if STATE.get("llm_recommendation_in_progress"):
                    return "🤖 LLM正在分析数据集，请稍候..."
                elif STATE.get("llm_recommended_algorithm"):
                    return f"🤖 LLM推荐算法: {STATE['llm_recommended_algorithm']}\n点击生成代码使用此算法"
                else:
                    return "🤖 请点击上方'大模型推荐'按钮开始推荐"
            
            elif current_mode == "自动推荐算法":
                if STATE.get("auto_recommendation_in_progress"):
                    return "🔬 自动推荐模型正在分析数据集，请稍候..."
                elif STATE.get("auto_recommended_algorithm"):
                    return f"🔬 自动推荐算法: {STATE['auto_recommended_algorithm']}\n点击生成代码使用此算法"
                else:
                    return "🔬 请点击上方'自动推荐'按钮开始推荐"
            
            else:
                return "⚠️ 请选择算法推荐模式"
        # ================== 事件绑定 ==================
    
    
        # 1. 预处理按钮
        preprocess_btn.click(
            fn=ui_preprocess_data,
            inputs=[file_input],
            outputs=[data_preview, status_box]
        )    

        # 2. 大模型推荐按钮
        llm_recommend_btn.click(
            fn=handle_llm_recommend_click,
            inputs=[],
            outputs=[status_box, current_mode, algorithm_info, llm_recommend_btn, auto_recommend_btn]
        )
        
        # 3. 自动推荐按钮
        auto_recommend_btn.click(
            fn=handle_auto_recommend_click,
            inputs=[],
            outputs=[status_box, current_mode, algorithm_info, llm_recommend_btn, auto_recommend_btn]
        )
        # 4. 配置环境按钮
        setup_env_btn.click(
            fn=ui_setup_remote_env,
            inputs=[],
            outputs=[env_log_box, status_box]
        )
        # 5. 生成代码按钮
        generate_btn.click(
            fn=ui_generate_code,
            inputs=[],
            outputs=[code_box, status_box,algorithm_info]
        )
        # 6. 运行训练按钮
        run_btn.click(
            fn=ui_run_remote_train,
            inputs=[code_box],
            outputs=[code_box, train_log_box, status_box]
        )
        # 7. 修复代码按钮
        fix_btn.click(
            fn=ui_fix_code,
            inputs=[code_box, error_input],
            outputs=[code_box, status_box]
        )
        # 在事件绑定部分添加：
        oneclick_btn.click(
            fn=ui_oneclick_execute,
            inputs=[file_input],
            outputs=[
                data_preview, 
                status_box, 
                algorithm_info,
                env_log_box,
                code_box,
                train_log_box,
                error_input
            ]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # 开启 debug 模式（自动重载），禁用公网分享（仅本地访问）
    demo.launch(
        debug=True,               # 关键：开启自动重载
        share=False               # 关闭公网分享链接
    )
