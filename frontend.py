# frontend.py
import requests
import gradio as gr

BACKEND_URL = "http://127.0.0.1:8000"

STATE = {
    "dataset_id": None,
    "algorithm_name": None,
    "last_train_log": "",
}


# ---------------------------
# Helpers
# ---------------------------

# 检测日志中是否包含常见错误关键词
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

# 封装 POST 请求的辅助函数
def _post_json(url: str, payload: dict, timeout: int):
    return requests.post(url, json=payload, timeout=timeout)


# ---------------------------
# ①+② 一键：生成代码 + 配置远程环境
# ---------------------------

def ui_generate_and_setup(file_path: str, algorithm_name: str):
    """
    一键运行：
    ① /generate_code：上传数据集 + 算法名 -> 返回 dataset_id + generated_code
    ② /setup_remote_env：用返回的 code + dataset_id -> 远程创建 .venv 并装依赖
    """
    if not file_path:
        return "# 请先上传数据集文件。", "", "❌ 未上传数据集。"

    if not algorithm_name or not algorithm_name.strip():
        return "# 请先输入算法名称，例如：XGBRegressor", "", "❌ 未输入算法名称。"

    algorithm_name = algorithm_name.strip()

    # ---- ① generate ----
    files = {"file": open(file_path, "rb")}
    data = {"algorithm_name": algorithm_name}
    try:
        resp = requests.post(
            f"{BACKEND_URL}/generate_code",
            data=data,
            files=files,
            timeout=600
        )
    finally:
        files["file"].close()

    if resp.status_code != 200:
        return "# 生成代码失败\n" + resp.text, "", f"❌ /generate_code 失败：{resp.status_code}"

    j = resp.json()
    dataset_id = j.get("dataset_id")
    code = j.get("generated_code", "")
    env_file = j.get("env_file", "")

    if not dataset_id:
        return "# 后端未返回 dataset_id。", "", "❌ 后端未返回 dataset_id。"

    STATE["dataset_id"] = dataset_id
    STATE["algorithm_name"] = algorithm_name
    STATE["last_train_log"] = ""

    status = f"✅ 已生成代码。dataset_id={dataset_id}"
    if env_file:
        status += f" | requirements.txt={env_file}"

    # ---- ② setup env ----
    if not code or not code.strip():
        return "# 后端未生成代码。", "", "❌ generated_code 为空，无法配置远程环境。"

    payload = {
        "dataset_id": dataset_id,
        "code": code,
        "algorithm_name": algorithm_name,
    }

    try:
        resp2 = _post_json(f"{BACKEND_URL}/setup_remote_env", payload, timeout=1800)
    except Exception as e:
        return code, "", f"❌ /setup_remote_env 请求失败：{e!r}"

    if resp2.status_code != 200:
        return code, "", f"❌ 远程环境配置失败：{resp2.status_code}\n{resp2.text}"

    j2 = resp2.json()
    env_log = j2.get("result", "后端未返回环境配置日志。")

    status += " | ✅ 远程环境已配置完成（.venv 已创建）"
    return code, env_log, status


# ---------------------------
# ③ 手动开始训练：但加入“自动修复”（不自动重跑）
# ---------------------------

def ui_run_remote_train(code: str):
    """
    手动点击训练按钮：
    - 调 /run_remote_train 获取训练日志
    - 若检测到报错：前端自动调用 /fix_code 修复代码并回填（但不自动重跑）
    """
    dataset_id = STATE.get("dataset_id")
    if not dataset_id:
        return code, "当前没有 dataset_id，请先点击「①+② 一键生成并配置环境」。", "❌ 未找到 dataset_id。"

    payload = {
        "dataset_id": dataset_id,
        "algorithm_name": STATE.get("algorithm_name"),
    }

    try:
        resp = _post_json(f"{BACKEND_URL}/run_remote_train", payload, timeout=3600)
    except Exception as e:
        return code, f"远程训练请求失败：{e!r}", "❌ 训练请求失败。"

    if resp.status_code != 200:
        return code, f"远程训练失败：{resp.status_code}\n{resp.text}", "❌ 训练失败。"

    j = resp.json()
    train_log = j.get("result", "后端未返回训练日志。")
    STATE["last_train_log"] = train_log

    # 情况 A：后端已经自动修复并返回 fixed_code（你后端若开启 AUTO_FIX）
    auto_fixed = bool(j.get("auto_fixed", False))
    fixed_code = j.get("fixed_code", None)

    if auto_fixed and fixed_code and fixed_code.strip():
        code = fixed_code
        status = "✅ 训练过程中后端已自动修复 train.py，代码已回填（未在前端自动重跑）。"
        return code, train_log, status

    # 情况 B：后端没自动修复（或没开启），前端来做“自动修复”（不重跑）
    if _looks_like_error(train_log):
        status = "⚠️ 检测到报错，正在自动调用 /fix_code 修复（不自动重跑）..."

        # 用训练日志作为 error_log
        fix_payload = {
            "dataset_id": dataset_id,
            "algorithm_name": STATE.get("algorithm_name"),
            "code": code or "",
            "error_log": train_log,
        }

        try:
            fix_resp = _post_json(f"{BACKEND_URL}/fix_code", fix_payload, timeout=600)
        except Exception as e:
            status = f"❌ 自动修复请求失败：{e!r}"
            return code, train_log, status

        if fix_resp.status_code != 200:
            status = f"❌ 自动修复失败：{fix_resp.status_code}\n{fix_resp.text}"
            return code, train_log, status

        fj = fix_resp.json()
        new_code = fj.get("fixed_code", "")
        if new_code and new_code.strip():
            code = new_code
            status = "✅ 已自动修复 train.py 并回填到代码框（请你手动再次点击训练）。"
        else:
            status = "❌ 自动修复：后端未返回 fixed_code。"

        return code, train_log, status

    # 无错误
    return code, train_log, "✅ 训练已完成（未检测到明显报错）。"


# ---------------------------
# （可选）手动修复按钮：你想自己贴报错时用
# ---------------------------

def ui_manual_fix_code(code: str, error_log: str):
    dataset_id = STATE.get("dataset_id")
    if not dataset_id:
        return code, "❌ 当前没有 dataset_id，请先一键生成。"

    err = (error_log or "").strip()
    if not err:
        err = (STATE.get("last_train_log") or "").strip()

    if not err:
        return code, "❌ 没有可用的报错信息：请粘贴 Traceback 或先训练一次。"

    payload = {
        "dataset_id": dataset_id,
        "algorithm_name": STATE.get("algorithm_name"),
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

    return fixed_code, "✅ 已手动修复并回填（请你再手动点击训练）。"


# ---------------------------
# UI
# ---------------------------

def build_demo():
    with gr.Blocks(title="LLM 生成训练代码 + AutoDL 远程训练", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            ## 🚀 使用说明
            
            1. **上传数据集**（csv/zip/rar/data/arff格式）
            2. **选择算法模式**：
               - 📝 **输入算法名**：手动输入算法名称
               - 🤖 **大模型自动推荐算法**：LLM分析数据集并推荐最佳算法
            3. 点击 **①+② 一键生成并配置环境**
            4. 在代码框中查看/修改生成的 train.py
            5. 点击 **③ 手动开始训练**
               - 若训练报错：系统自动修复代码并回填（**不自动重跑**）
               - 修复后请**手动再次点击训练**
            """
        )

        # 第一行：文件上传
        with gr.Row():
            file_input = gr.File(
                label="📁 上传数据集文件",
                file_types=[".csv", ".zip", ".rar", ".data", ".arff"],
                type="filepath",
                height=100
            )
        
        # 第二行：算法选择
        with gr.Row():
            with gr.Column(scale=1):
                algorithm_option = gr.Dropdown(
                    label="🔍 算法选择模式",
                    choices=["输入算法名", "大模型自动推荐算法"],
                    value="输入算法名",
                    interactive=True
                )
            
            with gr.Column(scale=2):
                algo_input = gr.Textbox(
                    label="📝 算法名称（当选择'输入算法名'时有效）",
                    placeholder="例如：XGBRegressor, RandomForestClassifier, LogisticRegression...",
                    visible=True,
                    interactive=True
                )
        
        # 第三行：操作按钮
        with gr.Row():
            oneclick_btn = gr.Button(
                "①+② 一键生成并配置环境",
                variant="primary",
                size="lg"
            )
            run_btn = gr.Button(
                "③ 手动开始训练",
                variant="secondary",
                size="lg"
            )
        
        # 第四行：状态和信息显示
        with gr.Row():
            status_box = gr.Textbox(
                label="📊 状态",
                lines=2,
                interactive=False
            )
        
        with gr.Row():
            algorithm_info = gr.Textbox(
                label="🎯 当前使用的算法",
                lines=1,
                placeholder="此处将显示实际使用的算法名称",
                interactive=False,
                visible=True
            )
        
        # 第五行：代码编辑区
        code_box = gr.Code(
            label="🐍 train.py（可编辑代码）",
            language="python",
            lines=28,
            interactive=True
        )
        
        # 第六行：日志显示区
        with gr.Row():
            with gr.Column(scale=1):
                env_log_box = gr.Textbox(
                    label="🔧 环境配置日志（AutoDL .venv）",
                    lines=14,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                train_log_box = gr.Textbox(
                    label="📈 训练日志（stdout/stderr）",
                    lines=14,
                    interactive=False
                )
        
        # 第七行：手动修复区域
        with gr.Row():
            manual_fix_btn = gr.Button(
                "🔧 （可选）手动修复代码",
                variant="secondary",
                size="sm"
            )
        
        with gr.Row():
            error_input = gr.Textbox(
                label="📋 错误信息粘贴区（可选）",
                lines=6,
                placeholder="可以粘贴报错信息（Traceback/stderr）\n留空则使用最近一次训练日志",
                interactive=True
            )
        
        # 交互逻辑函数
        def update_algorithm_input(algorithm_option):
            """
            根据算法选择模式更新算法输入框的显示状态
            """
            if algorithm_option == "输入算法名":
                return gr.Textbox(visible=True)
            else:
                return gr.Textbox(visible=False)
        
        def update_algorithm_info(algorithm_option, algorithm_name, dataset_info=None):
            """
            根据选择模式更新算法信息显示
            """
            if algorithm_option == "大模型自动推荐算法":
                return gr.Textbox(
                    value="🤖 等待LLM分析数据集并推荐算法...",
                    visible=True
                )
            elif algorithm_option == "输入算法名" and algorithm_name:
                return gr.Textbox(
                    value=f"📝 用户指定算法: {algorithm_name}",
                    visible=True
                )
            else:
                return gr.Textbox(
                    value="⚠️ 请选择算法模式或输入算法名称",
                    visible=True
                )
        
        # 算法模式选择时的实时更新
        algorithm_option.change(
            fn=update_algorithm_input,
            inputs=[algorithm_option],
            outputs=[algo_input]
        )
        
        algorithm_option.change(
            fn=update_algorithm_info,
            inputs=[algorithm_option, algo_input],
            outputs=[algorithm_info]
        )
        
        algo_input.change(
            fn=update_algorithm_info,
            inputs=[algorithm_option, algo_input],
            outputs=[algorithm_info]
        )
        
        # 修改 ui_generate_and_setup 函数（需要相应调整）
        def ui_generate_and_setup(file_path: str, algorithm_option: str, algorithm_name: str):
            """
            一键运行：根据算法选项决定是使用手动输入的算法名还是LLM推荐的算法名
            """
            if not file_path:
                return (
                    "# 请先上传数据集文件。",
                    "", 
                    "❌ 未上传数据集。",
                    "⚠️ 请先上传数据集"
                )
            
            # 如果是手动模式但未输入算法名
            if algorithm_option == "输入算法名" and (not algorithm_name or not algorithm_name.strip()):
                return (
                    "# 请输入算法名称",
                    "",
                    "❌ 请输入算法名称",
                    "⚠️ 请输入算法名称"
                )
            
            # 准备请求数据
            files = {"file": open(file_path, "rb")}
            data = {
                "algorithm_option": "manual" if algorithm_option == "输入算法名" else "auto",
                "algorithm_name": algorithm_name if algorithm_option == "输入算法名" else ""
            }
            
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/generate_code",
                    data=data,
                    files=files,
                    timeout=600
                )
            finally:
                files["file"].close()

            if resp.status_code != 200:
                return (
                    "# 生成代码失败\n" + resp.text,
                    "", 
                    "❌ 生成代码失败",
                    "⚠️ 后端返回错误"
                )

            j = resp.json()
            dataset_id = j.get("dataset_id")
            code = j.get("generated_code", "")
            env_file = j.get("env_file", "")
            algorithm_used = j.get("algorithm_used", "未知")
            algorithm_recommended = j.get("algorithm_recommended", False)
            
            if not dataset_id:
                return (
                    "# 后端未返回 dataset_id。",
                    "",
                    "❌ 后端未返回 dataset_id",
                    "⚠️ 后端错误"
                )

            STATE["dataset_id"] = dataset_id
            STATE["algorithm_name"] = algorithm_used
            STATE["last_train_log"] = ""

            # 构建算法信息
            if algorithm_recommended:
                algorithm_info_text = f"🤖 LLM推荐算法: {algorithm_used}"
            else:
                algorithm_info_text = f"📝 用户指定算法: {algorithm_used}"
            
            # ---- ② setup env ----
            if not code or not code.strip():
                return (
                    "# 后端未生成代码。",
                    "",
                    "❌ generated_code 为空",
                    algorithm_info_text
                )

            payload = {
                "dataset_id": dataset_id,
                "code": code,
                "algorithm_name": algorithm_used,
            }

            try:
                resp2 = _post_json(f"{BACKEND_URL}/setup_remote_env", payload, timeout=1800)
            except Exception as e:
                return (
                    code,
                    "",
                    f"❌ /setup_remote_env 请求失败：{e!r}",
                    algorithm_info_text
                )

            if resp2.status_code != 200:
                return (
                    code,
                    "",
                    f"❌ 远程环境配置失败：{resp2.status_code}\n{resp2.text}",
                    algorithm_info_text
                )

            j2 = resp2.json()
            env_log = j2.get("result", "后端未返回环境配置日志。")
            
            # 更新状态
            status_msg = "✅ 代码生成完成 | "
            status_msg += "🎯 环境配置完成" if "已配置完成" in env_log else "⚠️ 环境配置中"
            
            return code, env_log, status_msg, algorithm_info_text
        
        # 按钮点击事件绑定
        oneclick_btn.click(
            fn=ui_generate_and_setup,
            inputs=[file_input, algorithm_option, algo_input],
            outputs=[code_box, env_log_box, status_box, algorithm_info]
        )

        # 手动训练按钮（保持原逻辑）
        run_btn.click(
            fn=ui_run_remote_train,
            inputs=[code_box],
            outputs=[code_box, train_log_box, status_box]
        )

        # 手动修复按钮（保持原逻辑）
        manual_fix_btn.click(
            fn=ui_manual_fix_code,
            inputs=[code_box, error_input],
            outputs=[code_box, status_box]
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
