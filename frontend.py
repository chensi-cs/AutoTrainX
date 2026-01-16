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
    with gr.Blocks(title="LLM 生成训练代码 + AutoDL 远程训练") as demo:
        gr.Markdown(
            "## 使用说明\n"
            "1) 上传数据集 + 输入算法名 → 点击 **①+② 一键生成并配置环境**\n"
            "2) 你可以在代码框里改 train.py\n"
            "3) 点击 **③ 手动开始训练**\n"
            "   - 若训练报错：前端会自动调用 /fix_code 修复代码并回填（**不自动重跑**）\n"
            "   - 修复后你再手动点一次训练即可\n"
        )

        with gr.Row():
            file_input = gr.File(
                label="上传数据集文件（csv / zip / rar / data / arff）",
                file_types=[".csv", ".zip", ".rar", ".data", ".arff"],
                type="filepath",
            )
            algo_input = gr.Textbox(
                label="算法名称（例如：XGBRegressor, RandomForestClassifier 等）",
                placeholder="例如：XGBRegressor"
            )

        oneclick_btn = gr.Button("①+② 一键生成并配置环境（/generate_code + /setup_remote_env）")
        run_btn = gr.Button("③ 手动开始训练（/run_remote_train；报错会自动修复但不重跑）")

        with gr.Row():
            manual_fix_btn = gr.Button("（可选）手动修复代码（/fix_code）")

        status_box = gr.Textbox(label="状态", lines=2)

        code_box = gr.Code(
            label="train.py（可编辑；报错时会自动回填修复版）",
            language="python",
            lines=28
        )

        env_log_box = gr.Textbox(
            label="环境配置日志（AutoDL .venv）",
            lines=14
        )

        train_log_box = gr.Textbox(
            label="训练日志（stdout/stderr）",
            lines=18
        )

        error_input = gr.Textbox(
            label="（可选）粘贴报错信息（Traceback/stderr）；留空则用最近一次训练日志",
            lines=8,
            placeholder="Traceback (most recent call last): ..."
        )

        # 一键：生成 + 配置环境
        oneclick_btn.click(
            fn=ui_generate_and_setup,
            inputs=[file_input, algo_input],
            outputs=[code_box, env_log_box, status_box]
        )

        # 手动训练（但自动修复不重跑）
        run_btn.click(
            fn=ui_run_remote_train,
            inputs=[code_box],
            outputs=[code_box, train_log_box, status_box]
        )

        # 可选：手动修复
        manual_fix_btn.click(
            fn=ui_manual_fix_code,
            inputs=[code_box, error_input],
            outputs=[code_box, status_box]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
