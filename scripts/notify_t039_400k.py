#!/usr/bin/env python3
"""Watcher script that monitors the 400k benchmark and sends a Telegram notification upon completion."""

import os
import sys
import time
import json
import subprocess

REPO_ROOT = "/home/lantian/ASCAD"
OUT_DIR = os.path.join(REPO_ROOT, "experiments/out/t039_400k")
DONE_FILE = os.path.join(OUT_DIR, "DONE")
LOG_FILE = os.path.join(OUT_DIR, "pipeline.log")
TELEGRAM_SCRIPT = os.path.join(REPO_ROOT, "scripts/send_telegram.py")


def send_tg(text):
    subprocess.run([sys.executable, TELEGRAM_SCRIPT, text], cwd=REPO_ROOT)


def format_eval_report():
    lines = ["🎉 [ASCAD] 400k 终极大标杆实验 (Compact 15.65M + Decoder Loss) 训练评测完成！\n"]
    lines.append("📊 核心配置：")
    lines.append("• 训练规模：400,000 轨迹 (100% 对齐 MS23 官方论文)")
    lines.append("• 模型架构：CompactMultiResNet Variant B (15.65M 参数，压缩 88.6%)")
    lines.append("• 损失函数：多任务交叉熵 (wC=0.5) + 微分解码器损失 (weight=1.0)")
    lines.append("• 迭代轮数：15 Epochs, RAdam lr 1e-4\n")

    for s in [1, 2]:
        eval_path = os.path.join(OUT_DIR, f"t400k_dec_s{s}_best_eval.json")
        if not os.path.exists(eval_path):
            continue
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            dec = d.get("decoders", {}).get("as_trained", {}).get("summary", {})
            beta_top1 = d.get("heads", {}).get("beta", {}).get("top1_acc", 0.0)
            pi_acc = d.get("pi_element_acc", 0.0)
            
            n_r0 = dec.get("n_rank0_and_not_flat", 0)
            med_tr = dec.get("median_traces_to_rank0", "N/A")
            max_tr = dec.get("max_traces_to_rank0", "N/A")
            med_z = dec.get("median_z", 0.0)

            lines.append(f"🏆 【Seed {s} 评测结果】:")
            lines.append(f"• 攻破字节数：{n_r0} / 16 字节 Rank 0")
            lines.append(f"• 攻破所需轨迹中位数：{med_tr} 条 (最大 {max_tr} 条)")
            lines.append(f"• 密钥区分度 (z-score)：+{med_z:.2f}σ")
            lines.append(f"• 关键头准确率：β Top-1 = {beta_top1*100:.2f}%, π 槽位 = {pi_acc*100:.2f}%\n")
        except Exception as e:
            lines.append(f"⚠️ 解析 Seed {s} 评测文件失败: {e}\n")

    lines.append("📌 官方 Baseline 对标 (MS23 论文记录):")
    lines.append("• 官方 137.4M MultiResNet (400k 训练): 全密钥攻破需 ~60 条轨迹")
    lines.append("• 本次 15.65M CompactMultiResNet (400k 训练): 详见上方实测轨迹数！")
    return "\n".join(lines)


def main():
    print("[Watcher] Started monitoring 400k benchmark...")
    while True:
        if os.path.exists(DONE_FILE):
            print("[Watcher] DONE flag detected! Formatting report...")
            report = format_eval_report()
            print(report)
            send_tg(report)
            print("[Watcher] Telegram notification sent. Exiting.")
            break
        
        # Check if the service failed without creating DONE
        res = subprocess.run(["systemctl", "--user", "is-active", "t039-400k-bench.service"],
                             capture_output=True, text=True)
        status = res.stdout.strip()
        if status in ("failed", "inactive") and not os.path.exists(DONE_FILE):
            # Check if it just finished or actually failed
            time.sleep(5)
            if not os.path.exists(DONE_FILE):
                print(f"[Watcher] Service ended with status {status} but no DONE file found!")
                send_tg(f"⚠️ [ASCAD] 400k 训练服务异常退出 (status: {status})，请检查日志！")
                break
        
        time.sleep(30)


if __name__ == "__main__":
    main()
