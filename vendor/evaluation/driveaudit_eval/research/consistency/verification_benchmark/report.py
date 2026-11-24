from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(metrics: dict[str, Any], output: str | Path) -> None:
    discovery, background, rule = metrics["discovery"], metrics["background_compensation"], metrics["tcvv_rule_discrimination"]
    holdout, leaderboard = metrics["current_holdout"], metrics["leaderboard"]
    neg, severity, human = metrics["negative_pairs"], metrics["severity_monotonicity"], metrics["human_validation"]
    lines = ["# TCVV-Bench v1 validation report", "", "## 1. 实现状态", "",
             f"**{metrics['implementation_status']}**：代码和单场景合成验证已完成；严格跨场景负样本、独立标定与排行榜冻结被真实数据规模阻断。", "",
             "## 2. TCVV-Bench阶段状态", "", f"`{leaderboard['status']}`；总分={leaderboard['enable_total_score']}，版本=`{leaderboard['score_version']}`。", "",
             "## 3. 数据发现", "", f"可用/有效/无效场景：{discovery['available_history_scene_count']} / {discovery['strictly_pairable_scene_count']} / {len(discovery['invalid_scenes'])}",
             f"历史正样本窗口：{metrics['positive_pairs']['history_positive_window_count']}；paired future GT：0。",
             f"轨迹负样本：{neg['trajectory_negative_family_count']}类、{neg['trajectory_negative_sample_count']}窗口；图像负样本：{neg['image_negative_family_count']}类、{neg['image_negative_sample_count']}窗口。",
             f"跨场景负样本：{neg['cross_scene_status']}。", "",
             "## 4. 数据划分", "", f"{metrics['splits']}。`current_sample_001`固定为test/holdout，未进入阈值、融合或TCVV-C训练。", "",
             "## 5. 背景补偿结果", "", f"Raw Flow EPE={background['raw_flow_epe']:.6f}px；Residual Flow EPE={background['residual_flow_epe']:.6f}px。",
             f"Raw Cosine={background['raw_flow_cosine']:.6f}；Residual Cosine={background['residual_flow_cosine']:.6f}；Background Leakage={background['background_leakage']:.6f}。",
             "EPE下降但方向余弦未改善；背景环与目标深度不匹配造成的视差是已观察到的失败模式。", "",
             "## 6. TCVV-R判别结果", "", f"AUROC={rule.get('auroc'):.6f}；AUPRC={rule.get('auprc'):.6f}；Balanced Accuracy={rule.get('balanced_accuracy'):.6f}。",
             f"总体FPR={rule.get('false_positive_rate'):.6f}；Hard-negative FPR={rule.get('hard_negative_false_positive_rate'):.6f}；ECE={rule.get('expected_calibration_error'):.6f}。",
             "这些是同一holdout场景上的合成负样本诊断，不是独立多场景泛化成绩。", "",
             "## 7. TCVV-C判别结果", "", f"`{metrics['tcvv_calibrated']['status']}`；独立calibration scene为0，因此没有拟合Logistic Regression，也没有读取holdout。", "",
             "## 8. Corruption响应", "", f"35类均有结果；严格严重度单调通过率={severity['monotonic_pass_rate']:.6f}。",
             "身份交换、消失/进入相机等非局部运动错误表现出低敏感性；完整逐类结果见statistics和per_negative_family.csv。", "",
             "## 9. 多场景统计", "", "场景数=1。scene-level bootstrap已真实执行2000次，但单场景CI退化为点值，不能解释为跨场景稳定性。", "",
             "## 10. Current Holdout", "", f"Existing TCVV support={holdout['existing_tcvv']['projection_support_rate']:.6f}；TCVV-R support={holdout['tcvv_rule']['projection_support_rate']:.6f}。",
             f"净False-support reduction={holdout['tcvv_rule']['false_support_reduction']:.6f}；原始支持中被拒绝窗口={holdout['tcvv_rule']['raw_supported_rejected_count']}。",
             f"Future Entrant raw/rule support={holdout['future_entrant']['raw_conditional_support']}/{holdout['future_entrant']['rule_support']}；规则升高提示仍有ROI假支持风险。", "",
             "## 11. 人工验证", "", f"导出短片={human.get('exported_sample_count', 0)}；标注者=0；状态=`{human['status']}`；没有伪造相关性。", "",
             "## 12. 排行榜维度", "", "Presence、Motion、Temporal、Multi-view保持分维度输出；总分关闭。分数映射未拟合，不能称为tcvv_score_v1。", "",
             "## 13. 独立诊断", "", "MTPC=`long_horizon_tracking_diagnostic`；BEVFormer=`perception_utility_diagnostic`。二者均不进入TCVV-R或排行榜维度。", "",
             "## 14. 当前限制", "", "单场景、400×224、背景环深度近似、合成负样本、人工标注待完成、阈值与排行榜均为provisional。", "",
             "## 15. 输出文件", "", "核心：`final_metrics.json`、`calibrated_verifier/`、`residual_motion/`、`statistics/`、`human_audit/`、`leaderboard/`。"]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
