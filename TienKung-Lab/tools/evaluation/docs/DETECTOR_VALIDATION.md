# DETECTOR VALIDATION / 恢复检测器验收

状态：NOT_RUN

本轮按用户要求未启动实验。离线单元测试不等价于真实轨迹验收；协议保持 1.0-dev。

已有 baseline 推扰轨迹 0，最低需检查 20 条，且覆盖两个 baseline、轻/重扰动和失败。

| Baseline | E0 moving 完整间隔 | 达标间隔 | 达标比例 |
| --- | --- | --- | --- |

## 待验收事项

检查无扰动正常片段识别率、假恢复、长时间不恢复、接触抖动及重复同脚计步。检查 entry/confirmed 时刻，确认后跌倒不得算主成功。若正常片段大量不达标，停止冻结并报告证据，不自动放宽阈值。

人工验收记录写入 report/notes.yaml 的 detector_validation 项，逐条关联 evaluation_id / trial_id / trace SHA。冻结 gate 还必须记录并重新验证 source evaluation ID、subset、manifest hash、actual physics hash、protocol/metrics/physics/inference 身份及 evaluation runtime SHA；开发 subset 只接受 prepared full manifest 中 E0/E1 各自确定的前 N 条。

{
  "manifest_hash": "f35dea8b1267318cc09a667865dbea8b45bfc593033c1040ff1200e594070f98",
  "metrics_version": "practical_interval_confirm2_v1",
  "metrics_config_hash": "7150ef415a2651903d6182eca5c6a723204d110a69c79cb766cdd675b0527ae4",
  "metrics_reference_sha256": "0e1e68b0f2d91dd33698de3cb40e60c18461fbe34999a32c96e86adda288f73a",
  "physics_profile_hash": "d2d62e30507d9f2bc39ce2f3ccdbb48000f8eafd5e693db102eed8560e463122",
  "evaluation_runtime_sha256": null,
  "sources": [],
  "source_evaluation_ids": [],
  "source_subsets": {},
  "source_manifest_hashes": {},
  "actual_physics_hashes": {},
  "status": "NOT_RUN",
  "pushed_trajectories": [],
  "minimum_required": 20,
  "automatic_freeze": false
}
