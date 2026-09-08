# 五模型预算缩减 v1

预算在部分数据已经采集后调整，不是从实验开始预注册的小样本方案。
标准390 × 五模型全部保留。新的robustness共同清单为1960 × 五模型：
域内极限1192、OOD探索128、六类扰动640。旧额外数据仅作扩展数据。

`g1_reduced_budget.py prepare`先从旧manifest确定性选择共同清单，不读outcome或完成状态。
层内按指定SHA256排序；固定脚/相位配额循环处理奇数配额。随机波形条件只按hash选20条。
`reduced_budget_selection_audit.json`保存每层实际覆盖；不改变任何原trial字段。

`g1_reduced_sources.py`为已修复wrench的旧cohort执行LIGHT来源校验，支持独立引用
PARTIAL中的合格记录。完整身份、record/trace/event SHA及合法失败都保留在复用索引中。
其他INVALID或performance数据不自动进入主对比。只将PENDING_EXECUTION编译为独立run。

批大小保持64。末批未使用的槽位不施扰、不推进裁判、不产生trial结果。
新reset在实际执行前必须与旧paired_initial_states完全一致。
`STOP_AFTER_BATCH`请求只在当前批全部结果保存后生效，保持原manifest与PARTIAL进度。

代码兼容证明逐文件检查：除明确的预算身份、队列、末批槽位和安全暂停代码外，
旧运行时原文必须完全相同。物理step循环、native模型、wrench、detector、threshold/W/H不变。
新旧runtime hash均如实保留。此证明不声称GPU轨迹可逐位重现。

每个新run仍用LIGHT+SAMPLED封存。全部物理任务完成后，仅对主对比来源执行一次FULL，
不为旧额外数据重复全量回放。最终中文MD/DOCX使用共同子集；标准390独立一节。
OOD ±20独立附录。报告保留指定/实际施扰分母、sham、不确定性和单训练seed限制。
