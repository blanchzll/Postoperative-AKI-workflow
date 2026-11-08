# AKI临床补充分析交付

任务：AKI_CLINICAL_SUPPLEMENT_20261108_V1。仅分析和图源交接，旧稿和原20图不改。

先读：ACCEPTANCE.json、SCIENCE_CONCLUSION_MATRIX.csv、ANALYSIS_RESULTS_AND_REVIEW_ZH.md。

- QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv：七问实际回答及具体缺失输入。
- eMETHODS_FOR_WRITING_EN.md / eRESULTS_FOR_WRITING_EN.md：精简分析交接文本，父写作对话自行编号和整合。
- results/：完整聚合数值及不利/无方向结果。DAILY_POTENTIAL_DEMAND.csv含具体研究日期，仅内部分析，不进入交接压缩包或公开代码。
- figures/：3幅新增图、8个独立分面PDF和源CSV/code哈希。
- WORKBUDDY_ANALYSIS_DISPOSITIONS.csv：采纳、已解决、拒绝推断或转交写作的依据。
- audit/：来源、两种数值实现、科学整改、失败留档、运行/资源和公开医学定义。
- private/：患者级工作文件，不交公开仓库、不打包。

复算仅在t630，保留原冻结工作。共享统计环境解释器：/home/lei/codex-workspaces/shared-python-envs/envs/clinical-stats-py312-cpu-20261016-v1/bin/python。无需安装/升级包。

必要复算顺序：analyze_patient_policies.py → analyze_observation_outcomes.py（默认含MIMIC；--outcomes-only仅用于已有MIMIC输出后的硬结局续算）→ review_repairs.py → coverage_and_window_sensitivity.py → audit_reused_results.py → independent_verify.py → extra_verify.py → plot_supplement.py → build_handoff.py。遵守运行锁，先看现有验收，不为保持利用率重复计算。最终零事件中心精度规则见audit/ZERO_EVENT_CENTER_PRECISION_REPAIR.json及review_repairs.py。

七问的临床部署、真实提前量、实施因果获益、停监测安全均须服从实际可识别性；分析验收不代替这些临床资格。
