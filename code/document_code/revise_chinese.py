from pathlib import Path
import re,json,difflib
W=Path(__file__).resolve().parents[1];B=W.parent;T=W/'work/text'
z=(B/'manuscript_revision/MANUSCRIPT_ZH.md').read_text();orig=z
old='术中信息能够更新术前风险，但其增益随研究人群、结局定义和评价方式而变化。既有研究比较了连续信息阶段、高风险人群识别及术中变量加入后的模型表现。[5–7] 在胃肠道肿瘤手术中，需要进一步将信息更新与同一批患者的固定容量复核相连接，以衡量术毕评估所增加的事件识别。'
new='Lei等已按最高风险20%的记录比较围术期阶段，并以10%和30%进行敏感性分析。[5] Kim和Do等也评价了术中信息加入后的模型表现。[6,7] 本研究在胃肠道肿瘤手术中估计同记录、同容量下减少输入字段的捕获代价，并结合术毕更新和阈值迁移后的名单规模与事件覆盖进行评价。'
assert old in z;z=z.replace(old,new)
old='胃肠肿瘤手术后的急性肾损伤（AKI）风险评估涉及信息获取与复核资源的共同配置。本研究中，九项术前输入与27项输入的观察区分度接近，术毕更新的收益随评价场景和复核容量而变化。在各评价样本内比较事件捕获与名单规模，可以量化增加字段、术毕更新及扩大名单所对应的事件捕获变化。'
new='20%复核容量下，术前输入由27项减为九项，地理和时间评价分别少捕获2个和3个事件，配对区间均包含增加和减少的可能。术毕更新分别多捕获15个和1个事件。九输入名单由10%扩大至20%，则分别增加捕获8个和9个事件。这些比较将信息需求和评估时点与名单规模、事件捕获相连接。'
assert old in z;z=z.replace(old,new)
z=z.replace('Lei等评价了高风险人群的事件捕获','Lei等评价了固定最高风险比例内的事件捕获')
z=z.replace('完整阶段方案包含模型家族、开发和校准差异，各组成的贡献尚未单独识别。','阶段对比估计完整拟合系统的差异，其中包括各自的模型家族、开发程序与校准。')
old='九项常规术前输入为AKI风险评估提供字段更少的选择，其捕获代价可与本地采集负担及有依据的损失容忍条件共同衡量。术毕更新、容量扩展和阈值迁移的配置应结合事件覆盖与可承接的名单规模。'
new='20%容量下，术前输入由27项减至九项，观察捕获相差2至3个事件，区间同时包含增加和减少的可能。术毕增益在两条评价线中表现不同。结合配对捕获差、名单规模和阈值迁移后的实际覆盖，可以为本地信息采集与复核容量配置提供量化依据。'
assert old in z;z=z.replace(old,new)
(T/'MANUSCRIPT_ZH_scientific.md').write_text(z)
# Humanizer pass keeps the technical parallel comparisons and numerical ranges.
z=z.replace(' 因而，模型评价应同时报告',' 模型评价应同时报告')
(T/'MANUSCRIPT_ZH_humanizer.md').write_text(z)
(T/'MANUSCRIPT_ZH.md').write_text(z)
(W/'review/CHINESE_REVISIONS.diff').write_text(''.join(difflib.unified_diff(orig.splitlines(True),z.splitlines(True),fromfile='received_chinese',tofile='final_chinese')))
print('Chinese author review synchronized')
