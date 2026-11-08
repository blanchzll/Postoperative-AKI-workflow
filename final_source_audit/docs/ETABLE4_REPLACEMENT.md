## eTable 4. Stage-system and same-family performance and event-capture comparisons

| Evaluation | Comparison | Metric | Capacity | Unit | Difference | Lower 95% CI | Upper 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geographic paired | Stage systems | AUROC | — | AUROC difference | 0.129 | 0.065 | 0.195 |
| Geographic paired | Stage systems | Brier score | — | Probability-scale difference | -0.004 | -0.006 | -0.002 |
| Geographic paired | Stage systems | Captured events | 10% | Events per 1,000 operations | 11.226 | 4.307 | 18.198 |
| Geographic paired | Stage systems | Captured events | 20% | Events per 1,000 operations | 12.953 | 5.172 | 21.552 |
| Geographic paired | Same family: LR | AUROC | — | AUROC difference | 0.055 | -0.003 | 0.113 |
| Geographic paired | Same family: LR | Brier score | — | Probability-scale difference | -0.001 | -0.002 | 0.001 |
| Geographic paired | Same family: LR | Captured events | 10% | Events per 1,000 operations | 7.772 | 0.864 | 14.693 |
| Geographic paired | Same family: LR | Captured events | 20% | Events per 1,000 operations | 0.864 | -5.199 | 9.499 |
| Geographic paired | Same family: Voting | AUROC | — | AUROC difference | 0.061 | 0.016 | 0.113 |
| Geographic paired | Same family: Voting | Brier score | — | Probability-scale difference | -0.007 | -0.009 | -0.005 |
| Geographic paired | Same family: Voting | Captured events | 10% | Events per 1,000 operations | 0.864 | -4.325 | 7.765 |
| Geographic paired | Same family: Voting | Captured events | 20% | Events per 1,000 operations | 4.318 | -1.727 | 11.255 |
| Temporal paired | Stage systems | AUROC | — | AUROC difference | 0.030 | -0.074 | 0.133 |
| Temporal paired | Stage systems | Brier score | — | Probability-scale difference | 0.000 | -0.003 | 0.003 |
| Temporal paired | Stage systems | Captured events | 10% | Events per 1,000 operations | 0.000 | -5.946 | 7.913 |
| Temporal paired | Stage systems | Captured events | 20% | Events per 1,000 operations | 0.989 | -5.935 | 6.938 |
| Temporal paired | Same family: Voting | AUROC | — | AUROC difference | -0.022 | -0.109 | 0.072 |
| Temporal paired | Same family: Voting | Brier score | — | Probability-scale difference | 0.002 | -0.001 | 0.005 |
| Temporal paired | Same family: Voting | Captured events | 10% | Events per 1,000 operations | 1.978 | -0.989 | 6.931 |
| Temporal paired | Same family: Voting | Captured events | 20% | Events per 1,000 operations | -1.978 | -7.913 | 1.982 |
| Temporal paired | Same family: SGBT | AUROC | — | AUROC difference | 0.073 | -0.007 | 0.156 |
| Temporal paired | Same family: SGBT | Brier score | — | Probability-scale difference | -0.000 | -0.003 | 0.003 |
| Temporal paired | Same family: SGBT | Captured events | 10% | Events per 1,000 operations | 1.978 | -3.953 | 9.881 |
| Temporal paired | Same family: SGBT | Captured events | 20% | Events per 1,000 operations | 2.967 | -2.967 | 8.911 |
| Temporal paired | No overlap: stage systems | AUROC | — | AUROC difference | -0.013 | -0.128 | 0.107 |
| Temporal paired | No overlap: stage systems | Brier score | — | Probability-scale difference | 0.001 | -0.002 | 0.004 |
| Temporal paired | No overlap: stage systems | Captured events | 10% | Events per 1,000 operations | -5.420 | -12.212 | 2.714 |
| Temporal paired | No overlap: stage systems | Captured events | 20% | Events per 1,000 operations | -4.065 | -10.855 | 4.065 |

Differences are end-of-operation minus preoperative values. AUROC and Brier score differences are dimensionless; positive AUROC and negative Brier differences favor the end-of-operation system. Captured-event differences are events per 1,000 operative records at the indicated 10% or 20% batch review capacity. These are separate performance estimands. Intervals are percentile 95% confidence intervals from 2,000 hospital-stratified patient-group bootstrap samples. Geographic paired evaluation contains 1,158 records/72 events; temporal paired evaluation contains 1,011/36. The temporal non-overlap comparison excludes previously overlapping records (738 records/29 events).

Stage-system comparisons are Voting (preoperative) versus LR (end of operation) geographically, and SGBT versus Voting temporally. Same-family comparisons retain the named model family at both stages. LR denotes logistic regression; SGBT, stochastic gradient boosting. Source CSV model identifiers and source-defined precision are retained in ETABLE4_SOURCE_MAPPED.csv.
