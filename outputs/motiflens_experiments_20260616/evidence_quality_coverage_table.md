# MotifLens Evidence Quality and Coverage

## Evidence Coverage and Basic Quality

| Dataset | Nodes | Edges | Rule rows | Rule edge cov. | Rule score mean | Rule reason missing | LLM rows | LLM edge cov. | LLM support/counter | LLM score mean | LLM invalid score | Response JSON-like | Responses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Meta | 348984 | 1256508 | 321025 | 25.55% | 0.356 | 27.95% | 11179 | 0.89% | 4208/6971 | 0.726 | 0.00% | 100.00% | 2763 |
| ZipZap | 17447516 | 56160047 | 18771985 | 33.43% | 0.341 | 25.22% | 47460 | 0.08% | 20338/27122 | 0.728 | 0.00% | 100.00% | 14985 |
| Illicit-ETH | 2155979 | 11582033 | 2221711 | 19.18% | 0.378 | 20.72% | 13450 | 0.12% | 6390/7060 | 0.737 | 0.00% | 100.00% | 4671 |
| EPSD-Ponzi | 3317131 | 19751913 | 6118552 | 30.98% | 0.338 | 40.49% | 13119 | 0.07% | 6137/6982 | 0.760 | 0.00% | 100.00% | 4361 |

## LLM Motif Distribution

| Dataset | rapid_fanout_after_inflow | mule_collector | scatter_gather_laundering | pass_through_relay | peeling_chain_like | victim_drain_then_cashout | service_like_false_positive |
|---|---|---|---|---|---|---|---|
| Meta | 1604 (14.35%) | 587 (5.25%) | 70 (0.63%) | 390 (3.49%) | 1529 (13.68%) | 28 (0.25%) | 6971 (62.36%) |
| ZipZap | 8927 (18.81%) | 2597 (5.47%) | 2256 (4.75%) | 4473 (9.42%) | 2081 (4.38%) | 4 (0.01%) | 27122 (57.15%) |
| Illicit-ETH | 2868 (21.32%) | 1478 (10.99%) | 149 (1.11%) | 825 (6.13%) | 1049 (7.80%) | 21 (0.16%) | 7060 (52.49%) |
| EPSD-Ponzi | 3183 (24.26%) | 732 (5.58%) | 309 (2.36%) | 1128 (8.60%) | 785 (5.98%) | 0 (0.00%) | 6982 (53.22%) |

## Notes

- Rule evidence is deterministic and high coverage; high coverage can also introduce noisy or broad behavioral cues.

- LLM motif evidence is sparse but typed into support/counter motifs, which is the core MotifLens evidence signal.

- `service_like_false_positive` is counter-evidence and is useful for interpretable false-positive suppression.
