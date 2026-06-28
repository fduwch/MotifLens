# MotifLens Interpretability Case Study

This file contains paper-ready evidence cases for MotifLens. The examples focus on observed transaction motifs, support/counter evidence, and why the evidence is interpretable.

## Case 1: Existing Model Explanation Artifact

- Source: `outputs/paper_tables_clean_20260615/interpretable_fraud_example_llm_lift_illicit_seed42.json`

- dataset: Illicit-ETH

- seed: 42

- method: Ours = GraphSAGE + rule evidence + Qwen motif evidence

- address: 0x24aa18952e80707dad3ab4c9c97e787f2af337ca

- node_idx: 317242

- true_label: 1

- elapsed_seconds: 30.69

```json
{
  "Base GraphSAGE": {
    "probability": 0.7114150524139404,
    "threshold": 0.51,
    "predicted_fraud": true
  },
  "GraphSAGE + rule": {
    "probability": 0.8570509552955627,
    "threshold": 0.5700000000000001,
    "predicted_fraud": true
  },
  "Ours rule+Qwen": {
    "probability": 0.9294595122337341,
    "threshold": 0.51,
    "predicted_fraud": true
  }
}
```

## Case 2: Existing Model Explanation Artifact

- Source: `outputs/paper_tables_clean_20260615/interpretable_fraud_example_illicit_seed42.json`

- dataset: Illicit-ETH

- seed: 42

- method: Ours = GraphSAGE + rule evidence + Qwen motif evidence

- address: 0x33f74739ce6be3b2c38af76f5adb3866cb4784c4

- node_idx: 446041

- true_label: 1

- predicted_label: 1

- fraud_probability: 1.0

- decision_threshold: 0.51

- base_graphsage_probability: 0.9999974966049194

- elapsed_seconds: 24.0

```json
{
  "label": 1,
  "split": "test",
  "is_target": 1,
  "label_source": "illicit_eth_complete",
  "Sent_tnx": 221.0,
  "Received_Tnx": 101.0,
  "Unique_Received_From_Addresses": 99.0,
  "Unique_Sent_To_Addresses": 74.0,
  "total_transactions_(including_tnx_to_create_contract)": 322.0,
  "total_Ether_sent": 639.0573675,
  "total_ether_received": 501.5937353,
  "total_ether_balance": -137.4636322,
  "Total_ERC20_tnxs": 171.0
}
```

## Support motif: rapid fanout after inflow

| address | target_idx | edge_pos | evidence_type | polarity | evidence_score | reason_codes | observation_type | observation_score |
|---|---|---|---|---|---|---|---|---|
| 0x03b70dc31abf9cf6c1cf80bfeeb322e8d3dbb4ca | 39098 | 776 | rapid_fanout_after_inflow | support | 0.9 | outflow_shortly_after_inflow|receiver_has_low_degree|sender_high_out_degree | new_counterparty | 0.6444 |
| 0x03b70dc31abf9cf6c1cf80bfeeb322e8d3dbb4ca | 39098 | 777 | rapid_fanout_after_inflow | support | 0.9 | outflow_shortly_after_inflow|receiver_has_low_degree|sender_high_out_degree | new_counterparty | 0.6439 |
| 0x03b70dc31abf9cf6c1cf80bfeeb322e8d3dbb4ca | 39098 | 793 | rapid_fanout_after_inflow | support | 0.9 | outflow_shortly_after_inflow|receiver_has_low_degree|sender_high_out_degree | new_counterparty | 0.6611 |


## Support motif: mule collector

| address | target_idx | edge_pos | evidence_type | polarity | evidence_score | reason_codes | observation_type | observation_score |
|---|---|---|---|---|---|---|---|---|
| 0x04727c2bb15f2b83e7d3ce88b008aa68dccea0c4 | 45375 | 930 | mule_collector | support | 0.65 | receiver_has_low_degree | new_counterparty | 0.3415 |
| 0x046e3a705d9bcd0e53d2b45161b48e39f4bc4090 | 45223 | 932 | mule_collector | support | 0.75 | receiver_has_low_degree | high_value_transfer | 0.3595 |
| 0x058e49378461c239dae065114cd9fa1c0dbc25c1 | 54697 | 1048 | mule_collector | support | 0.8 | outflow_shortly_after_inflow|receiver_has_low_degree|sender_high_out_degree|receiver_consolidation_like|high_value_percentile | burst_activity | 0.4365 |


## Counter motif: service-like false positive

| address | target_idx | edge_pos | evidence_type | polarity | evidence_score | reason_codes | observation_type | observation_score |
|---|---|---|---|---|---|---|---|---|
| 0x0020731604c882cf7bf8c444be97d17b19ea4316 | 2128 | 1 | service_like_false_positive | counter | 0.55 | sender_high_out_degree|receiver_has_low_degree | burst_activity | 0.396 |
| 0x0020731604c882cf7bf8c444be97d17b19ea4316 | 2128 | 9 | service_like_false_positive | counter | 0.55 | sender_high_out_degree|receiver_has_low_degree | burst_activity | 0.3671 |
| 0x0020731604c882cf7bf8c444be97d17b19ea4316 | 2128 | 10 | service_like_false_positive | counter | 0.55 | sender_high_out_degree|receiver_has_low_degree | burst_activity | 0.3734 |


## Ponzi support motif: pass-through relay

| address | target_idx | edge_pos | evidence_type | polarity | evidence_score | reason_codes | observation_type | observation_score |
|---|---|---|---|---|---|---|---|---|
| 0x01680dc54cf0942bcabc1d6c955007e180ed4dd1 | 33630 | 5 | pass_through_relay | support | 0.85 | outflow_shortly_after_inflow|sender_high_out_degree|receiver_consolidation_like | burst_activity | 0.7442 |
| 0x01680dc54cf0942bcabc1d6c955007e180ed4dd1 | 33630 | 8 | pass_through_relay | support | 0.85 | outflow_shortly_after_inflow|sender_high_out_degree|receiver_consolidation_like | burst_activity | 0.7446 |
| 0x01680dc54cf0942bcabc1d6c955007e180ed4dd1 | 33630 | 9 | pass_through_relay | support | 0.85 | outflow_shortly_after_inflow|sender_high_out_degree|receiver_consolidation_like | burst_activity | 0.7446 |


## How to Use in the Paper

- Use one support case to show why MotifLens flags an address as fraudulent.

- Use one counter-evidence case to show that the LLM evidence is not merely adding positive fraud keywords; it can suppress service-like false positives.

- Report evidence as structured motif type, polarity, score, and grounded transaction observation rather than free-form explanation text.
