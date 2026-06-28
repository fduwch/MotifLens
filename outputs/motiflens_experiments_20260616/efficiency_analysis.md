# MotifLens Efficiency Analysis

## LLM Evidence Generation
| Dataset | Cards | Evidence edges | Elapsed hours | Cards/min | Evidence edges/min | Cards JSONL MB | Responses JSONL MB | Evidence CSV MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meta | 2764 | 11179 | 1.713 | 26.899 | 108.794 | 7.91 | 27.97 | 2.16 |
| ZipZap | 14985 | 47460 | 9.446 | 26.441 | 83.743 | 40.18 | 146.17 | 9.38 |
| Illicit-ETH | 4670 | 13450 | 2.882 | 27.005 | 77.777 | 11.48 | 47.61 | 2.65 |
| EPSD-Ponzi | 4360 | 13119 | 4.680 | 15.528 | 46.722 | 11.42 | 45.21 | 2.75 |

## Training And Feature Overhead
| Dataset | Done | Nodes | Edges | Base feat. | Evidence feat. | Final feat. | Feature overhead | Epochs mean | Best epoch mean | Batch size | Train nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meta | 5/5 | 348984 | 1256508 | 31 | 90 | 121 | +290.3% | 90.800 | 51.600 | 4096 | 1934 |
| ZipZap | 5/5 | 17447516 | 56160047 | 31 | 90 | 121 | +290.3% | 99.200 | 65.600 | 4096 | 10489 |
| Illicit-ETH | 5/5 | 2155979 | 11582033 | 63 | 90 | 153 | +142.9% | 81.000 | 41.000 | 4096 | 3269 |
| EPSD-Ponzi | 5/5 | 3317131 | 19751913 | 44 | 90 | 134 | +204.5% | 114.400 | 87.000 | 4096 | 3052 |
