# Multi-architecture ensemble macro-F1

Ensembling architectures: `gru`, `lstm`, `lstm_attn`, `tcn`, `transformer`

| Strategy | n_pooled | macro-F1 | 95% CI |
| --- | ---: | ---: | --- |
| mean_logit | 2205 | 0.4922 | [0.4595, 0.5235] |
| mean_softmax | 2205 | 0.4832 | [0.4490, 0.5158] |
| plurality_vote | 2205 | 0.4724 | [0.4405, 0.5032] |
