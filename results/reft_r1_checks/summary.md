# ReFT-r1 side checks

These checks are not main results. They use 550 BBQ test questions (25 per
category and context condition), so do not tune anything on them.
Strength is a multiple of `max_act`. Reproduce with `steering/reft_r1/checks.py`.

| Run | Invalid | Acc. ambig. | Acc. disambig. | Bias ambig. | Bias disambig. | Unknown ambig. | Unknown disambig. | A / B / C |
|---|---|---|---|---|---|---|---|---|
| pilot-550 | 0.000 | 0.135 | 0.531 | 0.029 | 0.093 | 0.135 | 0.062 | 106 / 220 / 224 |
| pilot-550-s0 | 0.000 | 0.135 | 0.531 | 0.029 | 0.093 | 0.135 | 0.062 | 106 / 220 / 224 |
| pilot-550-s05 | 0.002 | 0.277 | 0.469 | 0.066 | 0.075 | 0.277 | 0.175 | 79 / 372 / 98 |
| pilot-550-s1 | 0.278 | 0.348 | 0.388 | 0.065 | 0.179 | 0.348 | 0.316 | 57 / 322 / 18 |
| pilot-550-s2 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | 0 / 0 / 0 |
| probe-550-neg05 | 0.005 | 0.151 | 0.480 | -0.026 | 0.098 | 0.151 | 0.113 | 123 / 71 / 353 |
| probe-550-neg1 | 0.000 | 0.251 | 0.353 | -0.029 | 0.124 | 0.251 | 0.269 | 135 / 27 / 388 |
| probe-550-neg2 | 0.865 | 0.262 | 0.375 | -0.119 | -0.083 | 0.262 | 0.250 | 74 / 0 / 0 |
| probe-550-random05 | 0.002 | 0.124 | 0.531 | 0.066 | 0.159 | 0.124 | 0.084 | 125 / 174 / 250 |
| probe-550-random1 | 0.011 | 0.196 | 0.391 | 0.019 | 0.243 | 0.196 | 0.248 | 81 / 128 / 335 |

Development examples, mean log-probability per response token:

| Strength | Unbiased | Biased | Prefers unbiased (of 65) |
|---|---|---|---|
| -1.0 | -9.248 | -8.815 | 23 |
| -0.5 | -8.152 | -7.921 | 27 |
| 0.0 | -6.034 | -5.936 | 29 |
| 0.5 | -2.841 | -3.067 | 46 |
| 1.0 | -2.758 | -3.097 | 55 |
