# Measurement reliability (cfoot): 1.4b_s0_step143000

968 concepts, 224 contexts each, split into two fixed random halves of 112. Reliability = Spearman-Brown corrected split-half correlation across concepts (1 = the measure ranks concepts the same way on independent contexts; 0 = pure noise).

## Durability: context-sampling reliability (split-half)

| variant | mean d log p | split-half r | reliability |
|---|---|---|---|
| lora-code-10M | -0.067 | +0.822 | +0.902 |
| lora-code-1M | -0.025 | +0.892 | +0.943 |
| lora-code-20M | -0.068 | +0.790 | +0.883 |
| lora-code-5M | -0.063 | +0.789 | +0.882 |
| lora-code_s1-20M | -0.063 | +0.794 | +0.885 |
| lora-wiki_de-10M | -0.051 | +0.872 | +0.931 |
| lora-wiki_de-1M | -0.019 | +0.897 | +0.946 |
| lora-wiki_de-20M | -0.071 | +0.871 | +0.931 |
| lora-wiki_de-5M | -0.031 | +0.895 | +0.945 |
| lora-wiki_de_s1-20M | -0.064 | +0.872 | +0.932 |
| noise0.1s0 | -0.259 | +0.962 | +0.981 |
| noise0.1s1 | -0.264 | +0.964 | +0.982 |
| noise0.1s2 | -0.383 | +0.961 | +0.980 |
| prune0.25 | -0.464 | +0.916 | +0.956 |
| prune0.5 | -6.725 | +0.985 | +0.992 |
| base log p (sanity) | | +0.954 | +0.977 |

## Durability: is it the same trait across perturbations? (disjoint halves; replicate pairs first)

Correlation of variant 1 on half A with variant 2 on half B (and the swap, averaged). Shared base noise cannot inflate it. Divided by sqrt(rel1 x rel2) it is the correlation of the TRUE per-concept damage.

| variant 1 | variant 2 | r (disjoint) | disattenuated |
|---|---|---|---|
| lora-code-10M | lora-code-1M | +0.391 | +0.423 |
| lora-code-10M | lora-code-20M | +0.709 | +0.795 |
| lora-code-10M | lora-code-5M | +0.623 | +0.699 |
| lora-code-10M | lora-code_s1-20M | +0.652 | +0.729 |
| lora-code-1M | lora-code-20M | +0.412 | +0.452 |
| lora-code-1M | lora-code-5M | +0.371 | +0.407 |
| lora-code-1M | lora-code_s1-20M | +0.397 | +0.434 |
| lora-code-20M | lora-code-5M | +0.621 | +0.704 |
| lora-code-20M | lora-code_s1-20M | +0.728 | +0.824 |
| lora-code-5M | lora-code_s1-20M | +0.593 | +0.672 |
| lora-wiki_de-10M | lora-wiki_de-1M | +0.647 | +0.690 |
| lora-wiki_de-10M | lora-wiki_de-20M | +0.822 | +0.882 |
| lora-wiki_de-10M | lora-wiki_de-5M | +0.745 | +0.794 |
| lora-wiki_de-10M | lora-wiki_de_s1-20M | +0.792 | +0.850 |
| lora-wiki_de-1M | lora-wiki_de-20M | +0.655 | +0.698 |
| lora-wiki_de-1M | lora-wiki_de-5M | +0.687 | +0.726 |
| lora-wiki_de-1M | lora-wiki_de_s1-20M | +0.667 | +0.711 |
| lora-wiki_de-20M | lora-wiki_de-5M | +0.760 | +0.810 |
| lora-wiki_de-20M | lora-wiki_de_s1-20M | +0.831 | +0.892 |
| lora-wiki_de-5M | lora-wiki_de_s1-20M | +0.745 | +0.795 |
| noise0.1s0 | noise0.1s1 | +0.084 | +0.085 |
| noise0.1s0 | noise0.1s2 | +0.063 | +0.064 |
| noise0.1s1 | noise0.1s2 | +0.065 | +0.066 |
| lora-code-10M | lora-wiki_de-10M | +0.114 | +0.124 |
| lora-code-10M | lora-wiki_de-1M | +0.038 | +0.041 |
| lora-code-10M | lora-wiki_de-20M | +0.132 | +0.144 |
| lora-code-10M | lora-wiki_de-5M | +0.126 | +0.137 |
| lora-code-10M | lora-wiki_de_s1-20M | +0.123 | +0.134 |
| lora-code-10M | noise0.1s0 | +0.013 | +0.014 |
| lora-code-10M | noise0.1s1 | +0.083 | +0.088 |
| lora-code-10M | noise0.1s2 | +0.163 | +0.173 |
| lora-code-10M | prune0.25 | +0.191 | +0.205 |
| lora-code-10M | prune0.5 | +0.209 | +0.221 |
| lora-code-1M | lora-wiki_de-10M | +0.024 | +0.025 |
| lora-code-1M | lora-wiki_de-1M | -0.025 | -0.026 |
| lora-code-1M | lora-wiki_de-20M | +0.027 | +0.028 |
| lora-code-1M | lora-wiki_de-5M | +0.052 | +0.055 |
| lora-code-1M | lora-wiki_de_s1-20M | +0.012 | +0.013 |
| lora-code-1M | noise0.1s0 | -0.002 | -0.002 |
| lora-code-1M | noise0.1s1 | +0.002 | +0.002 |
| lora-code-1M | noise0.1s2 | +0.063 | +0.065 |
| lora-code-1M | prune0.25 | +0.123 | +0.129 |
| lora-code-1M | prune0.5 | +0.149 | +0.154 |
| lora-code-20M | lora-wiki_de-10M | +0.129 | +0.143 |
| lora-code-20M | lora-wiki_de-1M | +0.069 | +0.075 |
| lora-code-20M | lora-wiki_de-20M | +0.164 | +0.181 |
| lora-code-20M | lora-wiki_de-5M | +0.141 | +0.154 |
| lora-code-20M | lora-wiki_de_s1-20M | +0.158 | +0.175 |
| lora-code-20M | noise0.1s0 | -0.004 | -0.004 |
| lora-code-20M | noise0.1s1 | +0.060 | +0.064 |
| lora-code-20M | noise0.1s2 | +0.105 | +0.113 |
| lora-code-20M | prune0.25 | +0.172 | +0.187 |
| lora-code-20M | prune0.5 | +0.191 | +0.204 |
| lora-code-5M | lora-wiki_de-10M | +0.135 | +0.149 |
| lora-code-5M | lora-wiki_de-1M | +0.113 | +0.123 |
| lora-code-5M | lora-wiki_de-20M | +0.163 | +0.180 |
| lora-code-5M | lora-wiki_de-5M | +0.144 | +0.158 |
| lora-code-5M | lora-wiki_de_s1-20M | +0.162 | +0.179 |
| lora-code-5M | noise0.1s0 | -0.030 | -0.032 |
| lora-code-5M | noise0.1s1 | +0.094 | +0.101 |
| lora-code-5M | noise0.1s2 | +0.112 | +0.120 |
| lora-code-5M | prune0.25 | +0.212 | +0.231 |
| lora-code-5M | prune0.5 | +0.161 | +0.172 |
| lora-code_s1-20M | lora-wiki_de-10M | +0.135 | +0.149 |
| lora-code_s1-20M | lora-wiki_de-1M | +0.078 | +0.085 |
| lora-code_s1-20M | lora-wiki_de-20M | +0.174 | +0.191 |
| lora-code_s1-20M | lora-wiki_de-5M | +0.154 | +0.168 |
| lora-code_s1-20M | lora-wiki_de_s1-20M | +0.170 | +0.187 |
| lora-code_s1-20M | noise0.1s0 | -0.017 | -0.018 |
| lora-code_s1-20M | noise0.1s1 | +0.073 | +0.078 |
| lora-code_s1-20M | noise0.1s2 | +0.096 | +0.103 |
| lora-code_s1-20M | prune0.25 | +0.190 | +0.206 |
| lora-code_s1-20M | prune0.5 | +0.184 | +0.197 |
| lora-wiki_de-10M | noise0.1s0 | -0.049 | -0.051 |
| lora-wiki_de-10M | noise0.1s1 | -0.008 | -0.008 |
| lora-wiki_de-10M | noise0.1s2 | -0.008 | -0.008 |
| lora-wiki_de-10M | prune0.25 | +0.166 | +0.175 |
| lora-wiki_de-10M | prune0.5 | -0.121 | -0.126 |
| lora-wiki_de-1M | noise0.1s0 | -0.069 | -0.072 |
| lora-wiki_de-1M | noise0.1s1 | +0.009 | +0.009 |
| lora-wiki_de-1M | noise0.1s2 | -0.067 | -0.070 |
| lora-wiki_de-1M | prune0.25 | +0.129 | +0.136 |
| lora-wiki_de-1M | prune0.5 | -0.173 | -0.179 |
| lora-wiki_de-20M | noise0.1s0 | -0.050 | -0.053 |
| lora-wiki_de-20M | noise0.1s1 | +0.024 | +0.025 |
| lora-wiki_de-20M | noise0.1s2 | +0.017 | +0.018 |
| lora-wiki_de-20M | prune0.25 | +0.207 | +0.220 |
| lora-wiki_de-20M | prune0.5 | -0.066 | -0.069 |
| lora-wiki_de-5M | noise0.1s0 | -0.063 | -0.066 |
| lora-wiki_de-5M | noise0.1s1 | +0.048 | +0.050 |
| lora-wiki_de-5M | noise0.1s2 | +0.008 | +0.009 |
| lora-wiki_de-5M | prune0.25 | +0.180 | +0.189 |
| lora-wiki_de-5M | prune0.5 | -0.115 | -0.119 |
| lora-wiki_de_s1-20M | noise0.1s0 | -0.061 | -0.064 |
| lora-wiki_de_s1-20M | noise0.1s1 | +0.001 | +0.001 |
| lora-wiki_de_s1-20M | noise0.1s2 | -0.010 | -0.011 |
| lora-wiki_de_s1-20M | prune0.25 | +0.194 | +0.205 |
| lora-wiki_de_s1-20M | prune0.5 | -0.099 | -0.103 |
| noise0.1s0 | prune0.25 | -0.012 | -0.013 |
| noise0.1s0 | prune0.5 | +0.116 | +0.118 |
| noise0.1s1 | prune0.25 | -0.011 | -0.012 |
| noise0.1s1 | prune0.5 | +0.139 | +0.141 |
| noise0.1s2 | prune0.25 | +0.152 | +0.157 |
| noise0.1s2 | prune0.5 | +0.277 | +0.281 |
| prune0.25 | prune0.5 | +0.151 | +0.155 |

## Composite durability (mean of z-scored changes)

| set | variants | split-half r | reliability |
|---|---|---|---|
| LoRA 20M (all seeds) | 4 | +0.830 | +0.907 |
| all LoRA milestones | 10 | +0.832 | +0.908 |
| all variants | 15 | +0.847 | +0.917 |

## Topology: context-sampling reliability (split-half; base model)

gap = clusters as in token_senses.py (gap statistic); stable = only splits that reappear across the quarters of a half (prediction strength >= 0.8).

| measure | gap: mean (A) | gap: reliability | stable: mean (A) | stable: reliability |
|---|---|---|---|---|
| senses | 2.68 | +0.826 | 1.49 | +0.782 |
| bars | 16.34 | +0.740 | 9.97 | +0.644 |
| mean_len | 4.94 | +0.575 | 2.23 | +0.746 |
| total_pers | 56.96 | +0.870 | 21.41 | +0.859 |
| b1 | 5.90 | +0.620 | 5.11 | +0.666 |
| stable k (middle layers) | | | 1.49 | +0.782 |
## Ceiling: largest observable topology-durability correlation = sqrt(rel_topology x rel_durability)

Durability: all variants (reliability +0.917).

| measure | gap | stable |
|---|---|---|
| senses | 0.871 | 0.847 |
| bars | 0.824 | 0.769 |
| mean_len | 0.727 | 0.827 |
| total_pers | 0.893 | 0.888 |
| b1 | 0.754 | 0.782 |

![reliability](reliability_cfoot.png)
