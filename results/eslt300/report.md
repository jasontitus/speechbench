# speechbench report — run `eslt300`

Pulled 80 result file(s).

## common_voice_22_es

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.019 | 0.052 | 63.2 | 64.3 | 96 | 2 | 32 |
| fw-large-v3 | faster-whisper | 300 | 0.025 | 0.063 | 8.9 | 8.7 | 683 | 1184 | 208 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.027 | 0.065 | 14.9 | 14.8 | 404 | 288 | 124 |
| whisper-medium | transformers | 300 | 0.035 | 0.091 | 8.0 | 7.5 | 779 | 0 | 237 |
| whisper-large-v3 | transformers | 300 | 0.089 | 0.140 | 4.9 | 4.7 | 1412 | 320 | 426 |
| whisper-large-v3-turbo | transformers | 300 | 0.126 | 0.203 | 16.3 | 16.0 | 387 | 0 | 119 |
| whisper-large-v2 | transformers | 300 | 0.175 | 0.308 | 4.7 | 4.5 | 1540 | 340 | 473 |
| whisper-small | transformers | 300 | 0.197 | 0.291 | 17.9 | 16.6 | 401 | 0 | 123 |
| whisper-base | transformers | 300 | 0.385 | 0.707 | 33.3 | 31.4 | 254 | 0 | 79 |
| whisper-tiny | transformers | 300 | 0.523 | 0.955 | 42.4 | 40.5 | 214 | 0 | 67 |

## common_voice_22_lt

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.038 | 0.167 | 65.3 | 65.5 | 85 | 0 | 27 |
| whisper-large-v3 | transformers | 300 | 0.065 | 0.282 | 3.6 | 3.5 | 1581 | 0 | 476 |
| fw-large-v3 | faster-whisper | 300 | 0.066 | 0.285 | 7.0 | 6.7 | 788 | 256 | 239 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.081 | 0.325 | 13.0 | 12.5 | 424 | 288 | 129 |
| whisper-large-v3-turbo | transformers | 300 | 0.086 | 0.338 | 12.9 | 12.5 | 426 | 0 | 130 |
| whisper-large-v2 | transformers | 300 | 0.093 | 0.377 | 3.5 | 3.4 | 1619 | 0 | 489 |
| whisper-medium | transformers | 300 | 0.130 | 0.501 | 5.5 | 5.3 | 1034 | 0 | 312 |
| whisper-small | transformers | 300 | 0.203 | 0.723 | 11.7 | 11.3 | 490 | 0 | 149 |
| whisper-base | transformers | 300 | 0.299 | 0.909 | 22.7 | 22.0 | 252 | 0 | 78 |
| whisper-tiny | transformers | 300 | 0.474 | 1.094 | 28.7 | 27.7 | 236 | 0 | 73 |

## common_voice_25_lt

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.063 | 0.205 | 67.9 | 67.0 | 83 | 0 | 27 |
| whisper-large-v3 | transformers | 300 | 0.072 | 0.303 | 3.6 | 3.4 | 1624 | 0 | 489 |
| fw-large-v3 | faster-whisper | 300 | 0.073 | 0.306 | 7.2 | 6.8 | 799 | 288 | 241 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.097 | 0.366 | 13.3 | 12.7 | 431 | 288 | 131 |
| whisper-large-v2 | transformers | 300 | 0.104 | 0.394 | 3.6 | 3.4 | 1642 | 0 | 508 |
| whisper-large-v3-turbo | transformers | 300 | 0.115 | 0.392 | 13.2 | 12.8 | 446 | 0 | 136 |
| whisper-medium | transformers | 300 | 0.138 | 0.527 | 5.6 | 5.4 | 1053 | 0 | 318 |
| whisper-small | transformers | 300 | 0.213 | 0.727 | 11.8 | 11.3 | 503 | 0 | 153 |
| whisper-base | transformers | 300 | 0.326 | 0.917 | 23.1 | 22.2 | 271 | 0 | 83 |
| whisper-tiny | transformers | 300 | 0.564 | 1.184 | 28.3 | 27.2 | 266 | 0 | 81 |

## fleurs_es

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | transformers | 300 | 0.009 | 0.024 | 4.9 | 4.8 | 2514 | 0 | 760 |
| whisper-large-v2 | transformers | 300 | 0.010 | 0.028 | 4.8 | 4.7 | 2568 | 0 | 776 |
| fw-large-v3 | faster-whisper | 300 | 0.011 | 0.027 | 10.9 | 10.7 | 1111 | 416 | 338 |
| whisper-large-v3-turbo | transformers | 300 | 0.011 | 0.028 | 20.6 | 20.2 | 586 | 0 | 180 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.011 | 0.028 | 24.0 | 23.8 | 499 | 288 | 155 |
| whisper-medium | transformers | 300 | 0.012 | 0.035 | 7.5 | 7.4 | 1644 | 0 | 498 |
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.013 | 0.034 | 82.0 | 85.1 | 148 | 96 | 50 |
| whisper-small | transformers | 300 | 0.016 | 0.050 | 15.9 | 15.6 | 777 | 0 | 239 |
| whisper-base | transformers | 300 | 0.030 | 0.094 | 30.4 | 29.5 | 408 | 0 | 127 |
| whisper-tiny | transformers | 300 | 0.048 | 0.158 | 39.2 | 38.6 | 315 | 0 | 99 |

## fleurs_lt

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | transformers | 300 | 0.059 | 0.227 | 3.2 | 3.0 | 3446 | 0 | 1039 |
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.059 | 0.227 | 80.1 | 82.3 | 137 | 0 | 46 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.062 | 0.243 | 19.9 | 18.5 | 547 | 288 | 169 |
| fw-large-v3 | faster-whisper | 300 | 0.063 | 0.239 | 7.8 | 7.3 | 1401 | 448 | 426 |
| whisper-large-v2 | transformers | 300 | 0.071 | 0.277 | 3.2 | 2.9 | 3539 | 0 | 1067 |
| whisper-large-v3-turbo | transformers | 300 | 0.078 | 0.263 | 14.6 | 13.7 | 761 | 0 | 233 |
| whisper-medium | transformers | 300 | 0.104 | 0.406 | 4.9 | 4.5 | 2309 | 0 | 699 |
| whisper-small | transformers | 300 | 0.186 | 0.660 | 10.0 | 9.2 | 1127 | 0 | 344 |
| whisper-base | transformers | 300 | 0.311 | 0.876 | 18.6 | 17.5 | 617 | 0 | 189 |
| whisper-tiny | transformers | 300 | 0.520 | 1.184 | 23.1 | 22.1 | 577 | 0 | 177 |

## mls_es

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3-turbo | transformers | 300 | 0.014 | 0.034 | 22.6 | 22.8 | 651 | 272 | 212 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.015 | 0.036 | 28.1 | 28.0 | 527 | 328 | 175 |
| whisper-large-v2 | transformers | 300 | 0.015 | 0.037 | 4.8 | 4.9 | 3104 | 1522 | 966 |
| whisper-medium | transformers | 300 | 0.016 | 0.046 | 7.5 | 7.6 | 2001 | 892 | 617 |
| whisper-large-v3 | transformers | 300 | 0.016 | 0.036 | 5.3 | 5.3 | 2840 | 1448 | 869 |
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.022 | 0.067 | 77.2 | 77.0 | 196 | 262 | 75 |
| whisper-small | transformers | 300 | 0.024 | 0.071 | 15.6 | 15.8 | 961 | 352 | 305 |
| fw-large-v3 | faster-whisper | 300 | 0.031 | 0.053 | 11.7 | 11.9 | 1279 | 1288 | 401 |
| whisper-base | transformers | 300 | 0.043 | 0.132 | 29.9 | 30.3 | 501 | 174 | 167 |
| whisper-tiny | transformers | 300 | 0.063 | 0.198 | 39.0 | 39.1 | 385 | 92 | 133 |

## voxpopuli_es

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 300 | 0.040 | 0.064 | 66.1 | 70.5 | 182 | 620 | 75 |
| whisper-large-v2 | transformers | 300 | 0.052 | 0.075 | 3.9 | 3.9 | 2986 | 40 | 910 |
| whisper-large-v3 | transformers | 300 | 0.053 | 0.076 | 4.0 | 4.0 | 2956 | 80 | 901 |
| fw-large-v3 | faster-whisper | 300 | 0.054 | 0.078 | 9.1 | 9.1 | 1258 | 448 | 391 |
| whisper-medium | transformers | 300 | 0.054 | 0.081 | 6.1 | 6.0 | 1928 | 56 | 592 |
| fw-large-v3-turbo | faster-whisper | 300 | 0.055 | 0.078 | 21.2 | 21.0 | 545 | 288 | 177 |
| whisper-large-v3-turbo | transformers | 300 | 0.088 | 0.126 | 17.3 | 17.2 | 673 | 40 | 216 |
| whisper-small | transformers | 300 | 0.116 | 0.162 | 12.9 | 12.7 | 977 | 90 | 306 |
| whisper-base | transformers | 300 | 0.131 | 0.190 | 24.9 | 24.6 | 501 | 8 | 220 |
| whisper-tiny | transformers | 300 | 0.180 | 0.305 | 31.9 | 32.2 | 412 | 6 | 138 |

## voxpopuli_lt

| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 42 | 0.178 | 0.298 | 70.9 | 73.3 | 144 | 0 | 7 |
| fw-large-v3-turbo | faster-whisper | 42 | 0.185 | 0.304 | 17.4 | 17.1 | 648 | 288 | 28 |
| fw-large-v3 | faster-whisper | 42 | 0.197 | 0.289 | 7.5 | 7.2 | 1405 | 416 | 60 |
| whisper-large-v2 | transformers | 42 | 0.203 | 0.330 | 2.9 | 2.9 | 3573 | 0 | 151 |
| whisper-medium | transformers | 42 | 0.207 | 0.420 | 4.3 | 4.2 | 2404 | 0 | 102 |
| whisper-small | transformers | 42 | 0.251 | 0.578 | 9.2 | 9.0 | 1149 | 0 | 49 |
| whisper-base | transformers | 42 | 0.336 | 0.810 | 16.8 | 16.3 | 630 | 0 | 27 |
| whisper-large-v3-turbo | transformers | 42 | 0.405 | 0.841 | 14.3 | 13.1 | 824 | 0 | 36 |
| whisper-tiny | transformers | 42 | 0.530 | 1.056 | 20.8 | 21.4 | 610 | 0 | 27 |
| whisper-large-v3 | transformers | 42 | 0.546 | 0.607 | 2.7 | 2.9 | 5113 | 0 | 216 |
