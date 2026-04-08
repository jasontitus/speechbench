# speechbench report — run `full`

Pulled 105 result file(s).

## ami_ihm

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.192 | 0.153 | 13.7 | 10.4 | 97 | 0 | 4 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.192 | 0.162 | 13.9 | 10.8 | 95 | 0 | 4 |
| whisper-medium.en | transformers | 30 | 0.232 | 0.188 | 2.9 | 2.3 | 1008 | 224 | 82 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.240 | 0.197 | 4.0 | 3.1 | 334 | 224 | 11 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.248 | 0.199 | 3.8 | 3.0 | 343 | 288 | 11 |
| whisper-large-v2 | transformers | 30 | 0.248 | 0.190 | 1.8 | 1.3 | 726 | 0 | 23 |
| fw-large-v3 | faster-whisper | 30 | 0.256 | 0.218 | 2.6 | 2.1 | 482 | 256 | 17 |
| distil-large-v3 | transformers | 30 | 0.256 | 0.202 | 6.3 | 4.8 | 204 | 0 | 88 |
| whisper-large-v3-turbo | transformers | 30 | 0.288 | 0.236 | 5.1 | 4.1 | 250 | 0 | 9 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.296 | 0.222 | 0.8 | 0.6 | 1612 | 24 | 49 |
| parakeet-tdt-1.1b | nemo | 30 | 0.296 | 0.191 | 9.9 | 7.4 | 135 | 0 | 5 |
| parakeet-ctc-1.1b | nemo | 30 | 0.304 | 0.202 | 9.0 | 5.8 | 161 | 0 | 6 |
| parakeet-rnnt-1.1b | nemo | 30 | 0.304 | 0.209 | 8.5 | 6.2 | 163 | 2 | 6 |
| parakeet-ctc-0.6b | nemo | 30 | 0.320 | 0.208 | 15.5 | 9.9 | 90 | 0 | 4 |
| whisper-base.en | transformers | 30 | 0.328 | 0.273 | 9.3 | 8.6 | 134 | 0 | 5 |
| parakeet-rnnt-0.6b | nemo | 30 | 0.336 | 0.226 | 13.7 | 9.9 | 99 | 0 | 4 |
| whisper-small.en | transformers | 30 | 0.344 | 0.283 | 6.1 | 5.0 | 499 | 98 | 16 |
| whisper-tiny.en | transformers | 30 | 0.344 | 0.285 | 12.4 | 11.7 | 332 | 6 | 11 |
| gemma-4-E4B-it | transformers | 30 | 0.616 | 0.556 | 0.1 | 0.1 | 13317 | 5396 | 401 |
| gemma-4-E2B-it | transformers | 30 | 1.168 | 1.264 | 1.7 | 1.6 | 1028 | 0 | 32 |
| whisper-large-v3 | transformers | 30 | 3.080 | 3.401 | 1.7 | 1.5 | 2397 | 400 | 83 |

## earnings22

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fw-large-v3 | faster-whisper | 30 | 0.153 | 0.097 | 7.7 | 8.2 | 679 | 256 | 22 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.159 | 0.105 | 13.5 | 13.9 | 401 | 288 | 13 |
| whisper-large-v2 | transformers | 30 | 0.161 | 0.105 | 4.0 | 4.2 | 1337 | 0 | 42 |
| whisper-large-v3-turbo | transformers | 30 | 0.161 | 0.104 | 13.8 | 14.3 | 393 | 0 | 13 |
| distil-large-v3 | transformers | 30 | 0.166 | 0.106 | 16.9 | 17.7 | 316 | 170 | 11 |
| parakeet-ctc-1.1b | nemo | 30 | 0.172 | 0.106 | 31.1 | 35.3 | 178 | 0 | 7 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.174 | 0.106 | 46.2 | 42.6 | 121 | 0 | 5 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.174 | 0.111 | 14.5 | 14.5 | 379 | 224 | 13 |
| whisper-large-v3 | transformers | 30 | 0.178 | 0.110 | 3.9 | 4.2 | 1347 | 0 | 64 |
| whisper-small.en | transformers | 30 | 0.180 | 0.117 | 12.1 | 12.8 | 466 | 0 | 16 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.187 | 0.121 | 50.1 | 50.7 | 109 | 0 | 5 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.187 | 0.125 | 3.4 | 3.4 | 1664 | 48 | 51 |
| whisper-medium.en | transformers | 30 | 0.193 | 0.123 | 6.2 | 6.6 | 860 | 0 | 37 |
| whisper-base.en | transformers | 30 | 0.193 | 0.124 | 20.1 | 21.9 | 278 | 0 | 10 |
| parakeet-ctc-0.6b | nemo | 30 | 0.195 | 0.116 | 52.6 | 43.6 | 112 | 0 | 4 |
| parakeet-rnnt-1.1b | nemo | 30 | 0.197 | 0.132 | 29.5 | 30.8 | 185 | 0 | 7 |
| gemma-4-E4B-it | transformers | 30 | 0.197 | 0.120 | 0.2 | 0.2 | 30032 | 5440 | 903 |
| whisper-tiny.en | transformers | 30 | 0.210 | 0.136 | 24.0 | 24.2 | 237 | 0 | 9 |
| parakeet-rnnt-0.6b | nemo | 30 | 0.210 | 0.132 | 47.2 | 43.9 | 117 | 0 | 5 |
| parakeet-tdt-1.1b | nemo | 30 | 0.218 | 0.134 | 30.1 | 30.5 | 177 | 0 | 7 |
| gemma-4-E2B-it | transformers | 30 | 0.235 | 0.139 | 3.3 | 3.4 | 1673 | 0 | 52 |

## librispeech_clean

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-ctc-0.6b | nemo | 30 | 0.015 | 0.008 | 67.1 | 62.8 | 108 | 90 | 4 |
| parakeet-rnnt-1.1b | nemo | 30 | 0.015 | 0.006 | 33.9 | 32.4 | 253 | 146 | 19 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.016 | 0.006 | 39.0 | 35.2 | 243 | 164 | 8 |
| fw-large-v3 | faster-whisper | 30 | 0.016 | 0.007 | 8.6 | 8.2 | 774 | 360 | 24 |
| parakeet-tdt-1.1b | nemo | 30 | 0.016 | 0.006 | 34.3 | 36.6 | 199 | 106 | 7 |
| parakeet-rnnt-0.6b | nemo | 30 | 0.018 | 0.008 | 54.3 | 58.1 | 175 | 116 | 6 |
| whisper-large-v2 | transformers | 30 | 0.020 | 0.008 | 3.9 | 3.8 | 1739 | 1450 | 53 |
| parakeet-ctc-1.1b | nemo | 30 | 0.021 | 0.008 | 34.5 | 33.9 | 212 | 100 | 8 |
| whisper-large-v3-turbo | transformers | 30 | 0.023 | 0.008 | 15.9 | 14.7 | 418 | 270 | 24 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.025 | 0.008 | 53.3 | 54.6 | 187 | 164 | 7 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.025 | 0.007 | 18.6 | 15.3 | 372 | 296 | 12 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.025 | 0.007 | 17.1 | 14.5 | 399 | 328 | 13 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.026 | 0.009 | 4.4 | 3.4 | 1646 | 674 | 51 |
| distil-large-v3 | transformers | 30 | 0.033 | 0.009 | 21.0 | 19.2 | 315 | 188 | 10 |
| whisper-small.en | transformers | 30 | 0.034 | 0.010 | 12.8 | 12.7 | 594 | 388 | 40 |
| whisper-tiny.en | transformers | 30 | 0.038 | 0.012 | 28.8 | 27.6 | 251 | 92 | 8 |
| whisper-base.en | transformers | 30 | 0.038 | 0.011 | 22.9 | 22.0 | 319 | 174 | 11 |
| whisper-large-v3 | transformers | 30 | 0.039 | 0.012 | 4.0 | 3.9 | 1680 | 1448 | 51 |
| whisper-medium.en | transformers | 30 | 0.041 | 0.013 | 6.3 | 6.1 | 1110 | 900 | 106 |
| gemma-4-E4B-it | transformers | 30 | 0.044 | 0.017 | 0.2 | 0.2 | 39692 | 5520 | 1212 |
| gemma-4-E2B-it | transformers | 30 | 0.049 | 0.017 | 3.4 | 3.3 | 2103 | 54 | 65 |

## librispeech_other

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fw-large-v3 | faster-whisper | 30 | 0.008 | 0.003 | 6.9 | 6.6 | 738 | 288 | 23 |
| parakeet-tdt-1.1b | nemo | 30 | 0.012 | 0.004 | 30.5 | 29.4 | 168 | 0 | 30 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.014 | 0.006 | 47.5 | 45.3 | 112 | 0 | 4 |
| parakeet-rnnt-1.1b | nemo | 30 | 0.015 | 0.005 | 27.2 | 25.0 | 199 | 0 | 7 |
| parakeet-ctc-0.6b | nemo | 30 | 0.017 | 0.006 | 52.0 | 45.3 | 107 | 0 | 4 |
| parakeet-rnnt-0.6b | nemo | 30 | 0.017 | 0.007 | 50.5 | 43.6 | 104 | 0 | 4 |
| parakeet-ctc-1.1b | nemo | 30 | 0.019 | 0.008 | 29.9 | 27.5 | 181 | 2 | 7 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.022 | 0.010 | 14.0 | 11.7 | 374 | 224 | 12 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.024 | 0.010 | 46.4 | 41.3 | 114 | 0 | 4 |
| whisper-medium.en | transformers | 30 | 0.027 | 0.012 | 4.8 | 4.7 | 1100 | 0 | 34 |
| whisper-large-v3-turbo | transformers | 30 | 0.027 | 0.011 | 12.8 | 12.1 | 399 | 0 | 24 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.029 | 0.012 | 3.3 | 2.6 | 1650 | 70 | 51 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.030 | 0.014 | 13.4 | 11.2 | 389 | 288 | 13 |
| whisper-large-v2 | transformers | 30 | 0.032 | 0.012 | 3.1 | 3.2 | 1669 | 0 | 51 |
| whisper-large-v3 | transformers | 30 | 0.039 | 0.015 | 3.1 | 3.1 | 1655 | 0 | 51 |
| whisper-base.en | transformers | 30 | 0.042 | 0.018 | 17.1 | 18.0 | 320 | 0 | 11 |
| whisper-small.en | transformers | 30 | 0.047 | 0.018 | 9.6 | 9.6 | 556 | 0 | 18 |
| gemma-4-E4B-it | transformers | 30 | 0.052 | 0.023 | 0.1 | 0.1 | 39396 | 5460 | 1183 |
| distil-large-v3 | transformers | 30 | 0.054 | 0.023 | 16.2 | 15.2 | 313 | 0 | 10 |
| gemma-4-E2B-it | transformers | 30 | 0.057 | 0.025 | 2.6 | 2.6 | 2037 | 0 | 62 |
| whisper-tiny.en | transformers | 30 | 0.059 | 0.026 | 21.7 | 22.9 | 247 | 0 | 9 |

## voxpopuli_en

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-rnnt-0.6b | nemo | 30 | 0.051 | 0.028 | 75.3 | 84.2 | 114 | 0 | 17 |
| parakeet-tdt-1.1b | nemo | 30 | 0.058 | 0.034 | 42.8 | 46.7 | 200 | 0 | 31 |
| parakeet-rnnt-1.1b | nemo | 30 | 0.059 | 0.034 | 41.9 | 45.0 | 203 | 0 | 152 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.060 | 0.037 | 78.4 | 86.2 | 111 | 0 | 17 |
| parakeet-ctc-1.1b | nemo | 30 | 0.060 | 0.033 | 46.9 | 51.3 | 184 | 0 | 79 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.062 | 0.040 | 75.1 | 84.7 | 115 | 0 | 17 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.069 | 0.041 | 5.4 | 5.1 | 1658 | 50 | 66 |
| parakeet-ctc-0.6b | nemo | 30 | 0.073 | 0.039 | 94.1 | 105.8 | 94 | 0 | 16 |
| whisper-medium.en | transformers | 30 | 0.075 | 0.046 | 7.3 | 7.1 | 1176 | 0 | 59 |
| fw-large-v3 | faster-whisper | 30 | 0.084 | 0.051 | 9.9 | 9.5 | 857 | 288 | 39 |
| whisper-large-v2 | transformers | 30 | 0.084 | 0.050 | 4.6 | 4.5 | 1840 | 0 | 69 |
| whisper-large-v3 | transformers | 30 | 0.084 | 0.051 | 4.7 | 4.5 | 1844 | 0 | 91 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.087 | 0.052 | 19.9 | 19.2 | 432 | 288 | 27 |
| whisper-small.en | transformers | 30 | 0.089 | 0.051 | 15.3 | 15.4 | 563 | 0 | 97 |
| whisper-large-v3-turbo | transformers | 30 | 0.091 | 0.056 | 19.1 | 18.8 | 445 | 0 | 48 |
| distil-large-v3 | transformers | 30 | 0.096 | 0.053 | 24.7 | 23.9 | 344 | 2 | 23 |
| whisper-base.en | transformers | 30 | 0.097 | 0.053 | 27.0 | 27.8 | 320 | 0 | 23 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.098 | 0.054 | 21.6 | 21.5 | 401 | 224 | 26 |
| whisper-tiny.en | transformers | 30 | 0.118 | 0.063 | 35.5 | 36.0 | 245 | 0 | 21 |
| gemma-4-E2B-it | transformers | 30 | 0.124 | 0.074 | 3.8 | 3.9 | 2287 | 0 | 83 |
| gemma-4-E4B-it | transformers | 30 | 0.141 | 0.087 | 0.2 | 0.2 | 44566 | 5474 | 1351 |
