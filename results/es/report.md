# speechbench report — run `es`

Pulled 36 result file(s).

## fleurs_es

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E2B-it | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| parakeet-tdt-0.6b-v3 | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-tiny | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-large-v2 | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-large-v3 | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| fw-large-v3-turbo | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-small | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-medium | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-base | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| fw-large-v3 | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| gemma-4-E4B-it | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
| whisper-large-v3-turbo | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |

## mls_es

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fw-large-v3-turbo | faster-whisper | 30 | 0.020 | 0.005 | 30.0 | 30.7 | 509 | 328 | 18 |
| whisper-large-v3-turbo | transformers | 30 | 0.022 | 0.007 | 24.4 | 23.9 | 632 | 270 | 21 |
| whisper-large-v2 | transformers | 30 | 0.022 | 0.007 | 5.3 | 5.1 | 2939 | 1450 | 102 |
| whisper-large-v3 | transformers | 30 | 0.022 | 0.007 | 5.7 | 5.5 | 2735 | 1448 | 88 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.026 | 0.009 | 77.0 | 78.2 | 266 | 194 | 11 |
| whisper-medium | transformers | 30 | 0.029 | 0.009 | 8.0 | 7.7 | 1948 | 896 | 61 |
| whisper-small | transformers | 30 | 0.043 | 0.013 | 16.6 | 16.0 | 944 | 356 | 31 |
| gemma-4-E4B-it | transformers | 30 | 0.048 | 0.017 | 0.2 | 0.2 | 72304 | 5532 | 2192 |
| fw-large-v3 | faster-whisper | 30 | 0.061 | 0.045 | 12.4 | 12.6 | 1276 | 330 | 41 |
| gemma-4-E2B-it | transformers | 30 | 0.064 | 0.021 | 4.7 | 4.6 | 3310 | 30 | 102 |
| whisper-base | transformers | 30 | 0.106 | 0.027 | 30.7 | 30.3 | 512 | 174 | 18 |
| whisper-tiny | transformers | 30 | 0.206 | 0.066 | 39.6 | 39.1 | 398 | 92 | 15 |

## voxpopuli_es

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.057 | 0.037 | 63.9 | 67.0 | 175 | 224 | 172 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.076 | 0.056 | 21.5 | 20.4 | 521 | 288 | 29 |
| whisper-large-v3 | transformers | 30 | 0.078 | 0.061 | 4.0 | 4.0 | 2866 | 2 | 177 |
| whisper-large-v3-turbo | transformers | 30 | 0.079 | 0.058 | 17.2 | 16.9 | 652 | 0 | 34 |
| whisper-large-v2 | transformers | 30 | 0.079 | 0.058 | 4.0 | 4.0 | 2888 | 0 | 119 |
| whisper-medium | transformers | 30 | 0.086 | 0.064 | 6.1 | 6.1 | 1908 | 4 | 70 |
| fw-large-v3 | faster-whisper | 30 | 0.101 | 0.077 | 9.4 | 9.3 | 1216 | 352 | 50 |
| whisper-small | transformers | 30 | 0.102 | 0.071 | 12.7 | 12.5 | 918 | 12 | 40 |
| gemma-4-E4B-it | transformers | 30 | 0.104 | 0.073 | 0.2 | 0.2 | 64811 | 5472 | 1959 |
| whisper-base | transformers | 30 | 0.149 | 0.083 | 24.4 | 24.2 | 480 | 0 | 27 |
| gemma-4-E2B-it | transformers | 30 | 0.165 | 0.131 | 5.8 | 3.9 | 2827 | 40 | 97 |
| whisper-tiny | transformers | 30 | 0.196 | 0.101 | 31.8 | 32.0 | 366 | 0 | 110 |
