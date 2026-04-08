# speechbench report — run `es2`

Pulled 20 result file(s).

## common_voice_22_es

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.081 | 0.053 | 48.5 | 47.6 | 120 | 2 | 4 |
| fw-large-v3 | faster-whisper | 30 | 0.085 | 0.052 | 8.5 | 8.3 | 688 | 256 | 21 |
| whisper-large-v3 | transformers | 30 | 0.095 | 0.055 | 4.8 | 4.3 | 2050 | 400 | 62 |
| whisper-large-v3-turbo | transformers | 30 | 0.105 | 0.058 | 15.2 | 14.6 | 382 | 0 | 12 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.108 | 0.061 | 14.2 | 14.3 | 411 | 288 | 13 |
| whisper-medium | transformers | 30 | 0.153 | 0.068 | 7.4 | 7.1 | 804 | 0 | 25 |
| whisper-large-v2 | transformers | 30 | 0.851 | 0.864 | 4.4 | 4.4 | 2129 | 382 | 134 |
| whisper-tiny | transformers | 30 | 0.925 | 0.485 | 38.3 | 37.6 | 249 | 6 | 8 |
| whisper-small | transformers | 30 | 0.966 | 0.871 | 15.7 | 15.4 | 634 | 108 | 20 |
| whisper-base | transformers | 30 | 1.081 | 0.533 | 30.8 | 30.9 | 321 | 8 | 10 |

## fleurs_es

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | transformers | 30 | 0.024 | 0.008 | 4.8 | 4.7 | 2443 | 1448 | 74 |
| whisper-large-v2 | transformers | 30 | 0.025 | 0.008 | 4.8 | 4.7 | 2455 | 1520 | 95 |
| fw-large-v3 | faster-whisper | 30 | 0.025 | 0.008 | 10.8 | 10.6 | 1073 | 360 | 33 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.026 | 0.010 | 23.8 | 23.8 | 483 | 328 | 15 |
| whisper-large-v3-turbo | transformers | 30 | 0.030 | 0.013 | 20.4 | 20.1 | 564 | 270 | 18 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.031 | 0.014 | 67.6 | 70.1 | 242 | 138 | 8 |
| whisper-medium | transformers | 30 | 0.036 | 0.011 | 7.2 | 7.0 | 1623 | 886 | 51 |
| whisper-small | transformers | 30 | 0.052 | 0.015 | 14.9 | 14.6 | 784 | 344 | 25 |
| whisper-base | transformers | 30 | 0.090 | 0.031 | 30.4 | 29.7 | 387 | 174 | 13 |
| whisper-tiny | transformers | 30 | 0.166 | 0.055 | 39.3 | 39.0 | 299 | 92 | 10 |
