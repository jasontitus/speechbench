# speechbench report — run `en_cv22`

Pulled 23 result file(s).

## common_voice_22_en

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-rnnt-1.1b | nemo | 30 | 0.063 | 0.041 | 29.3 | 30.1 | 234 | 78 | 8 |
| parakeet-tdt-1.1b | nemo | 30 | 0.067 | 0.034 | 34.4 | 35.4 | 148 | 54 | 6 |
| parakeet-tdt-0.6b-v2 | nemo | 30 | 0.090 | 0.058 | 60.7 | 60.1 | 130 | 112 | 4 |
| parakeet-ctc-1.1b | nemo | 30 | 0.094 | 0.053 | 40.4 | 40.5 | 127 | 30 | 4 |
| parakeet-tdt-0.6b-v3 | nemo | 30 | 0.102 | 0.061 | 59.9 | 59.2 | 142 | 132 | 5 |
| whisper-medium | transformers | 30 | 0.118 | 0.063 | 9.5 | 8.9 | 566 | 854 | 18 |
| whisper-large-v3 | transformers | 30 | 0.122 | 0.069 | 5.8 | 5.5 | 928 | 1406 | 28 |
| parakeet-rnnt-0.6b | nemo | 30 | 0.122 | 0.061 | 57.6 | 56.8 | 141 | 50 | 5 |
| whisper-large-v3-turbo | transformers | 30 | 0.122 | 0.059 | 16.9 | 15.5 | 305 | 264 | 10 |
| fw-large-v3-turbo | faster-whisper | 30 | 0.122 | 0.059 | 13.9 | 14.1 | 374 | 328 | 12 |
| whisper-large-v2 | transformers | 30 | 0.129 | 0.072 | 5.3 | 5.3 | 1002 | 1480 | 119 |
| parakeet-tdt_ctc-110m | nemo | 30 | 0.129 | 0.067 | 3.2 | 3.2 | 1629 | 674 | 49 |
| whisper-small | transformers | 30 | 0.133 | 0.069 | 21.2 | 19.8 | 257 | 334 | 8 |
| fw-large-v3 | faster-whisper | 30 | 0.133 | 0.075 | 8.6 | 8.1 | 595 | 298 | 18 |
| parakeet-ctc-0.6b | nemo | 30 | 0.133 | 0.075 | 71.4 | 66.7 | 73 | 28 | 3 |
| distil-large-v3 | transformers | 30 | 0.145 | 0.075 | 19.5 | 18.8 | 264 | 186 | 8 |
| whisper-small.en | transformers | 30 | 0.153 | 0.076 | 21.1 | 19.4 | 257 | 334 | 8 |
| whisper-medium.en | transformers | 30 | 0.157 | 0.091 | 8.9 | 8.4 | 604 | 852 | 19 |
| whisper-base.en | transformers | 30 | 0.157 | 0.094 | 36.5 | 33.9 | 148 | 170 | 5 |
| fw-distil-large-v3 | faster-whisper | 30 | 0.161 | 0.079 | 14.5 | 14.8 | 358 | 296 | 11 |
| whisper-base | transformers | 30 | 0.184 | 0.096 | 38.3 | 34.7 | 144 | 170 | 5 |
| whisper-tiny | transformers | 30 | 0.239 | 0.135 | 44.6 | 41.5 | 121 | 88 | 4 |
| whisper-tiny.en | transformers | 30 | 0.263 | 0.143 | 48.2 | 44.5 | 113 | 88 | 4 |
