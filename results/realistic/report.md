# speechbench report — run `realistic`

Pulled 32 result file(s).

## ami_ihm

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v2 | nemo | 10 | 0.194 | 0.222 | 7.9 | 8.0 | 123 | 0 | 2 |
| parakeet-rnnt-1.1b | nemo | 10 | 0.226 | 0.119 | 4.9 | 4.4 | 196 | 0 | 3 |
| whisper-tiny.en | transformers | 10 | 0.258 | 0.230 | 10.2 | 10.7 | 97 | 0 | 2 |
| whisper-large-v3-turbo | transformers | 10 | 0.258 | 0.185 | 3.7 | 3.8 | 251 | 0 | 3 |
| parakeet-tdt-1.1b | nemo | 10 | 0.323 | 0.163 | 5.9 | 5.9 | 163 | 0 | 3 |
| parakeet-ctc-1.1b | nemo | 10 | 0.323 | 0.178 | 6.8 | 5.5 | 146 | 0 | 3 |
| gemma-4-E4B-it | transformers | 10 | 0.742 | 0.644 | 0.1 | 0.1 | 10089 | 5376 | 102 |
| qwen3-asr-1.7b | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |

## earnings22

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3-turbo | transformers | 10 | 0.179 | 0.120 | 10.5 | 11.4 | 513 | 0 | 6 |
| gemma-4-E4B-it | transformers | 10 | 0.185 | 0.108 | 0.2 | 0.2 | 32028 | 5440 | 323 |
| parakeet-rnnt-1.1b | nemo | 10 | 0.185 | 0.123 | 24.1 | 27.0 | 241 | 0 | 4 |
| parakeet-ctc-1.1b | nemo | 10 | 0.185 | 0.101 | 34.4 | 35.4 | 163 | 0 | 3 |
| parakeet-tdt-0.6b-v2 | nemo | 10 | 0.202 | 0.121 | 39.0 | 44.3 | 139 | 0 | 2 |
| whisper-tiny.en | transformers | 10 | 0.220 | 0.144 | 23.9 | 25.4 | 239 | 0 | 3 |
| parakeet-tdt-1.1b | nemo | 10 | 0.220 | 0.130 | 22.7 | 26.0 | 238 | 0 | 4 |
| qwen3-asr-1.7b | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |

## librispeech_clean

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-tdt-0.6b-v2 | nemo | 10 | 0.004 | 0.002 | 22.9 | 21.2 | 548 | 172 | 7 |
| parakeet-rnnt-1.1b | nemo | 10 | 0.004 | 0.002 | 30.5 | 31.4 | 441 | 148 | 6 |
| whisper-large-v3-turbo | transformers | 10 | 0.004 | 0.002 | 15.5 | 14.9 | 704 | 328 | 8 |
| parakeet-tdt-1.1b | nemo | 10 | 0.004 | 0.002 | 31.7 | 33.1 | 267 | 104 | 4 |
| parakeet-ctc-1.1b | nemo | 10 | 0.004 | 0.002 | 43.6 | 42.2 | 194 | 110 | 3 |
| whisper-tiny.en | transformers | 10 | 0.020 | 0.009 | 24.6 | 25.2 | 392 | 92 | 5 |
| gemma-4-E4B-it | transformers | 10 | 0.020 | 0.010 | 0.2 | 0.2 | 49226 | 5514 | 516 |
| qwen3-asr-1.7b | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |

## voxpopuli_en

| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parakeet-rnnt-1.1b | nemo | 10 | 0.016 | 0.007 | 36.1 | 37.2 | 262 | 0 | 18 |
| parakeet-tdt-1.1b | nemo | 10 | 0.016 | 0.007 | 37.2 | 39.8 | 256 | 0 | 16 |
| parakeet-ctc-1.1b | nemo | 10 | 0.024 | 0.011 | 47.3 | 50.5 | 202 | 0 | 19 |
| parakeet-tdt-0.6b-v2 | nemo | 10 | 0.032 | 0.019 | 55.6 | 59.2 | 175 | 0 | 16 |
| whisper-large-v3-turbo | transformers | 10 | 0.056 | 0.030 | 19.2 | 18.9 | 487 | 0 | 21 |
| whisper-tiny.en | transformers | 10 | 0.072 | 0.036 | 34.4 | 39.2 | 279 | 0 | 18 |
| gemma-4-E4B-it | transformers | 10 | 0.080 | 0.042 | 0.2 | 0.2 | 47010 | 5472 | 485 |
| qwen3-asr-1.7b | ? | 0 | 0.000 | 0.000 | 0.0 | 0.0 | 0 | 0 | 0 |
