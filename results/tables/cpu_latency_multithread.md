Multi-thread check, reported as a limitation of this setup. *_vs_1_thread compares each median with the same model's 1-thread median from cpu_latency: below 1x means more threads helped, above 1x means they made it slower. Where INT8 gets slower with more threads while FP32 gets faster, the cause is most likely the threading of this PyTorch build on Windows (not investigated further), not INT8 as such. Median over rounds (lowest-highest round in brackets), milliseconds. One laptop only (13th Gen Intel(R) Core(TM) i5-13420H, 12 logical CPUs), PyTorch 2.14.1+cu130, onednn INT8 engine, batch size 1, seed-0 models. Times on other machines or engines can differ a lot.

| model | threads | fp32_ms | int8_ms | fp32_vs_1_thread | int8_vs_1_thread |
|---|---|---|---|---|---|
| resnet18 | 2 | 39.3 (39.2-39.7) | 226.1 (221.8-230.0) | 0.36x | 7.69x |
| resnet18 | 4 | 25.9 (23.0-30.0) | 133.1 (126.4-190.0) | 0.24x | 4.53x |
| resnet18 | 8 | 26.9 (26.3-26.9) | 121.7 (121.3-122.1) | 0.25x | 4.14x |
| mobilenet_v2 | 2 | 17.9 (17.7-19.4) | 10.1 (9.8-10.4) | 0.55x | 0.80x |
| mobilenet_v2 | 4 | 15.6 (15.4-20.0) | 10.1 (10.1-11.8) | 0.48x | 0.80x |
| mobilenet_v2 | 8 | 15.9 (15.6-16.3) | 11.1 (10.5-11.3) | 0.49x | 0.88x |
| googlenet | 2 | 74.1 (73.0-75.1) | 27.0 (26.6-29.7) | 0.47x | 1.20x |
| googlenet | 4 | 51.9 (49.4-52.7) | 24.2 (23.9-24.2) | 0.33x | 1.07x |
| googlenet | 8 | 43.5 (43.1-44.0) | 30.6 (30.4-32.1) | 0.27x | 1.36x |
