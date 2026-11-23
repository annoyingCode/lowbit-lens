CPU time per image on 1 thread, in milliseconds. 5 rounds alternate FP32 and INT8; each round's value is the median of 100 runs after 10 warm-up runs. Shown: median over rounds (lowest-highest round in brackets); int8_faster_by is the median of per-round FP32/INT8 ratios. One laptop only (13th Gen Intel(R) Core(TM) i5-13420H, 12 logical CPUs), PyTorch 2.14.1+cu130, onednn INT8 engine, batch size 1, seed-0 models. Times on other machines or engines can differ a lot.

| model | fp32_ms | int8_ms | int8_faster_by |
|---|---|---|---|
| resnet18 | 109.1 (92.4-214.4) | 29.4 (19.3-85.7) | 3.7x (2.5-4.9) |
| mobilenet_v2 | 32.7 (29.3-57.8) | 12.6 (11.1-31.3) | 2.7x (1.8-2.9) |
| googlenet | 158.4 (152.1-333.6) | 22.6 (17.2-75.8) | 7.3x (4.4-9.1) |
