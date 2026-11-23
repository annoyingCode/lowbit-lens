Activation range search, tested before choosing (D16): PyTorch's stock HistogramObserver (always assumes 256 levels) vs a variant told the real number of levels. Test accuracy, seed-0 models. The stock observer was equal or better, so all results use it.

| model | seed | setting | stock_acc | levels_aware_acc |
|---|---|---|---|---|
| resnet18 | 0 | W4A4 | 0.9819 | 0.9673 |
| resnet18 | 0 | W2A8 | 0.1692 | 0.1692 |
| resnet18 | 0 | W2A2 | 0.1698 | 0.1210 |
| mobilenet_v2 | 0 | W4A4 | 0.0716 | 0.0713 |
| mobilenet_v2 | 0 | W2A8 | 0.1824 | 0.1824 |
| mobilenet_v2 | 0 | W2A2 | 0.1824 | 0.1824 |
