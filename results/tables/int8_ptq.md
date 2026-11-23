Real INT8 post-training quantization (512 calibration images from the training set, PyTorch's standard onednn recipe). FP32 and INT8 are both evaluated on the CPU on the official test set (3421 images); int8_acc_clean leaves out the 6 duplicated test images. same_prediction = share of test images where INT8 predicts the same class as FP32; changed = number of test images whose prediction changed. Sizes are saved weights in MB (10^6 bytes). Values are mean ± sample SD over seeds; changed_of_3421 lists seeds 0, 1, 2.

| model | fp32_acc | int8_acc | int8_acc_clean | same_prediction | changed_of_3421 | fp32_MB | int8_MB | smaller_by |
|---|---|---|---|---|---|---|---|---|
| resnet18 | 0.9923 ± 0.0017 | 0.9914 ± 0.0017 | 0.9917 ± 0.0017 | 0.9991 ± 0.0000 | 3 / 3 / 3 | 44.8 | 11.3 | 3.96x |
| mobilenet_v2 | 0.9917 ± 0.0007 | 0.9908 ± 0.0007 | 0.9911 ± 0.0007 | 0.9979 ± 0.0009 | 4 / 10 / 8 | 9.2 | 2.7 | 3.46x |
| googlenet | 0.9923 ± 0.0009 | 0.9922 ± 0.0009 | 0.9925 ± 0.0009 | 0.9994 ± 0.0003 | 1 / 2 / 3 | 22.6 | 5.9 | 3.86x |
