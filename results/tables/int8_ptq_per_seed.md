Real INT8 post-training quantization (512 calibration images from the training set, PyTorch's standard onednn recipe). FP32 and INT8 are both evaluated on the CPU on the official test set (3421 images); int8_acc_clean leaves out the 6 duplicated test images. same_prediction = share of test images where INT8 predicts the same class as FP32; changed = number of test images whose prediction changed. Sizes are saved weights in MB (10^6 bytes).

| model | seed | fp32_acc | int8_acc | int8_acc_clean | same_prediction | changed | right_to_wrong | wrong_to_right | fp32_MB | int8_MB |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet18 | 0 | 0.9930 | 0.9921 | 0.9924 | 0.9991 | 3 | 3 | 0 | 44.8 | 11.3 |
| mobilenet_v2 | 0 | 0.9924 | 0.9912 | 0.9915 | 0.9988 | 4 | 4 | 0 | 9.2 | 2.7 |
| googlenet | 0 | 0.9915 | 0.9912 | 0.9915 | 0.9997 | 1 | 1 | 0 | 22.6 | 5.9 |
| resnet18 | 1 | 0.9936 | 0.9927 | 0.9930 | 0.9991 | 3 | 3 | 0 | 44.8 | 11.3 |
| mobilenet_v2 | 1 | 0.9909 | 0.9901 | 0.9903 | 0.9971 | 10 | 6 | 3 | 9.2 | 2.7 |
| googlenet | 1 | 0.9933 | 0.9930 | 0.9933 | 0.9994 | 2 | 1 | 0 | 22.6 | 5.9 |
| resnet18 | 2 | 0.9904 | 0.9895 | 0.9898 | 0.9991 | 3 | 3 | 0 | 44.8 | 11.3 |
| mobilenet_v2 | 2 | 0.9918 | 0.9912 | 0.9915 | 0.9977 | 8 | 5 | 3 | 9.2 | 2.7 |
| googlenet | 2 | 0.9921 | 0.9924 | 0.9927 | 0.9991 | 3 | 1 | 2 | 22.6 | 5.9 |
