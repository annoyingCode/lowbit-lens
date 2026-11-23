FP32 baselines on BloodMNIST 224px, seeds [0, 1, 2]. Test accuracy is for the checkpoint with the best validation accuracy; hyperparameters were not tuned (see docs/DECISIONS.md). test_acc uses the official test set; test_acc_clean leaves out the 6 test images that are exact copies of training or validation images (results/tables/duplicate_images.csv). Values are mean ± sample SD over seeds.

| model | params_M | seeds | val_acc | test_acc | test_acc_clean |
|---|---|---|---|---|---|
| resnet18 | 11.18 | 3 | 0.9936 ± 0.0012 | 0.9923 ± 0.0017 | 0.9926 ± 0.0017 |
| mobilenet_v2 | 2.23 | 3 | 0.9942 ± 0.0000 | 0.9917 ± 0.0007 | 0.9920 ± 0.0007 |
| googlenet | 5.61 | 3 | 0.9945 ± 0.0009 | 0.9923 ± 0.0009 | 0.9926 ± 0.0009 |
