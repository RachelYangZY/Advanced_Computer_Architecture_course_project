# Accuracy sweep (ImageNet-1k validation subset)

1000 images from ILSVRC/imagenet-1k validation, 224x224 bicubic, model-card normalization.

| Configuration | Images | Top-1 % | Top-5 % | Images with logits != reference | Top-1 agrees with reference |
|---|---|---|---|---|---|
| FP32 DeiT-Tiny | 1000 | 70.7 | 90.3 | - | - |
| TFLite W8A8 (quantized model) | 1000 | 65.7 | 88.1 | - | - |
| B0 generated C, vs TFLite | 1000 | 65.7 | 88.1 | 0 (max abs diff 0) | 100.0% |
| B1 generated C, vs TFLite | 1000 | 65.7 | 88.1 | 0 (max abs diff 0) | 100.0% |
| B1 vs B0 (optimization effect) | 1000 | 65.7 | 88.1 | 0 (max abs diff 0) | 100.0% |
| B1 on Spike, vs B0 | 100 | - | - | 0 | - |
| X1MAC on Spike, vs B0 | 100 | - | - | 0 | - |
| X1 on Spike, vs B0 | 100 | - | - | 0 | - |

Reference column: TFLite for the B0/B1 rows, B0 for the B1-vs-B0 and Spike rows.
