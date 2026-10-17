| config | retired | MACs/instr | (ld+st)/MAC | instr/output | compressed | multiply | xqmac8 | xrequant | load | store | branch/jump | int_alu/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1926706 | 0.0255 | 20.0765 | 30104.8 | 741248 | 147584 | 0 | 0 | 789125 | 197675 | 49979 | 742343 |
| B1 | 347098 | 0.1416 | 2.0104 | 5423.4 | 148774 | 49216 | 0 | 0 | 98688 | 129 | 49886 | 149179 |
| X1 | 51310 | 0.9579 | 0.2604 | 801.7 | 38071 | 0 | 6144 | 64 | 12672 | 129 | 6558 | 25743 |

Per-function retired instructions:

| function | B0 | B1 | X1 |
|---|---:|---:|---:|
| _start | 3 | 3 | 3 |
| int8_gemm_reference | 1919076 | 344684 | 0 |
| int8_gemm_x1 | 0 | 0 | 49929 |
| main | 4555 | 1451 | 1378 |
| requantize_i32_to_i8 | 3072 | 960 | 0 |

B1 vs B0: 81.98% fewer (5.55x)
X1 vs B0: 97.34% fewer (37.55x)
X1 vs B1: 85.22% fewer (6.76x)
