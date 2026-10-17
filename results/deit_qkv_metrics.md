| config | retired | MACs/instr | (ld+st)/MAC | instr/output | compressed | multiply | xqmac8 | xrequant | load | store | branch/jump | int_alu/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 489010 | 0.0251 | 20.3060 | 7640.8 | 188288 | 36992 | 0 | 0 | 199301 | 50219 | 13115 | 189383 |
| B1 | 89050 | 0.1380 | 2.0417 | 1391.4 | 38182 | 12352 | 0 | 0 | 24960 | 129 | 13022 | 38587 |
| X1 | 14446 | 0.8506 | 0.2917 | 225.7 | 10424 | 0 | 1536 | 64 | 3456 | 129 | 1950 | 7311 |

Per-function retired instructions:

| function | B0 | B1 | X1 |
|---|---:|---:|---:|
| _start | 3 | 3 | 3 |
| int8_gemm_reference | 481380 | 86636 | 0 |
| int8_gemm_x1 | 0 | 0 | 13065 |
| main | 4555 | 1451 | 1378 |
| requantize_i32_to_i8 | 3072 | 960 | 0 |

B1 vs B0: 81.79% fewer (5.49x)
X1 vs B0: 97.05% fewer (33.85x)
X1 vs B1: 83.78% fewer (6.16x)
