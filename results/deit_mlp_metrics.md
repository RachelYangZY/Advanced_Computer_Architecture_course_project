| config | retired | MACs/instr | (ld+st)/MAC | instr/output | compressed | multiply | xqmac8 | xrequant | load | store | branch/jump | int_alu/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 11651515 | 0.0253 | 20.1903 | 12137.0 | 4479456 | 886656 | 0 | 0 | 4758740 | 1195609 | 306273 | 4504237 |
| B1 | 2111029 | 0.1397 | 2.0306 | 2199.0 | 903455 | 295872 | 0 | 0 | 596160 | 2689 | 305301 | 911007 |
| X1 | 328109 | 0.8988 | 0.2806 | 341.8 | 237969 | 0 | 36864 | 960 | 80064 | 2689 | 42453 | 165079 |

Per-function retired instructions:

| function | B0 | B1 | X1 |
|---|---:|---:|---:|
| _start | 3 | 3 | 3 |
| int8_gemm_reference | 11531418 | 2072140 | 0 |
| int8_gemm_x1 | 0 | 0 | 304604 |
| main | 18288 | 5804 | 5804 |
| requantize_i32_to_i8 | 46080 | 14400 | 0 |
| stage_gelu | 19214 | 6152 | 6152 |
| stage_requant_fc1 | 29200 | 10009 | 9229 |
| stage_requant_fc2 | 7312 | 2521 | 2317 |

B1 vs B0: 81.88% fewer (5.52x)
X1 vs B0: 97.18% fewer (35.51x)
X1 vs B1: 84.46% fewer (6.43x)
