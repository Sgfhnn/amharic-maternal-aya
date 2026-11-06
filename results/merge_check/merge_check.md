# Merge check: merged v4c model vs v4c predictions

First 16 test rows, greedy, max 256 new tokens, fp16, Tesla T4.

| Pair | Identical answers | Mean chrF | Min chrF |
|---|---|---|---|
| merged vs stored | 13/16 | 97.71 | 75.36 |
| adapter vs stored | 13/16 | 97.71 | 75.36 |
| merged vs adapter | 16/16 | 100.0 | 100.0 |

`stored` = the original v4c eval (all 550 rows in length-sorted batches of 16, so the first 16 rows sat in different batches with different padding); `adapter` = the same model loaded the way that eval loaded it (merged v3 SFT + v4c adapter merged on GPU) on this 16-row batch; `merged` = the single uploaded fp16 model.
