# Task 3 hand calculations (copy onto paper, photograph, put photos in lab03/results/)

Convention: FLOPs = 2 x MACs. Biases, activations, pooling, LayerNorm, softmax ignored.
thop reports MACs, so compare hand FLOPs with 2 x thop MACs.

## A. Parameters (check against torchinfo)
MLP: 784*256+256 = 200,960 | 256*128+128 = 32,896 | 128*10+10 = 1,290  -> 235,146
CNN: conv1 1*16*9+16 = 160 | conv2 16*32*9+32 = 4,640 | fc 1568*10+10 = 15,690 -> 20,490
Attention: patch proj 49*64+64 = 3,200 | pos-emb 16*64 = 1,024
  per encoder layer: in_proj 3*64*64+3*64 = 12,480 | out_proj 64*64+64 = 4,160
                     FFN 64*128+128 = 8,320 and 128*64+64 = 8,256 | 2 LayerNorms 2*128 = 256 -> 33,472
  2 layers = 66,944 | head 64*10+10 = 650   -> 71,818
NOTE: none of the three models is in the 100k-150k target (235k / 20k / 72k). The manual's architectures
are fixed, so use them as given and state the discrepancy in the report.

## B. One layer of each type (required)
Linear 784->256:   2 * 784 * 256 = 401,408
Conv2d(1,16,3,pad=1) on 28x28 (H_out = W_out = 28):  2 * 3 * 3 * 1 * 16 * 28 * 28 = 225,792
Attention layer (seq=16, d_model=64, ff=128):
  QKV projections      2 * 16 * 64 * (3*64)  = 393,216
  scores  QK^T         2 * 16^2 * 64         =  32,768
  weights x V          2 * 16^2 * 64         =  32,768
  output projection    2 * 16 * 64 * 64      = 131,072
  FFN linear 1         2 * 16 * 64 * 128     = 262,144
  FFN linear 2         2 * 16 * 128 * 64     = 262,144
  one encoder layer total                    = 1,114,112

## C. Whole-model totals (to compare with thop)
MLP:   401,408 + 65,536 + 2,560                              =   469,504
CNN:   conv1 225,792 + conv2 (2*9*16*32*14*14) 1,806,336 + fc 31,360 = 2,063,488
Attention: patch proj (2*49*64*16) 100,352 + 2 * 1,114,112 + head 1,280 = 2,329,856

## D. Expected gap with thop (explain in the report)
- thop counts MACs, not FLOPs: multiply by 2.
- thop counts nn.Linear / nn.Conv2d but has no rule for nn.MultiheadAttention, so the Q/K/V and output
  projections and the QK^T and AV matmuls are missed. Expect thop's Attention number to be well below 2.33M.
- thop may add small terms (ReLU, LayerNorm, pooling) that the hand formula ignores.
Check your actual thop numbers in results/cost.csv before writing the explanation.