import numpy as np

N = 3
T_enc = 4
T_dec = 7
H = 5

hs_enc = np.random.randn(N,T_enc,H)
print(hs_enc.shape)

hs_dec = np.random.randn(N,T_dec,H)
print(hs_dec.shape)

# T個のAttentionレイヤの受け皿を初期化
layers = []

# T個のAttentionの重みの受け皿を初期化
attention_weights = []

# T個のコンテキストの受け皿を初期化
cs = np.empty((N, T_dec, H))

# Time Attentionレイヤの処理
# for t in range(T_dec):
#     # t番目のAttentionレイヤを作成
#     layer = Attention()
    
#     # t番目のコンテキストを計算
#     cs[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
    
#     # t番目のAttentionレイヤを格納
#     layers.append(layer)
    
#     # t番目のAttentionの重みを格納
#     attention_weights.append(layer.attention_weight)
    
# T個のコンテキストを確認
print(np.round(cs[0], 2)) # 0番目のコンテキスト
print(cs.shape)

print(np.round(attention_weights[0], 2))
print(np.sum(attention_weights[0], axis=1))
print(attention_weights[0].shape)