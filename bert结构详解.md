#### 模型架构
```
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      # 如此重复11层，直到大厦崩塌：）
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
```
参考：[bert代码详解](https://zhuanlan.zhihu.com/p/360988428)

dropout的地方：

<img width="1286" alt="image" src="https://github.com/LilinNK/Graduation-Project/assets/96948927/67ea393b-1d5a-4d44-b2fa-8e2d562d4f11">

BertModel包含复杂的封装和较多的组件。以bert-base为例，主要组件如下：  
总计Dropout出现了1+(1+1+1)x12=37次；  
总计LayerNorm出现了1+(1+1)x12=25次；  
总计dense全连接层出现了(1+1+1)x12+1=37次，并不是每个dense都配了激活函数  

LayerNorm目的是将word-embeddings约束在半径为1的球内，每次都是为了这个目的，在attention后先残差再LayerNorm为的同样是不断修正原词向量但是又不会偏离原向量太远，从而在该向量的一个小区域内形成了一词多义，上下文信息。为啥不能用bacthnorm，因为每次输入的句子可能不一样长，如果做batch会导致偏后的词均值接近于0，且不同句子之间词的差别较大，可能均值方差也存在很大差距，但同一句子之间词的相似度会高一些。

[模型参数估计的例子](https://zhuanlan.zhihu.com/p/144582114)

for some test
