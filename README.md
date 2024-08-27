# A.K.U. (Artificial Knowledge Ubiquity)
## what is AKU?
**AKU** シリーズはTransformerアーキテクチャをベースとする日本語に特化した言語モデル群です。\
サイズはそのほとんどを `<1B param` とする予定であり、単一の汎用モデルではなく個別のタスクにフォーカスしたモデルとなります。

**AKU** という名前は、モデルが獲得した知識をデバイスを問わず、至る所で使用してほしいという思いを込めました。

## 1. AKU-d_ms-0.5B-v0.1
### Overview
|                    |                                                                   |
| ------------------ | ----------------------------------------------------------------- |
| Architecture       | Decoder-Only Transformer                                          |
| Type               | Mistral                                                           |
| Parameter          | 0.5B                                                              |
| Language           | ja                                                                |
| Training Data      | Wikipedia<br>mC4<br>CC100<br>OSCAR<br>StarCoder<br>Synthetic Data |
| Training Data Size | -- B tokens                                                       |

### Description
- Decoder-Onlyの基盤モデルです。
- 2024/10/12に公開予定です。

## 2. AKU-d_ms-0.5B-chat-v0.1
### Overview
[AKU-d_ms-0.5B-v0.1](## 1. AKU-d_ms-0.5B-v0.1)を対話システムのためにファインチューニングしたモデルです。

### Description
- 2024/10/12に公開予定です。
