# A.K.U. (Artificial Knowledge Ubiquity)
## what is AKU?
**AKU** シリーズはTransformerアーキテクチャをベースとする日本語に特化した言語モデル群です。\
サイズはそのほとんどを `<1B param` とする予定であり、単一の汎用モデルではなく個別のタスクにフォーカスしたモデルとなります。

**AKU** という名前は、モデルが獲得した知識を、デバイスや環境を問わず至る所で使用してほしいという思いを込めました。

## 1. AKU-d_ms-0.5B-chat-v0.1
### Overview
|                    |                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Architecture       | Decoder-Only Transformer                                                                          |
| Type               | Mistral                                                                                           |
| Parameter          | 519M                                                                                              |
| Language           | ja                                                                                                |
| Training Data      | See [Hugging Face repo](https://huggingface.co/datasets/YukiTomita-CC/AKU-d_ms-0.5B-v0.1_dataset) |
| Training Data Size | 1.56 B tokens                                                                                     |

### Description
- 2024/10/12に公開予定です。
- 会話(雑談)タスクに特化させたモデルです。
