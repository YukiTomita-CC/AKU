# AKU Chat Template
## Template
```
<BOS><ROLE>User</ROLE>
{{user_1}}
<ROLE>Aku</ROLE>
{{aku_1}}
<ROLE>User</ROLE>
{{user_2}}
<ROLE>Aku</ROLE>
{{aku_2}}
<ROLE>User</ROLE>
{{user_3}} <ATTR> likability: {{1 ~ 5}} mood: {{1 ~ 5}} </ATTR>
<ROLE>Aku</ROLE>

```

## Reason ja
- 単純に会話文だけでなく、人物の特徴や過去の対話履歴などのコンテキストを含めたかった
- 加えて、人らしさを出すために好感度と機嫌をSteerLM形式で設定できるようにした
- 必要なのは重要な情報にモデルがAttentionしやすいテンプレートがベストだと思った
  - XMLのタグのように重要な情報の始まりと終わりを明示することで、タグの中身にAttentionしやすいのでは？と考えている
  - ここではタグの名前は意味をもたない。なぜなら、`<ROLE>`などをTokenizerのspecial tokenに設定しているので、モデルには「role」という単語の意味は伝わっていない
  - しかし、学習の過程で`<ROLE>`と`</ROLE>`に挟まれたトークン列を人物名と認識できるようになるはず
  - 同様にただの平文よりもcontextや、`</ROLE>\n`の後のトークン列にAttentionしやすくなっているはず

## Reason en
- I wanted to include not just the conversation text, but also context such as character traits and past dialogue history
- Additionally, to add human-like qualities, I made it possible to set likability and mood in SteerLM format
- I thought the best template would be one that allows the model to easily attend to important information
  - By explicitly marking the beginning and end of important information with XML-like tags, I believe it becomes easier for the model to attend to the content within the tags
  - The names of the tags themselves don't carry meaning here. This is because we've set tags like `<ROLE>` as special tokens in the tokenizer, so the model doesn't understand the word "role" as such
  - However, through the learning process, the model should be able to recognize that the token sequence between `<ROLE>` and `</ROLE>` represents a person's name
  - Similarly, it should be easier for the model to attend to the context and the token sequence after `</ROLE>\n` compared to plain text
