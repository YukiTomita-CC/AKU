from transformers import T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained("models/inc_batch_2epoch")

# print("INFO: Check the special tokens")
# check_tokens = {
#     "bos_token": tokenizer.bos_token,
#     "eos_token": tokenizer.eos_token,
#     "pad_token": tokenizer.pad_token,
#     "unk_token": tokenizer.unk_token,
#     "context_begin_token": "<CONTEXT>",
#     "context_end_token": "</CONTEXT>",
#     "role_begin_token": "<ROLE>",
#     "role_end_token": "</ROLE>",
#     "line_break_token": "\n"
# }

# print (f"| Token ID | {'Token Name':<19} | {'String':<10} |")
# print("|----------|---------------------|------------|")
# for token_name, token_string in check_tokens.items():
#     print(f"| {tokenizer.convert_tokens_to_ids(token_string):<8} | {token_name:<19} | {token_string:<10} |")
# print()


# print("INFO: Check the vocabulary size")
# print(f"Vocabulary size: {tokenizer.vocab_size}")
# print()


print("INFO: Check the encoding and decoding of some examples")
examples = [
    # "Hello, my name is John.",
    # "I am a student.",
    # "こんにちは、私の名前はAkuです。",
    # "私は学生です。",
    # "<BOS><CONTEXT>Akuは社会人として働いていて、趣味は音楽です。</CONTEXT>\n\n<ROLE>User</ROLE>\n趣味は何ですか?\n<ROLE>Aku</ROLE>\nこんにちは。音楽が好きです。<EOS>",
    "<CONTEXT>*UserとAkuは家で二人でのんびりしている</CONTEXT>\n\n<ROLE>User</ROLE>\n来週の土日でプチ旅行しない? <ATTR> likability: 5 mood: 5 </ATTR>\n<ROLE>Aku</ROLE>\n"
]

for example in examples:
    print(f"Example: {example}")
    encoded_example = tokenizer.encode(example, add_special_tokens=False)
    print(f"Encoded: {encoded_example}")
    print(f"Decoded: ", end="")
    for token_id in encoded_example:
        print(tokenizer.decode([token_id]), end="/")
    print("")
    print("-"*50)


# print("INFO: Check the encoding and decoding of some emojis")
# emoji_list = ["😊", "🥰", "😉", "🤗", "😭", "🤣", "🥺", "💓", "✨"]

# for emoji in emoji_list:
#     print(f"Emoji: {emoji}")
#     encoded_emoji = tokenizer.encode(emoji)
#     print(f"Encoded: {encoded_emoji}")
#     print(f"Decoded: {tokenizer.decode(encoded_emoji)}")
#     print("-"*50)

print("INFO: Check the decoding of some examples")
examples = [
    [308, 832, 285, 16687, 2011, 1196, 19136, 328, 370, 6324, 5519, 1376, 11, 3],
    [308, 27403, 342, 438, 1326, 286, 5571, 299, 388, 1826, 1376, 3]
]

for example in examples:
    print(f"Example: {example}")
    print(f"Decoded: {tokenizer.decode(example, skip_special_tokens=True)}")
    print("-"*50)