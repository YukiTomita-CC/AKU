import os

import sentencepiece as spm
from transformers import LlamaTokenizer, T5Tokenizer


def train_tokenizer(corpus_file: str="corpus.txt", save_as_llama_tokenizer: bool=True, save_as_t5_tokenizer: bool=True):
    MODEL_PREFIX = "models/sentencepiece_model"
    OUTPUT_DIR_1 = "models/custom_llama_tokenizer"
    OUTPUT_DIR_2 = "models/custom_t5_tokenizer"

    os.makedirs("models", exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_type="unigram",
        model_prefix=MODEL_PREFIX,
        vocab_size=32000,
        accept_language=["ja", "en"],
        character_coverage=0.9995,
        user_defined_symbols=user_defined_symbols,
        byte_fallback=True,
        add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        unk_id=3,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_piece="<UNK>",
        bos_piece="<BOS>",
        eos_piece="<EOS>",
        pad_piece="<PAD>"
        )

    if save_as_llama_tokenizer:
        i = 1
        while os.path.exists(OUTPUT_DIR_1):
            OUTPUT_DIR_1 = f"models/custom_llama_tokenizer_{i}"
            i += 1
        os.makedirs(OUTPUT_DIR_1)

        tokenizer1 = LlamaTokenizer(
            vocab_file=MODEL_PREFIX+".model",
            unk_token="<UNK>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            legacy=False
        )

        tokenizer1.save_pretrained(OUTPUT_DIR_1)

    if save_as_t5_tokenizer:
        j = 1
        while os.path.exists(OUTPUT_DIR_2):
            OUTPUT_DIR_2 = f"models/custom_t5_tokenizer_{j}"
            j += 1
        os.makedirs(OUTPUT_DIR_2)
        
        tokenizer2 = T5Tokenizer(
            vocab_file=MODEL_PREFIX+".model",
            unk_token="<UNK>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            extra_ids=0,
            legacy=False
        )

        tokenizer2.save_pretrained(OUTPUT_DIR_2)


if __name__ == "__main__":
    user_defined_symbols = [
        "<CONTEXT>",
        "</CONTEXT>",
        "<ROLE>",
        "</ROLE>",
        "User",
        "Aku",
        "😊",
        "🥰",
        "😉",
        "🤗",
        "😭",
        "🤣",
        "😆",
        "🙄",
        "😡",
        "😲",
        "😳",
        "😔",
        "😇",
        "😙",
        "🥳",
        "🤔",
        "🥺",
        "🥱",
        "💓",
        "✨",
        "<ATTR>",
        "</ATTR>",
        "likability",
        "mood",
        "\n",
        " ",
        "<SEP>"
    ]

    train_tokenizer(save_as_t5_tokenizer=False)
