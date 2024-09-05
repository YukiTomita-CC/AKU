import os

import sentencepiece as spm
from transformers import T5Tokenizer


MODEL_PREFIX = "sentencepice_model"
OUTPUT_DIR = "output/custom_tokenizer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_type="unigram",
    model_prefix=MODEL_PREFIX,
    vocab_size=32000,
    accept_language=["ja", "en"],
    character_coverage=0.9995,
    user_defined_symbols=["<CONTEXT>","</CONTEXT>","<ROLE>","</ROLE>", "User", "Aku", "😊", "🥰", "😉", "🤗", "😭", "🤣", "🥺", "💓", "✨"],
    byte_fallback=True,
    add_dummy_prefix=False,
    unk_id=3,
    bos_id=0,
    eos_id=1,
    pad_id=2,
    unk_piece="<UNK>",
    bos_piece="<BOS>",
    eos_piece="<EOS>",
    pad_piece="<PAD>"
    )

tokenizer = T5Tokenizer(
    vocab_file=MODEL_PREFIX+".model",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
    pad_token="<PAD>",
    extra_ids=0,
    legacy=False
)

tokenizer.save_pretrained(OUTPUT_DIR)
