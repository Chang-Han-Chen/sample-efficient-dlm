import pickle

import torch

import prepare


class DummyEnc:
    n_vocab = 8192

    def encode_single_token(self, token):
        return {
            prepare.BOS_TOKEN: 0,
            prepare.MASK_TOKEN: 1,
        }[token]

    def encode_ordinary(self, text):
        return [2, 3, 4]

    def encode_ordinary_batch(self, texts, num_threads=8):
        return [self.encode_ordinary(text) for text in texts]

    def decode(self, ids):
        return "decoded"


def test_tokenizer_from_directory_reads_tokenizer_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    data_dir = cache_dir / "shards"
    tokenizer_dir = cache_dir / "tokenizer"
    data_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)

    with open(tokenizer_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(DummyEnc(), f)

    tok = prepare.Tokenizer.from_directory(str(tokenizer_dir))

    assert tok.vocab_size == 8192
    assert tok.mask_token_id == 1
    assert tok.get_bos_token_id() == 0


def test_make_dataloader_supports_runtime_and_legacy_calls(monkeypatch):
    tok = prepare.Tokenizer(DummyEnc())

    def fake_document_batches(split, tokenizer_batch_size=128):
        while True:
            yield ["hello", "world", "tokens", "more"], 7

    real_empty = torch.empty

    def fake_empty(*size, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs.pop("device", None)
        return real_empty(*size, **kwargs)

    monkeypatch.setattr(prepare, "_document_batches", fake_document_batches)
    monkeypatch.setattr(prepare.torch, "empty", fake_empty)

    runtime_dl = prepare.make_dataloader(tok, 2, 4, "train")
    inputs, targets, epoch = next(runtime_dl)
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert epoch == 7

    legacy_dl = prepare.make_dataloader(
        tok, batch_size=2, seq_len=4, split="train"
    )
    batch = next(legacy_dl)
    assert batch.shape == (2, 4)
    assert batch.dtype == torch.long
