from functools import partial

import transformers.models.auto.tokenization_auto
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase

from open_instruct.finetune import encode_with_messages_format


def print_red(skk):
    print("\033[91m {}\033[00m".format(skk))


def print_green(skk):
    print("\033[92m {}\033[00m".format(skk))


def __preprocess(example, tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def __filter(example):
    return example["n_labels"] > 0


def transform_dataset(dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
                      tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
    dataset = dataset.map(
        partial(__preprocess, tokenizer=tokenizer, max_seq_len=max_seq_len),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=1,  # type: ignore
    )

    print("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(__filter, batched=False, num_proc=1)  # type: ignore
    print(f"Filtered out {n - len(dataset):,d} examples")

    print("Counting tokens...")
    total_tokens = 0
    for ex in dataset:
        assert len(ex["input_ids"]) == max_seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    print(f"Total tokens: {total_tokens:,d}")

    return dataset


def prepare_dataset(dataset: [Dataset | DatasetDict], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> DataLoader:
    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=2048,
        add_bos=False,
    )

    lm_dataset = dataset.map(
        encode_function,
        batched=False,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in dataset.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_dataset.set_format(type="pt")
    lm_dataset = lm_dataset.filter(lambda example: (example["labels"] != -100).any())

    train_dataloader = DataLoader(
        lm_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=1,
    )

    return train_dataloader
