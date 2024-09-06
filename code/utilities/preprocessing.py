from functools import partial

from datasets import DatasetDict, Dataset
from open_instruct.finetune import encode_with_messages_format
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase

from dotenv import load_dotenv

load_dotenv()

__client = OpenAI()

__paraphrasing_system_prompt = """
You are a paraphrasing expert who is specialized in rewriting text (questions, statements, etc.) without altering the content. 
Keep in mind, that the meaning must not change after the paraphrasing. 
Just output the paraphrased text without any additional information.
"""

def paraphrase_input(input: str):
    response = __client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": __paraphrasing_system_prompt},
            {"role": "user", "content": input}
        ],
        seed=42,
        temperature=1,
    )

    return response.choices[0].message.content


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
