from transformers.data.data_collator import DataCollatorMixin
from datasets import load_dataset
from dataclasses import dataclass
from collections.abc import Callable
from typing import List
from rd_util.Prompts import get_prompt

@dataclass
class DatasetDescriptor:
    huggingface_config: str
    normalize_function: Callable
    type: str
    columns_to_remove: List[str]

class DataCollatorForTokenizedQueries(DataCollatorMixin):
    def __init__(self, tokenizer, padding=True, max_length = None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        queries = [sample["query"] for sample in batch]
        
        # Pad the "queries"
        tokenized_queries = self.tokenizer.pad(
            {"input_ids": queries},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create a new dictionary for the collated batch
        collated_batch = {
            "input_ids": tokenized_queries["input_ids"],
            "attention_mask": tokenized_queries["attention_mask"]
        }
        
        for key in batch[0].keys():
            collated_batch[key] = [sample[key] for sample in batch]

        del collated_batch["query"]

        return collated_batch
    
def prepare_queries(sample, tokenizer, systemprompt, tokenize=False, template_kwargs=None):
    question = sample["question"]
    # The first answer is considered the ground truth
    gt_candidates = sample["gt_candidates"]

    messages = [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": question}
        ]
    if getattr(tokenizer, "chat_template", None):
        query = tokenizer.apply_chat_template(
                        messages,
                        tokenize=tokenize,
                        add_generation_prompt=True,
                        **dict(template_kwargs or {})
                )
    else:
        # Fallback for tokenizers without a built-in chat template, e.g. some Gemma checkpoints.
        query_text = (
            f"<bos><start_of_turn>system\n{systemprompt}<end_of_turn>\n"
            f"<start_of_turn>user\n{question}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        query = tokenizer.encode(query_text, add_special_tokens=False) if tokenize else query_text

    sample_new = {"query": query, "gt_candidates": gt_candidates}
    return sample_new

def _normalize_triviaqa_dataset_map_function(example):
    question = example['question']
    answer = example['answer']['value'] if example['answer']['value'] else ''
    answer_aliases = example['answer']['normalized_aliases'] if example['answer']['value'] else ''
    return {"question": question, "answer": answer, "gt_candidates": answer_aliases, "is_multiple_choice": False}

def _normalize_hotpotqa_dataset_map_function(sample):
    question = sample["question"]
    answer = sample["answer"] if sample.get("answer") else ""
    candidates = [answer] if answer else []
    return {"question": question, "answer": answer, "gt_candidates": candidates, "is_multiple_choice": False}

def _normalize_hotpotqa_context_dataset_map_function(sample):
    """HotpotQA with the distractor passages supplied.

    Must match ``_hotpot_context(..., method="RD")`` in ``csr/data.py``, so RD is
    evaluated on the prompt it was trained on.
    """
    question = sample["question"]
    answer = sample["answer"] if sample.get("answer") else ""
    candidates = [answer] if answer else []

    ctx = sample.get("context", {}) or {}
    parts = []
    for title, sentences in zip(ctx.get("title", []) or [],
                                ctx.get("sentences", []) or []):
        body = " ".join(t.strip() for t in (sentences or []) if t)
        if title or body:
            parts.append(f"Title: {title}\n{body}")
    context = "\n\n".join(parts).strip()

    if context:
        question = f"Context: {context}, Question: {question}"

    return {"question": question, "answer": answer, "gt_candidates": candidates, "is_multiple_choice": False}


datasets = {
    "triviaqa": DatasetDescriptor
    (
        huggingface_config = {"path": "trivia_qa", "name": "unfiltered"}, 
        normalize_function = _normalize_triviaqa_dataset_map_function, 
        type = "open", 
        columns_to_remove = ['question_id', 'question_source', 'entity_pages', 'search_results']
    ),
    "hotpotqa": DatasetDescriptor
    (
        huggingface_config = {"path": "hotpotqa/hotpot_qa", "name": "distractor"},
        normalize_function = _normalize_hotpotqa_dataset_map_function,
        type = "open",
        columns_to_remove = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    ),
    "hotpotqa_context": DatasetDescriptor
    (
        huggingface_config = {"path": "hotpotqa/hotpot_qa", "name": "distractor"},
        normalize_function = _normalize_hotpotqa_context_dataset_map_function,
        type = "context_open",
        columns_to_remove = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    ),
}

def get_dataset_descriptor(dataset):
    try:
        dataset_desc = datasets[dataset]
    except KeyError:
        raise NotImplementedError(f"This dataset has not been implemented: '{dataset}'")
    
    return dataset_desc

def load_prepared_dataset(dataset, split, method, tokenizer, template_kwargs=None,
                          system_prompt=None):
    dataset_descriptor = get_dataset_descriptor(dataset)

    dataset = load_dataset(**dataset_descriptor.huggingface_config, split=split)
    dataset_norm = dataset.map(lambda x: dataset_descriptor.normalize_function(x), remove_columns=dataset_descriptor.columns_to_remove)

    print(dataset_norm[0].keys())
    # train_RD passes method.system_prompt; get_prompt is the upstream fallback
    system_prompt = system_prompt or get_prompt(dataset_descriptor.type)
    dataset = dataset_norm.map(lambda x: prepare_queries(x, tokenizer, system_prompt, tokenize=True,
                                                         template_kwargs=template_kwargs))

    return dataset
