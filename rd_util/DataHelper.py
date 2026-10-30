from transformers.data.data_collator import DataCollatorMixin
from datasets import load_dataset
import numpy as np
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
    # Optional: map a requested split name (e.g. "validation") to the dataset's
    # actual split name (e.g. "test"). Lets datasets that don't ship a
    # "validation" split (like big-math-digits) work without changing callers.
    split_map: dict = None

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

def shuffle_options_medqa(dataset):
    options = [sample["options"] for sample in dataset]

    def augment_options(sample, all_samples):
        idx = np.random.choice(np.arange(len(all_samples)))
        sample["options"] = all_samples[idx]
        return sample
    dataset = dataset.map(lambda x: augment_options(x, options))
    return dataset

def _normalize_triviaqa_dataset_map_function(example):
    question = example['question']
    answer = example['answer']['value'] if example['answer']['value'] else ''
    answer_aliases = example['answer']['normalized_aliases'] if example['answer']['value'] else ''
    return {"question": question, "answer": answer, "gt_candidates": answer_aliases, "is_multiple_choice": False}

def _normalize_medqa_dataset_map(sample):
    question = sample["question"]
    options = [f"{d['key']}: {d['value']}" for d in sample["options"]]
    options = ", ".join(options)
    question = f"{question} {options}"
    correct_answer = sample['answer_idx']

    candidates = [correct_answer]

    return {"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": True}

def _normalize_commonsenseqa_dataset_map_function(sample):
    question = sample["question"]
    options = [f"{l}: {t}" for l, t in zip(sample["choices"]["label"], sample["choices"]["text"])]
    options = ", ".join(options)
    question = f"{question} {options}"

    correct_answer = sample['answerKey']
    candidates = [correct_answer]

    return {"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": True}

def _normalize_truthfulqa_generation_dataset_map_function(sample):
    question = sample["question"]
    correct_answer = sample["best_answer"]
    candidates = sample["correct_answers"]

    return {"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": False}

def _normalize_truthfulqa_mc_dataset_map_function(sample):
    question = sample["question"]
    
    answer_keys = ["A", "B", "C", "D", "E"]
    options = [f"{key}: {c}" for key, c in zip(answer_keys, sample["mc1_targets"]["choices"])]
    options = ", ".join(options)
    question = f"{question} {options}"

    correct_answer = answer_keys[sample["mc1_targets"]["labels"].index(1)]
    candidates =  [correct_answer]

    return {"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": True}

def _normalize_squad_dataset_map_function(sample):
    question = sample["question"]
    context = sample["context"]
    correct_answer = sample["answers"]["text"][0] if len(sample["answers"]["text"][0]) > 0 else ""
    candidates = sample["answers"]["text"]

    question = f"Context: {context}, Question: {question}"

    return {"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": False}

def _normalize_qampari(sample):
    question = sample["question"]
    answer = ", ".join(sample["answers"])
    answer_aliases = []
    for a in sample["answer_list"]["aliases"]:
        answer_aliases += a

    return {"question": question, "answer": answer, "gt_candidates": answer_aliases, "is_multiple_choice": False}

def _normalize_hotpotqa_dataset_map_function(sample):
    question = sample["question"]
    answer = sample["answer"] if sample.get("answer") else ""
    candidates = [answer] if answer else []
    return {"question": question, "answer": answer, "gt_candidates": candidates, "is_multiple_choice": False}

def _normalize_bigmath_dataset_map_function(sample):
    # big-math-digits row schema: {problem, answer, source, domain, llama8b_solve_rate}
    question = sample.get("problem", "")
    answer = str(sample.get("answer", "")).strip()
    candidates = [answer] if answer else []
    return {"question": question, "answer": answer, "gt_candidates": candidates, "is_multiple_choice": False}

def _normalize_hotpotqa_context_dataset_map_function(sample):
    """HotpotQA with the distractor passages supplied.

    The passage layout is ``_format_hotpot_context`` from ``csr/data.py`` and the
    question phrasing is RD's own context phrasing (the same
    ``f"Context: {context}, Question: {question}"`` that
    ``_normalize_squad_dataset_map_function`` uses, which is what
    ``_hotpot_context(..., method="RD")`` renders at evaluation time). Training
    and evaluation therefore build the prompt identically -- an RD checkpoint
    only behaves as trained if it sees its own phrasing.
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

### ADD NORMALIZE FUNCTION HERE:


datasets = {
    "triviaqa": DatasetDescriptor
    (
        huggingface_config = {"path": "trivia_qa", "name": "unfiltered"}, 
        normalize_function = _normalize_triviaqa_dataset_map_function, 
        type = "open", 
        columns_to_remove = ['question_id', 'question_source', 'entity_pages', 'search_results']
    ),
    "medqa": DatasetDescriptor
    (
        huggingface_config = {"path": "bigbio/med_qa", "trust_remote_code": True}, 
        normalize_function = _normalize_medqa_dataset_map, 
        type = "mc", 
        columns_to_remove = ["meta_info", "answer_idx", "options"]
    ),
    "commonsenseqa": DatasetDescriptor
    (
        huggingface_config = {"path": "tau/commonsense_qa"}, 
        normalize_function = _normalize_commonsenseqa_dataset_map_function, 
        type = "mc", 
        columns_to_remove = ["id", "question", "choices", "answerKey", "question_concept"]
    ),
    "squad": DatasetDescriptor
    (
        huggingface_config = {"path": "rajpurkar/squad"}, 
        normalize_function = _normalize_squad_dataset_map_function, 
        type = "context_open", 
        columns_to_remove = ["id", "title", "context", "answers", "question"]
    ),
    "truthfulqa_generation": DatasetDescriptor
    (
        huggingface_config = {"path": "truthfulqa/truthful_qa", "name": "generation"}, 
        normalize_function = _normalize_truthfulqa_generation_dataset_map_function, 
        type = "open", 
        columns_to_remove = ["type", "category", "best_answer", "correct_answers", "incorrect_answers", "source"]
    ),
    "truthfulqa_mc": DatasetDescriptor
    (
        huggingface_config = {"path": "truthfulqa/truthful_qa", "name": "mutliple_choice"}, 
        normalize_function = _normalize_truthfulqa_mc_dataset_map_function, 
        type = "mc", 
        columns_to_remove = ["mc1_targets", "mc2_targets"]
    ),
    "qampari": DatasetDescriptor
    (
        huggingface_config = {"path": "iohadrubin/qampari", "name": "reranking_bm25"}, 
        normalize_function = _normalize_qampari, 
        type = "single_answer", 
        columns_to_remove = ['question_text', 'answer_list', 'qid', 'question', 'answers', 'positive_ctxs', 'hard_negative_ctxs']
    ),
    "hotpotqa": DatasetDescriptor
    (
        huggingface_config = {"path": "hotpotqa/hotpot_qa", "name": "distractor"},
        normalize_function = _normalize_hotpotqa_dataset_map_function,
        type = "open",
        columns_to_remove = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    ),
    "bigmath": DatasetDescriptor
    (
        huggingface_config = {"path": "mehuldamani/big-math-digits"},
        normalize_function = _normalize_bigmath_dataset_map_function,
        type = "open",  # uses Prompts["open"] -> "Answer: <answer>, Confidence: <0..10>"
        columns_to_remove = ["problem", "answer", "source", "domain", "llama8b_solve_rate"],
        # big-math-digits has only train/test, no validation split — remap.
        split_map = {"validation": "test"},
    ),
        "hotpotqa_context": DatasetDescriptor
    (
        huggingface_config = {"path": "hotpotqa/hotpot_qa", "name": "distractor"},
        normalize_function = _normalize_hotpotqa_context_dataset_map_function,
        # "context_open" is the prompt that tells the model a context is
        # supplied; "hotpotqa" above uses "open" because it withholds it.
        type = "context_open",
        columns_to_remove = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    ),
    ### ADD DATASET DESCRIPTOR HERE:

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

    # Allow datasets to remap requested split names (e.g. bigmath has no
    # "validation" split, only "test").
    if dataset_descriptor.split_map and split in dataset_descriptor.split_map:
        split = dataset_descriptor.split_map[split]

    dataset = load_dataset(**dataset_descriptor.huggingface_config, split=split)
    dataset_norm = dataset.map(lambda x: dataset_descriptor.normalize_function(x), remove_columns=dataset_descriptor.columns_to_remove)

    print(dataset_norm[0].keys())
    # one source for train and eval; get_prompt picks a different scale per type
    system_prompt = system_prompt or get_prompt(dataset_descriptor.type)
    dataset = dataset_norm.map(lambda x: prepare_queries(x, tokenizer, system_prompt, tokenize=True,
                                                         template_kwargs=template_kwargs))

    return dataset
