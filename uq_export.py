# %% [markdown]
# ### Utilities

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import scipy.special

class EntailmentDeberta:
    def __init__(self, device="cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def check_implication_batch(self, text1_list, text2_list, batch_size=32):
        results = []
        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i+batch_size]
            batch_text2 = text2_list[i:i+batch_size]
            
            inputs = self.tokenizer(batch_text1, batch_text2, return_tensors="pt", 
                                  padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().tolist()
                results.extend(predictions)
        return results
    
    def check_implication_deg_e_batch(self, text1_list, text2_list, batch_size=32):
        results = []
        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i+batch_size]
            batch_text2 = text2_list[i:i+batch_size]
            inputs = self.tokenizer(batch_text1, batch_text2, return_tensors="pt", 
                                padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                entailment_probs = probs[:, 2].cpu().tolist()
                results.extend(entailment_probs)
        return results

def load_data(jsonl_path):
    """
    Load JSONL data where each line has format:
    {
        "question": "...",
        "samples": ["answer1", "answer2", ...],
        "probs": [[logprob_tokens1], [logprob_tokens2], ...]  # optional
    }
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data

def save_results(data, output_path):
    """Save results back to JSONL"""
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved results to {output_path}")

# %% [markdown]
# ### Semantic Entropy

# %%
def get_semantic_ids_batched(strings_list, model, strict_entailment=False):
    n = len(strings_list)
    if n <= 1:
        return list(range(n))
    
    text1_list, text2_list = [], []
    pair_map = {}
    
    for i in range(n):
        for j in range(i+1, n):
            pair_map[(i, j)] = len(text1_list) // 2
            text1_list.extend([strings_list[i], strings_list[j]])
            text2_list.extend([strings_list[j], strings_list[i]])
    
    implications = model.check_implication_batch(text1_list, text2_list)
    
    semantic_set_ids = [-1] * n
    next_id = 0
    
    for i in range(n):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            
            for j in range(i+1, n):
                if semantic_set_ids[j] == -1:
                    pair_idx = pair_map[(i, j)]
                    implication_1 = implications[pair_idx * 2]
                    implication_2 = implications[pair_idx * 2 + 1]
                    
                    if strict_entailment:
                        semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
                    else:
                        implications_pair = [implication_1, implication_2]
                        semantically_equivalent = (0 not in implications_pair) and ([1, 1] != implications_pair)
                    
                    if semantically_equivalent:
                        semantic_set_ids[j] = next_id
            
            next_id += 1
    
    return semantic_set_ids

def cluster_assignment_entropy(semantic_ids):
    n_generations = len(semantic_ids)
    if n_generations <= 1:
        return 0.0
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    entropy = - (probabilities * np.log(probabilities)).sum()
    max_entropy = np.log(n_generations)
    return entropy / max_entropy

def compute_semantic_entropy(samples, model):
    """Compute semantic entropy for a list of samples"""
    if len(samples) == 0:
        return 0.0
    semantic_ids = get_semantic_ids_batched(samples, model)
    return cluster_assignment_entropy(semantic_ids)

# %% [markdown]
# ### Degree of Disagreement

# %%
def compute_pairwise_entailment_batch(question, answers, entailment_model):
    N = len(answers)
    if N <= 1:
        return np.zeros((N, N))
    
    text1_list, text2_list = [], []
    indices = []
    
    for i in range(N):
        for j in range(N):
            if i != j:
                premise = question + " " + answers[i]
                hypothesis = question + " " + answers[j]
                text1_list.append(premise)
                text2_list.append(hypothesis)
                indices.append((i, j))
    
    probs = entailment_model.check_implication_deg_e_batch(text1_list, text2_list)
    
    scores = np.zeros((N, N))
    for idx, (i, j) in enumerate(indices):
        scores[i, j] = probs[idx]
    
    return scores

def compute_deg_e(entailment_scores):
    N = entailment_scores.shape[0]
    if N <= 1:
        return np.array([0.0]), 0.0
    deg_e_vector = np.sum(entailment_scores, axis=1) / (N - 1)
    return deg_e_vector, np.mean(deg_e_vector)

def compute_degree_of_disagreement(question, samples, model):
    """Compute degree of disagreement for a list of samples"""
    if len(samples) == 0:
        return 0.0
    entailment_scores = compute_pairwise_entailment_batch(question, samples, model)
    _, deg_e_mean = compute_deg_e(entailment_scores)
    return deg_e_mean

# %% [markdown]
# ### Semantic Density

# %%
def clean_text(text):
    for delim in ['.', '\n', 'Q:', 'A:', 'Question:', 'Answer:']:
        if delim in text:
            text = text.split(delim)[0]
    return text

def nli_semantic_distance_batch(premises, hypotheses, model, tokenizer, device):
    if not premises:
        return []
    
    inputs = tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().numpy()
    
    probs = scipy.special.softmax(logits, axis=1)
    pc, pn = probs[:, 2], probs[:, 1]
    dists = pc + 0.5 * pn
    return np.maximum(0.0, 1.0 - dists)

def compute_semantic_density(samples, probs, model):
    """Compute semantic density for a list of samples with their probabilities"""
    if len(samples) == 0:
        return 0.0
    
    ref_text = clean_text(samples[0])
    premises_d1, hypotheses_d1 = [], []
    premises_d2, hypotheses_d2 = [], []
    p_bs = []
    
    for i, sample in enumerate(samples):
        if i < len(probs) and probs[i] is not None:
            try:
                logprob = sum(probs[i]) if isinstance(probs[i], list) else probs[i]
                p_b = math.exp(logprob)
                p_bs.append(p_b)
                
                premises_d1.append(sample)
                hypotheses_d1.append(ref_text)
                premises_d2.append(ref_text)
                hypotheses_d2.append(sample)
            except (TypeError, ValueError):
                continue
    
    if not premises_d1:
        return 0.0
    
    d1_batch = nli_semantic_distance_batch(premises_d1, hypotheses_d1, model.model, model.tokenizer, model.device)
    d2_batch = nli_semantic_distance_batch(premises_d2, hypotheses_d2, model.model, model.tokenizer, model.device)
    
    sd_accum = sum(0.5 * (d1 + d2) * p_b for d1, d2, p_b in zip(d1_batch, d2_batch, p_bs))
    lik_sum = sum(p_bs)
    
    return sd_accum / (lik_sum + 1e-12)

# %% [markdown]
# ### Perplexity (Inversed)

# %%
def calc_inv_perplexity(sequence):
    try:
        if isinstance(sequence, list):
            mean_val = np.mean(sequence)
        else:
            mean_val = sequence
        if mean_val is None:
            return np.nan
        avg_nll = -mean_val
        return np.exp(-avg_nll)
    except (TypeError, ValueError):
        return np.nan

def compute_perplexity(probs):
    """Compute average inverse perplexity for a list of probability sequences"""
    if len(probs) == 0:
        return np.nan
    
    valid_probs = [p for p in probs if p is not None]
    if not valid_probs:
        return np.nan
    
    step_scores = [calc_inv_perplexity(p) for p in valid_probs]
    return np.mean(step_scores)

# %% [markdown]
# ### Normalized Entropy

# %%
def calc_normalized_entropy(sample_logprob):
    try:
        if isinstance(sample_logprob, list):
            probs = np.exp(sample_logprob)
            avg_prob = np.mean(probs)
        else:
            avg_prob = np.exp(sample_logprob)
        if avg_prob is None:
            return np.nan
        return avg_prob
    except (TypeError, ValueError):
        return np.nan

def compute_normalized_entropy(probs):
    """Compute average normalized entropy for a list of probability sequences"""
    if len(probs) == 0:
        return np.nan
    
    valid_probs = [p for p in probs if p is not None]
    if not valid_probs:
        return np.nan
    
    step_scores = [calc_normalized_entropy(p) for p in valid_probs]
    return np.mean(step_scores)

# %% [markdown]
# ### Main Processing

# %%
if __name__ == "__main__":
    # Configuration
    input_path = "path/to/input.jsonl"
    output_path = "path/to/output.jsonl"
    
    # Initialize model
    print("Loading entailment model...")
    deberta_model = EntailmentDeberta(device="cuda:0")
    
    # Load data
    print("Loading data...")
    data = load_data(input_path)
    
    # Compute UQ metrics
    print("Computing uncertainty metrics...")
    for entry in tqdm(data):
        samples = entry.get('samples', [])
        probs = entry.get('probs', [])
        question = entry.get('question', '')
        
        # Semantic Entropy
        entry['semantic_entropy'] = compute_semantic_entropy(samples, deberta_model)
        
        # Degree of Disagreement
        entry['deg_e'] = compute_degree_of_disagreement(question, samples, deberta_model)
        
        # Semantic Density (requires probs)
        if probs:
            entry['semantic_density'] = compute_semantic_density(samples, probs, deberta_model)
        
        # Perplexity (requires probs)
        if probs:
            entry['perplexity'] = compute_perplexity(probs)
        
        # Normalized Entropy (requires probs)
        if probs:
            entry['norm_entropy'] = compute_normalized_entropy(probs)
    
    # Save results
    save_results(data, output_path)
    print("Done!")