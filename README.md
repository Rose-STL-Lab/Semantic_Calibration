# Uncertainty Quantification for LLM Trajectories

Compute uncertainty metrics for sampled LLM trajectories using semantic-based and probability-based methods.

## Installation
```bash
pip install transformers torch numpy scipy tqdm
```

## Data Format

### Input JSONL
Each line should contain:
```json
{
  "question": "What is the capital of France?",
  "samples": [
    "Paris",
    "Paris is the capital", 
    "The capital is Paris"
  ],
  "probs": [
    [-0.1, -0.2, -0.15],
    [-0.12, -0.18, -0.20],
    [-0.11, -0.19, -0.16]
  ]
}
```

**Required fields:**
- `question`: The input question/prompt
- `samples`: List of sampled answers/trajectories (minimum 1)

**Optional fields:**
- `probs`: List of token log-probabilities for each sample (required for semantic density, perplexity, and normalized entropy)

### Output JSONL
Original data + uncertainty metrics:
```json
{
  "question": "What is the capital of France?",
  "samples": [...],
  "probs": [...],
  "semantic_entropy": 0.234,
  "deg_e": 0.156,
  "semantic_density": 0.089,
  "perplexity": 1.234,
  "norm_entropy": 0.867
}
```

## Usage
```python
python uq_metrics.py
```

**Configure paths in the script:**
```python
input_path = "path/to/input.jsonl"
output_path = "path/to/output.jsonl"
```

## Metrics

| Metric | Description | Requires Probs |
|--------|-------------|----------------|
| **Semantic Entropy** | Entropy over semantic clusters of answers | No |
| **Degree of Disagreement** | Average entailment probability between answer pairs | No |
| **Semantic Density** | Probability-weighted semantic distance from reference | Yes |
| **Perplexity** | Inverse perplexity (higher = more confident) | Yes |
| **Normalized Entropy** | Average probability (exponential of log-probs) | Yes |

## Notes

- First run downloads DeBERTa-v2-xlarge-mnli (~1.5GB)
- GPU recommended (set `device="cuda:0"` or `device="cpu"`)
- Processing time: ~1-5 seconds per datapoint depending on number of samples

## Citation

Based on methods from:
- Semantic Entropy: Kuhn et al. (2023)
- Degree of Disagreement: Lin et al. (2023)
- Semantic Density: Kadavath et al. (2022)