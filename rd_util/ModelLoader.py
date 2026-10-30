from unsloth import FastLanguageModel
from trl import AutoModelForCausalLMWithValueHead
import trl
from peft import LoraConfig

import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# get_peft_model builds a fresh zero-init adapter, so a resumed run has to load the
# saved weights over it; going through PeftModel.from_pretrained instead would skip
# unsloth's kernel patching
def _restore_adapter(peft_model, adapter_dir):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing, so the resumed adapter cannot be loaded")
    sd = load_file(path)
    result = set_peft_model_state_dict(peft_model, sd)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if not sd or len(unexpected) == len(sd):
        raise SystemExit(f"none of the {len(sd)} tensors in {path} matched this model")
    print(f"[Resume] adapter <- {path} ({len(sd)} tensors, {len(unexpected)} unexpected)",
          flush=True)


# trl writes the value head beside the adapter as pytorch_model.bin
def _restore_value_head(model, adapter_dir):
    path = os.path.join(adapter_dir, "pytorch_model.bin")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing, so the resumed value head cannot be loaded")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    head = {k[len("v_head."):]: v for k, v in sd.items() if k.startswith("v_head.")}
    if not head:
        raise SystemExit(f"{path} holds no v_head.* tensors")
    model.v_head.load_state_dict(head)
    print(f"[Resume] value head <- {path}", flush=True)

def _disable_shared_gpu_false_positive():
    """See csr.unsloth_compat -- the implementation is shared with the CSR path."""
    from csr.unsloth_compat import disable_shared_gpu_false_positive
    disable_shared_gpu_false_positive()


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def _fastlm_from_pretrained(model_name, max_seq_length, load_in_4bit, fix_tokenizer,
                            use_exact_model_name=True):
    """Load a base model the way RD's PPO loop expects it.

    Deliberately does NOT pass ``gpu_memory_utilization``: that knob only means
    something on unsloth's vLLM path (``fast_inference=True``), which RD does not
    use. On unsloth 2025.1.5 the argument is forwarded into
    ``LlamaForCausalLM.__init__`` and raises TypeError; retrying after dropping it
    then trips unsloth's half-applied Trainer patch, which surfaces as a
    misleading "does not support multi GPU setups" error.
    """
    _disable_shared_gpu_false_positive()
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        fix_tokenizer=fix_tokenizer,
        # never let unsloth substitute its own pre-quantized build
        use_exact_model_name=use_exact_model_name,
    )

# unsloth's own default set, used when a model yaml declares no target_modules
_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]


def _lora_params(lora):
    lora = lora or {}
    return (int(lora.get("rank", 32)),
            int(lora.get("alpha", 32)),
            float(lora.get("dropout", 0.0)),
            list(lora.get("target_modules", _DEFAULT_TARGET_MODULES)),
            int(lora.get("random_state", 3407)))


def load_lora_model_tokenizer(model_dir, is_unsloth, device, max_seq_length=1048,
                              use_exact_model_name=True, lora=None, adapter_dir=None):
    rank, alpha, dropout, target_modules, random_state = _lora_params(lora)
    if is_unsloth:
        model, tokenizer = _fastlm_from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fix_tokenizer=False,
            use_exact_model_name=use_exact_model_name,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = rank,
            target_modules = target_modules,
            lora_alpha = alpha,
            lora_dropout = dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth", 
            random_state = random_state,
            use_rslora = False,  
            loftq_config = None
        )

        if adapter_dir:
            _restore_adapter(model, adapter_dir)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        if adapter_dir:
            _restore_value_head(model, adapter_dir)
        trl.trainer.peft_module_casting_to_bf16(model)
    
    else:
        model_id = model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lora_config = LoraConfig(
                                r=rank,
                                lora_alpha=alpha,
                                lora_dropout=dropout,
                                bias="none",
                                target_modules=target_modules,
                                task_type="CAUSAL_LM"
                                )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)
        if adapter_dir:
            _restore_adapter(model.pretrained_model, adapter_dir)
            _restore_value_head(model, adapter_dir)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side='left'

    return model, tokenizer

def load_model_tokenizer(model_dir, is_unsloth, device, max_seq_length=1048,
                         use_exact_model_name=True):
    if is_unsloth:
        model, tokenizer = _fastlm_from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            fix_tokenizer=False,
            use_exact_model_name=use_exact_model_name,
        )
    else:
        model_id = model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side='left'

    return model, tokenizer
    
def load_tokenizer(dir):
    tokenizer = AutoTokenizer.from_pretrained(dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return tokenizer
