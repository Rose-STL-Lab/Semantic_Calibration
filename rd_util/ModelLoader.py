from unsloth import FastLanguageModel
from trl import AutoModelForCausalLMWithValueHead
import trl
from peft import LoraConfig

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def load_lora_model_tokenizer(model_dir, is_unsloth, device, max_seq_length=1048,
                              use_exact_model_name=True):
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
            r = 32,
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha = 32,
            lora_dropout = 0, 
            bias = "none",
            use_gradient_checkpointing = "unsloth", 
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        trl.trainer.peft_module_casting_to_bf16(model)
    
    else:
        model_id = model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lora_config = LoraConfig(
                                r=32,
                                lora_alpha=32,
                                lora_dropout=0.05,
                                bias="none",
                                target_modules=[
                                    "q_proj",
                                    "k_proj",
                                    "v_proj",
                                    "o_proj",
                                    "gate_proj",
                                    "up_proj",
                                    "down_proj",
                                ],
                                task_type="CAUSAL_LM"
                                )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)
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
