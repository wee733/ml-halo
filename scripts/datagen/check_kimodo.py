#!/usr/bin/env python3
"""Exercise Kimodo's deferred Llama/PEFT imports with small random CPU weights."""

import json
import tempfile
from types import SimpleNamespace

import torch
import transformers
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.skeleton import G1Skeleton34
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig


def main():
    torch.set_num_threads(2)
    model_class = LLM2Vec._get_model_class("LlamaConfig", enable_bidirectional=True)
    report = {"transformers": transformers.__version__, "weights": "small random CPU fixture"}
    inputs = torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    for attention in ("eager", "sdpa"):
        torch.manual_seed(17)
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=32, pad_token_id=0, use_cache=False,
        )
        config._attn_implementation = attention
        model = model_class(config).eval()
        with torch.inference_mode():
            output = model(input_ids=inputs, attention_mask=mask).last_hidden_state
            assert output.shape == (2, 4, 32) and torch.isfinite(output).all()
            changed = inputs.clone()
            changed[0, -1] = 7
            changed_output = model(input_ids=changed, attention_mask=mask).last_hidden_state
            future_delta = (output[0, 0] - changed_output[0, 0]).abs().max().item()
            encoder = LLM2Vec(model, SimpleNamespace(padding_side="left"))
            embedding = encoder.forward({
                "input_ids": inputs, "attention_mask": mask.clone(), "embed_mask": mask.clone(),
            })
            assert embedding.shape == (2, 32) and torch.isfinite(embedding).all()
        # The pinned upstream 5.1 runtime also gives zero here with a 2-D mask.
        report[attention] = {
            "forward_and_pooling": "passed", "masked_future_token_effect": future_delta,
        }

    adapter_model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION",
    )).eval()
    with torch.inference_mode():
        assert torch.isfinite(adapter_model(input_ids=inputs, attention_mask=mask).last_hidden_state).all()
    with tempfile.TemporaryDirectory(prefix="halo-kimodo-compat-") as temporary:
        adapter_model.save_pretrained(temporary, save_embedding_layers=False)
        merged = adapter_model.merge_and_unload()
        assert not any("lora_" in name for name, _ in merged.named_parameters())
        reloaded = PeftModel.from_pretrained(model_class(config), temporary).eval()
        with torch.inference_mode():
            assert torch.isfinite(reloaded(input_ids=inputs, attention_mask=mask).last_hidden_state).all()
    report["peft_forward_merge_reload"] = "passed"
    MujocoQposConverter(G1Skeleton34())
    report["g1_skeleton_converter"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
