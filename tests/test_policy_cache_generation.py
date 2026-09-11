import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.niah_recall import build_policy_cache


def test_policy_cache_changes_next_token_logits_for_same_prefix():
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cpu')
    model.eval()

    prompt = (
        'You are given a short list of facts.\n'
        'Fact 0: the answer is not the secret fact.\n'
        'Fact 1: the answer is not the secret fact.\n'
        'Fact 2: the secret answer is 42.\n'
        'Question: What is the secret fact? Reply with the exact fact only.'
    )
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

    stream_cache = build_policy_cache(model, input_ids, 'streamingllm', window_size=16, survivor_every=8)
    rsqr_cache = build_policy_cache(model, input_ids, 'rsqr', window_size=16, survivor_every=8)

    stream_logits = model(
        input_ids=input_ids[:, -1:],
        past_key_values=stream_cache,
        use_cache=True,
    ).logits
    rsqr_logits = model(
        input_ids=input_ids[:, -1:],
        past_key_values=rsqr_cache,
        use_cache=True,
    ).logits

    assert not torch.allclose(stream_logits, rsqr_logits, atol=1e-5, rtol=1e-3)
