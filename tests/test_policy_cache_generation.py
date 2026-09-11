import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.niah_recall import build_policy_cache, make_strong_needle_prompt


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


def test_streamingllm_cache_keeps_sink_tokens_plus_recent_window():
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cpu')
    model.eval()

    prompt = ' '.join(f'fact {i}: the answer is not the secret fact.' for i in range(20))
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    cache = build_policy_cache(model, input_ids, 'streamingllm', window_size=12, sink_size=4, survivor_every=8)

    kept_len = cache.layers[0].keys.shape[-2]
    assert kept_len == 16, f'expected 16 kept positions (4 sink + 12 recent), got {kept_len}'

    # The first four positions are kept as the sink set and the last twelve are the recent window.
    kept_positions = list(range(0, 4)) + list(range(len(input_ids[0]) - 12, len(input_ids[0])))
    assert cache.layers[0].keys.shape[-2] == len(kept_positions)
    assert torch.equal(
        cache.layers[0].keys[:, :, :4, :].sum(dim=(0, 1, 3)),
        model(input_ids=input_ids, use_cache=True).past_key_values.layers[0].keys[:, :, :4, :].sum(dim=(0, 1, 3)),
    )


def test_make_strong_needle_prompt_uses_unique_secret_phrase_and_exact_instruction():
    needle_fact = 'The exact secret is BAKED-42.'
    prompt = make_strong_needle_prompt(context_length=32, needle_depth=18, needle_fact=needle_fact)

    assert 'BAKED-42' in prompt
    assert 'Return only the exact sentence' in prompt
    assert 'Sentence 18' in prompt or 'Sentence 18:' in prompt
    assert 'sentence contains the exact keyphrase' in prompt.lower()
