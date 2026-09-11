from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_short_prompt(secret_index: int, sentence_count: int = 8, secret_text: str = 'The exact secret is BAKED-42.') -> str:
    sentences = []
    for i in range(sentence_count):
        if i == secret_index:
            sentences.append(f'Sentence {i}: {secret_text}')
        else:
            sentences.append(f'Sentence {i}: the answer is not the secret fact.')
    prompt = (
        'You are given a short list of sentences. Return only the exact sentence that contains the secret phrase.\n'
        + '\n'.join(sentences)
        + '\nQuestion: Which sentence contains the secret phrase? Reply with only that sentence.'
    )
    return prompt


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def exact_or_fuzzy_match(prediction: str, target: str) -> tuple[float, float]:
    pred = normalize(prediction)
    target_norm = normalize(target)
    exact = 1.0 if target_norm in pred or pred in target_norm else 0.0
    fuzzy = 1.0 if normalize(target_norm.split()[-1]) in pred else 0.0
    return exact, fuzzy


def run_single_prompt(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def evaluate_base_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    secret_positions: list[int] | None = None,
    sentence_count: int = 8,
    secret_text: str = 'The exact secret is BAKED-42.',
) -> list[dict]:
    positions = secret_positions or [0, 1, 2, 4, 7]
    rows = []
    for pos in positions:
        prompt = make_short_prompt(pos, sentence_count=sentence_count, secret_text=secret_text)
        answer = run_single_prompt(model, tokenizer, prompt)
        exact, fuzzy = exact_or_fuzzy_match(answer, secret_text)
        rows.append({
            'secret_position': pos,
            'sentence_count': sentence_count,
            'prompt': prompt,
            'generated_text': answer,
            'exact_match': exact,
            'fuzzy_match': fuzzy,
            'secret_text': secret_text,
        })
    return rows


def main(output_path: str = 'qwen_1_5b_secret_benchmark.jsonl'):
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
    )
    model.eval()

    rows = evaluate_base_model(model, tokenizer)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    return rows


if __name__ == '__main__':
    rows = main()
    print(json.dumps(rows, sort_keys=True, indent=2))
