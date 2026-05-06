#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four baselines for SemEval-2026 Task 11 Subtask 1:
1) prompt engineering zero-shot
2) in-context learning with 4 most similar training examples
3) full fine-tuning of Qwen-8B
4) PEFT/LoRA fine-tuning of Qwen-8B

Each baseline writes:
  outputs/subtask1_<baseline>_predictions.json
  outputs/subtask1_<baseline>_metrics.json

The final summary only contains:
  accuracy, content_effect, combined_score

Example:
python run_subtask1_baselines.py \
  --repo_dir ./semeval_2026_task_11 \
  --train_file ./semeval_2026_task_11/data/subtask1/train.json \
  --reference_file ./semeval_2026_task_11/data/subtask1/dev.json \
  --eval_file ./semeval_2026_task_11/data/subtask1/dev.json \
  --model_name_or_path Qwen/Qwen3-8B \
  --output_dir outputs \
  --run_prompt 1 --run_icl 1 --run_full 0 --run_peft 1
"""

import argparse
import gc
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None

SYSTEM_PROMPT = """You are a formal logic solver for syllogistic reasoning.
Judge only whether the conclusion follows logically from the premises.
Ignore real-world plausibility, common sense, and factual knowledge.
Treat all content words as abstract categories.
Return only valid JSON with one key: validity.
""".strip()

ZERO_SHOT_USER_PROMPT = """Determine whether the following syllogism is formally valid.

Important rules:
- Ignore whether the content sounds true or false in the real world.
- Treat nouns as abstract categories.
- Only formal validity matters.
- Output exactly one JSON object: {{"validity": true}} or {{"validity": false}}.

Syllogism:
{syllogism}
""".strip()

ICL_USER_PROMPT = """Determine whether the target syllogism is formally valid.

Important rules:
- Ignore whether the content sounds true or false in the real world.
- Treat nouns as abstract categories.
- Only formal validity matters.
- The examples below are retrieved because they are textually similar, but you must still reason by formal structure.
- Output exactly one JSON object: {{"validity": true}} or {{"validity": false}}.

Examples:
{examples}

Target syllogism:
{syllogism}
""".strip()

ASSISTANT_TRUE = '{"validity": true}'
ASSISTANT_FALSE = '{"validity": false}'


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_syllogism(item: Dict[str, Any]) -> str:
    if "syllogism" in item and item["syllogism"]:
        return str(item["syllogism"])
    # Fallback for possible alternative schemas.
    parts = []
    if "premises" in item:
        parts.extend(item["premises"] if isinstance(item["premises"], list) else [str(item["premises"])])
    if "conclusion" in item:
        parts.append(str(item["conclusion"]))
    return "\n".join(parts)


def parse_validity(text: str) -> bool:
    t = text.strip().lower()
    # Prefer explicit JSON-like booleans.
    m = re.search(r'"?validity"?\s*:\s*(true|false)', t)
    if m:
        return m.group(1) == "true"
    # Then robust fallbacks.
    if re.search(r"\binvalid\b", t):
        return False
    if re.search(r"\bvalid\b", t):
        return True
    if re.search(r"\bfalse\b", t):
        return False
    if re.search(r"\btrue\b", t):
        return True
    # Conservative fallback.
    return False


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"


def build_train_text_and_prompt(tokenizer, syllogism: str, label: bool) -> Tuple[str, str]:
    user_prompt = ZERO_SHOT_USER_PROMPT.format(syllogism=syllogism)
    answer = ASSISTANT_TRUE if label else ASSISTANT_FALSE
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": answer}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
        full_text = prompt_text + " " + answer
    return full_text, prompt_text


@dataclass
class CausalClsDataset(torch.utils.data.Dataset):
    items: List[Dict[str, Any]]
    tokenizer: Any
    max_length: int

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        full_text, prompt_text = build_train_text_and_prompt(
            self.tokenizer,
            get_syllogism(item),
            bool(item["validity"]),
        )
        full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        prompt = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(features: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in features:
        n = max_len - len(x["input_ids"])
        batch["input_ids"].append(torch.cat([x["input_ids"], torch.full((n,), pad_token_id, dtype=torch.long)]))
        batch["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        batch["labels"].append(torch.cat([x["labels"], torch.full((n,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in batch.items()}


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_inference(model_name_or_path: str, dtype: str = "bf16"):
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model

def load_peft_model_for_inference(
    base_model_name_or_path: str,
    adapter_dir: str,
    dtype: str = "bf16",
):
    if PeftModel is None:
        raise RuntimeError("peft is not installed. Please run: pip install peft")

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )

    model.eval()
    return tokenizer, model

def generate_predictions(
    model,
    tokenizer,
    eval_items: List[Dict[str, Any]],
    output_path: str,
    prompt_builder,
    batch_size: int = 4,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    preds = []
    for start in tqdm(range(0, len(eval_items), batch_size), desc=f"generating {Path(output_path).name}"):
        batch_items = eval_items[start : start + batch_size]
        prompts = [build_chat_prompt(tokenizer, prompt_builder(x)) for x in batch_items]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        gen_texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for item, text in zip(batch_items, gen_texts):
            preds.append({"id": item["id"], "validity": parse_validity(text)})
    save_json(preds, output_path)
    return preds


def retrieve_topk_examples(train_items: List[Dict[str, Any]], eval_items: List[Dict[str, Any]], k: int = 4):
    train_texts = [get_syllogism(x) for x in train_items]
    eval_texts = [get_syllogism(x) for x in eval_items]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", lowercase=True, min_df=1)
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    sims = cosine_similarity(x_eval, x_train)
    topk = np.argsort(-sims, axis=1)[:, :k]
    return topk


def make_icl_prompt_builder(train_items: List[Dict[str, Any]], eval_items: List[Dict[str, Any]], k: int = 4):
    topk = retrieve_topk_examples(train_items, eval_items, k=k)
    id_to_pos = {item["id"]: i for i, item in enumerate(eval_items)}

    def _builder(item: Dict[str, Any]) -> str:
        pos = id_to_pos[item["id"]]
        example_blocks = []
        for j, train_idx in enumerate(topk[pos], 1):
            ex = train_items[int(train_idx)]
            label = "true" if bool(ex["validity"]) else "false"
            example_blocks.append(
                f"Example {j}:\nSyllogism: {get_syllogism(ex)}\nAnswer: {{\"validity\": {label}}}"
            )
        return ICL_USER_PROMPT.format(examples="\n\n".join(example_blocks), syllogism=get_syllogism(item))

    return _builder


def run_official_eval(repo_dir: str, reference_file: str, prediction_file: str, output_file: str) -> Dict[str, float]:
    eval_script = Path(repo_dir) / "evaluation_kit" / "task 1 & 3" / "evaluation_script.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Cannot find official evaluation script: {eval_script}")
    spec = importlib.util.spec_from_file_location("official_eval_task13", str(eval_script))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.run_full_scoring(reference_file, prediction_file, output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "content_effect": metrics.get("content_effect", 0.0),
        "combined_score": metrics.get("combined_score", 0.0),
    }


def free_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    baseline_name: str,
    model_name_or_path: str,
    train_items: List[Dict[str, Any]],
    output_model_dir: str,
    max_length: int,
    epochs: float,
    lr: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    dtype: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    deepspeed_config: Optional[str],
):
    tokenizer = load_tokenizer(model_name_or_path)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if use_lora:
        if get_peft_model is None:
            raise RuntimeError("peft is not installed. Please run: pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = CausalClsDataset(train_items, tokenizer, max_length=max_length)
    args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=(dtype == "bf16"),
        fp16=(dtype == "fp16"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        deepspeed=deepspeed_config if deepspeed_config else None,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda xs: collate_batch(xs, tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    return output_model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", required=True, help="Path to cloned official repository")
    parser.add_argument("--train_file", required=True, help="Subtask 1 training JSON with validity labels")
    parser.add_argument("--eval_file", required=True, help="Subtask 1 examples to predict")
    parser.add_argument("--reference_file", required=True, help="Gold reference JSON for official evaluation")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", default="outputs_subtask1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)

    parser.add_argument("--run_prompt", type=int, default=1)
    parser.add_argument("--run_icl", type=int, default=1)
    parser.add_argument("--run_full", type=int, default=0)
    parser.add_argument("--run_peft", type=int, default=1)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--full_lr", type=float, default=1e-5)
    parser.add_argument("--peft_lr", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--deepspeed_config", default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_json(args.train_file)
    eval_items = load_json(args.eval_file)
    if args.limit_train:
        train_items = train_items[: args.limit_train]
    if args.limit_eval:
        eval_items = eval_items[: args.limit_eval]

    summary: Dict[str, Dict[str, float]] = {}

    if args.run_prompt:
        tokenizer = load_tokenizer(args.model_name_or_path)
        model = load_model_for_inference(args.model_name_or_path, dtype=args.dtype)
        pred_path = str(out_dir / "subtask1_prompt_predictions.json")
        metric_path = str(out_dir / "subtask1_prompt_metrics.json")
        generate_predictions(
            model,
            tokenizer,
            eval_items,
            pred_path,
            prompt_builder=lambda x: ZERO_SHOT_USER_PROMPT.format(syllogism=get_syllogism(x)),
            batch_size=args.gen_batch_size,
        )
        summary["prompt_engineering"] = run_official_eval(args.repo_dir, args.reference_file, pred_path, metric_path)
        free_memory(model, tokenizer)

    if args.run_icl:
        tokenizer = load_tokenizer(args.model_name_or_path)
        model = load_model_for_inference(args.model_name_or_path, dtype=args.dtype)
        pred_path = str(out_dir / "subtask1_icl4_predictions.json")
        metric_path = str(out_dir / "subtask1_icl4_metrics.json")
        icl_builder = make_icl_prompt_builder(train_items, eval_items, k=4)
        generate_predictions(
            model,
            tokenizer,
            eval_items,
            pred_path,
            prompt_builder=icl_builder,
            batch_size=max(1, args.gen_batch_size // 2),
        )
        summary["icl_top4"] = run_official_eval(args.repo_dir, args.reference_file, pred_path, metric_path)
        free_memory(model, tokenizer)

    if args.run_full:
        full_model_dir = str(out_dir / "qwen8b_full_sft")
        train_model(
            baseline_name="full",
            model_name_or_path=args.model_name_or_path,
            train_items=train_items,
            output_model_dir=full_model_dir,
            max_length=args.max_length,
            epochs=args.epochs,
            lr=args.full_lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dtype=args.dtype,
            use_lora=False,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            deepspeed_config=args.deepspeed_config,
        )
        tokenizer = load_tokenizer(full_model_dir)
        model = load_model_for_inference(full_model_dir, dtype=args.dtype)
        pred_path = str(out_dir / "subtask1_full_training_predictions.json")
        metric_path = str(out_dir / "subtask1_full_training_metrics.json")
        generate_predictions(
            model,
            tokenizer,
            eval_items,
            pred_path,
            prompt_builder=lambda x: ZERO_SHOT_USER_PROMPT.format(syllogism=get_syllogism(x)),
            batch_size=args.gen_batch_size,
        )
        summary["full_training"] = run_official_eval(args.repo_dir, args.reference_file, pred_path, metric_path)
        free_memory(model, tokenizer)

    if args.run_peft:
        peft_model_dir = str(out_dir / "qwen8b_lora_sft")
        train_model(
            baseline_name="peft",
            model_name_or_path=args.model_name_or_path,
            train_items=train_items,
            output_model_dir=peft_model_dir,
            max_length=args.max_length,
            epochs=args.epochs,
            lr=args.peft_lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dtype=args.dtype,
            use_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            deepspeed_config=None,
        )
        tokenizer, model = load_peft_model_for_inference(
            base_model_name_or_path=args.model_name_or_path,
            adapter_dir=peft_model_dir,
            dtype=args.dtype,
        )
        pred_path = str(out_dir / "subtask1_peft_lora_predictions.json")
        metric_path = str(out_dir / "subtask1_peft_lora_metrics.json")
        generate_predictions(
            model,
            tokenizer,
            eval_items,
            pred_path,
            prompt_builder=lambda x: ZERO_SHOT_USER_PROMPT.format(syllogism=get_syllogism(x)),
            batch_size=args.gen_batch_size,
        )
        summary["peft_lora"] = run_official_eval(args.repo_dir, args.reference_file, pred_path, metric_path)
        free_memory(model, tokenizer)

    summary_path = out_dir / "summary_metrics.json"
    save_json(summary, str(summary_path))

    # Final console output: only the three official metrics for each selected baseline.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
