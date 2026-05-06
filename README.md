# IN5550-Track: LLM Reasoning

This track comes from **SemEval-2026 Task 11, Subtask 1: English syllogistic validity prediction**.

The task asks whether a natural-language syllogism is **formally valid**, regardless of whether the content sounds plausible in the real world. In other words, the model should judge logical form rather than factual or common-sense believability.

## 1. Task Description

Given an English syllogism, the system predicts whether the conclusion follows logically from the premises. 

The important point is that **validity** and **plausibility** are different:

- **validity**: whether the conclusion is logically entailed by the premises.
- **plausibility**: whether the argument sounds believable according to common sense.

For example:

```text
All dogs are fish.
All fish are mammals.
Therefore, all dogs are mammals.
```

This syllogism is **implausible** in the real world, but it is **formally valid** because the conclusion follows from the two premises.

By contrast:

```text
All dogs are mammals.
Therefore, all mammals are dogs.
```

This may sound related to real-world knowledge, but it is **formally invalid** because the conclusion does not follow from the premise.

The goal of this task is therefore to test whether models can perform **content-independent formal reasoning**.

Each example contains:

```json
{
  "id": "bff2af61-d4b0-4147-8a5b-ff4fe1892559",
  "syllogism": "There are no bikes that can be called cars. It is also true that every bike is a type of vehicle. This has led to the conclusion that a portion of vehicles are bikes.",
  "validity": false,
  "plausibility": true
}
```

## 2. Evaluation Metrics

For Subtask 1, the official evaluation reports three metrics:

- `accuracy`
- `content_effect`
- `combined_score`

The official ranking metric rewards high accuracy while penalizing large content effects:

```text
combined_score = accuracy / (1 + ln(1 + content_effect))
```

A system should therefore not only be accurate overall, but also robust to whether the argument content sounds plausible or implausible.



### 2.1 Accuracy

`accuracy` is the standard binary classification accuracy:

```text
accuracy = number of correct validity predictions/number of examples
```

A prediction is correct if the predicted `validity` matches the gold `validity`.



### 2.2 Content Effect

`content_effect` measures whether the model's validity prediction is biased by real-world plausibility.

The dataset separates examples along two dimensions:

| Logical validity | Content plausibility | Meaning |
|---|---|---|
| valid | plausible | Logic says yes, content sounds believable |
| valid | implausible | Logic says yes, content sounds unbelievable |
| invalid | plausible | Logic says no, content sounds believable |
| invalid | implausible | Logic says no, content sounds unbelievable |

A content-independent reasoner should perform similarly across plausible and implausible examples.

For the intra-content effect, we compare:

```text
CE_intra =|acc_valid_plausible - acc_valid_implausible| + |acc_invalid_plausible - acc_invalid_implausible|
```

For the inter-content effect, we compare:

```text
CE_inter = |acc_valid_plausible - acc_invalid_plausible| + |acc_valid_implausible - acc_invalid_implausible|
```

```text
content_effect = CE_intra + CE_inter
```

If the model is much better on plausible-valid examples than implausible-valid examples, it is likely relying on real-world plausibility rather than formal logic.

If the model is much worse on plausible-invalid examples than on implausible-invalid ones, it is likely being fooled by arguments whose conclusions sound plausible.

A lower `content_effect` is better.

Intuitively:

```text
content_effect = 0
```

means the model is not affected by plausibility.

A large `content_effect` means the model's errors are systematically correlated with whether the content sounds believable.



### 2.3 Combined Score

The official combined score is:

```text
combined_score = accuracy / (1 + ln(1 + content_effect))
```

This means:

- higher `accuracy` improves the score;
- higher `content_effect` lowers the score;
- two models with the same accuracy can have different combined scores if one is more content-biased.

Example:

| Model | Accuracy | Content Effect | Combined Score |
|---|---|---|---:|
| Model A | 0.85 | 0.05 | higher |
| Model B | 0.85 | 0.50 | lower |

Although both models have the same accuracy, Model B is penalized more because it is more affected by content plausibility.



## 3. Implemented Baselines

This project implements four baselines using Qwen-8B as the base model. After prediction, each baseline is evaluated with the official evaluation script. The final summary only keeps:

- `accuracy`
- `content_effect`
- `combined_score`

#### 3.1 Prompt Engineering Baseline

This is a zero-shot prompting baseline.

The model is directly prompted to judge whether the syllogism is formally valid. The prompt explicitly tells the model to ignore real-world plausibility and treat nouns as abstract categories.

This baseline is simple and fast, but it may still be affected by content plausibility.



#### 3.2 In-Context Learning Baseline

This baseline retrieves the 4 most similar examples from the training set and uses them as demonstrations.

Similarity is computed using TF-IDF over the syllogism text.

This baseline tests whether local examples help the model infer the correct formal reasoning pattern.


#### 3.3 PEFT / LoRA Fine-Tuning Baseline

This baseline uses parameter-efficient fine-tuning with LoRA.

Instead of updating all model parameters, LoRA trains a small number of adapter parameters.

This is usually much cheaper than full fine-tuning and is the recommended training-based baseline to run first.

We use PEFT/LoRA instead of Full fine-tuning. Full-parameter AdamW training of Qwen-8B exceeded the memory of a single 40GB GPU. We therefore report PEFT/LoRA as the training-based baseline.




## 4. How to Run

The Slurm script supports four switches:

- `RUN_PROMPT`
- `RUN_ICL`
- `RUN_PEFT`

Each switch can be set to `1` or `0`.

Run all baselines:

```bash
RUN_PROMPT=1 RUN_ICL=1 RUN_PEFT=1 \
sbatch run_subtask1_baselines_fox.slurm
```



## 5. Output Files

The script writes one prediction file per baseline:

```text
outputs/subtask1/prompt_predictions.json
outputs/subtask1/icl_predictions.json
outputs/subtask1/full_predictions.json
outputs/subtask1/peft_predictions.json
```

The final metric summary is saved as:

```text
outputs/subtask1/summary_metrics.json
```



## 6. Baseline Results

Fill this table after running the baselines.

| Baseline | Base Model | Training | Accuracy | Content Effect | Combined Score |
|---|---|---|---:|---:|---:|
| Prompt Engineering | Qwen-8B | No | 47.1204 | 34.375 | 10.3198 |
| In-Context Learning, top-4 similar examples | Qwen-8B | No | 49.2147 | 48.9583 | 10.0209 |
| PEFT / LoRA Fine-Tuning | Qwen-8B | Yes, LoRA adapters | 97.3822 | 3.1915 | 40.0246 |


## 7. Existing Public Baselines

This section lists public implementations or papers that are useful for comparison.



### 7.1 Official SemEval-2026 Task 11 Repository

Link:  
https://github.com/neuro-symbolic-ai/semeval_2026_task_11

The official repository provides the task description, data, mock prediction files, and evaluation scripts.



### 7.2 ITLC: Normalization and Deterministic Parsing

Code:  
https://github.com/SEACrowd/ITLC_semeval2026_shared_task_11

Paper:  
https://arxiv.org/abs/2603.02676

ITLC uses structural abstraction. It converts natural-language syllogisms into canonical logical representations and then applies deterministic parsing to determine validity.

This is a strong neuro-symbolic direction because the system removes surface content and focuses on logical form.



### 7.3 FregeLogic: LLM Ensemble + Z3 Solver

Paper:  
https://arxiv.org/abs/2604.18328

FregeLogic is a hybrid neuro-symbolic system for Subtask 1. It combines five LLM classifiers based on Llama 4 Maverick, Llama 4 Scout, and Qwen3-32B with different prompting strategies. When the ensemble has a narrow disagreement, it calls a Z3 SMT solver as a formal-logic tiebreaker.

I did not find a public GitHub repository for FregeLogic at the time of writing.



### 7.4 Dual Parser Neurosymbolic System

Code:  
https://github.com/Thiyaga1586/Syllogistic_Reasoning

This implementation uses a dual-parser neuro-symbolic pipeline for Subtask 1. According to its README, it uses a fine-tuned T5-small parser, self-consistency decoding, an 8-region Venn SAT solver for symbolic entailment checking, and a counterfactually trained DistilRoBERTa fallback.

This direction is useful because the symbolic solver can be robust to content effect when the logical form is parsed correctly.



## 8. Further Reading

Official task repository:  
https://github.com/neuro-symbolic-ai/semeval_2026_task_11

Official SemEval-2026 website:  
https://semeval.github.io/SemEval2026/tasks.html

Official Codabench Subtask 1 page:  
https://www.codabench.org/competitions/11928/

Task paper/reference from the official repository:  
Valentino et al. SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models.

Related papers:

- ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs  
  https://arxiv.org/abs/2603.02676

- FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction  
  https://arxiv.org/abs/2604.18328
