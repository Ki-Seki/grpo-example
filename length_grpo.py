import re
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

wandb.init(entity="ssc-team", project="grpo")

def get_gsm8k_questions(split = "train", keep_only=0.25) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.shuffle(seed=42)
    data = data.select(range(int(len(data) * keep_only)))
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': "Answer the user's question while making sure to use the same number of characters as the question."},
            {'role': 'user', 'content': x['question']}
        ]
    })
    return data

def same_length_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"\nQuestion:\n{q}\n", '-'*20, f"\nResponse:\n{responses[0]}")
    return [-abs(len(q)-len(r)) for r in responses]

training_args = GRPOConfig(
    output_dir="outputs/Qwen-0.5B-GRPO",
    run_name="Qwen-0.5B-GRPO-gsm8k",
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="wandb",
    push_to_hub=True,
    hub_model_id="Ki-Seki/Qwen2.5-0.5B-Parrot"
)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[same_length_reward_func],
    args=training_args,
    train_dataset=get_gsm8k_questions()
)
trainer.train()