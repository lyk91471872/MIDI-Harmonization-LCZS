import torch
import random
import re
import json
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType


def tokenized(example, tokenizer, max_len=2048, max_src_len=1536):
    """
    Tokenize one example for causal LM fine-tuning.
    Masks out the prompt (instruction + input) and computes labels for the output.
    """
    src_text = example['instruction'] + example['input']
    src_tokens = tokenizer.tokenize(src_text)[:max_src_len]

    max_tgt_len = max_len - len(src_tokens) - 3
    tgt_tokens = tokenizer.tokenize(example['output'])[:max_tgt_len]

    tokens = src_tokens + tgt_tokens
    if not example['output'].strip().endswith(tokenizer.eos_token):
        tokens.append(tokenizer.eos_token)
    assert len(tokens) <= max_len, f'Sequence too long: {len(tokens)} > {max_len}'

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels = [-100] * len(src_tokens) + input_ids[len(src_tokens):]

    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels   = labels[-max_len:]

    return {'input_ids': input_ids, 'labels': labels}


def make_prefix_continuations(example, num_augments=10):
    """
    Generate prefix-continuation variants for one example.
    Splits the example['output'] at random space or newline positions.
    """
    augmented = []
    text = example['output']
    positions = [m.start() for m in re.finditer(r"[ \n]", text)]

    for _ in range(num_augments):
        if not positions:
            break
        split_idx = random.choice(positions)
        prefix = text[:split_idx].rstrip()
        suffix = text[split_idx:].lstrip()

        augmented.append({
            'id': example.get('id', '') + f'_pref{split_idx}',
            'instruction': example['instruction'] + ' Do NOT stop until you have four voices.',
            'input': example['input'] + prefix,
            'output': suffix
        })
    return augmented


def plot_loss_curve(log_history, output_path='loss_curve.png'):
    """
    Extracts training loss by step from Trainer log_history and plots loss curve.
    """
    steps = []
    losses = []
    for record in log_history:
        if 'loss' in record:
            steps.append(record.get('step', len(steps)))
            losses.append(record['loss'])
    if steps and losses:
        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel('Step')
        plt.ylabel('Training Loss')
        plt.title('Loss Curve')
        plt.savefig(output_path)
        plt.close()


def main():
    # 1) Set device
    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    # 2) Load original dataset
    ds = load_from_disk('data/dataset_jsb_abc')

    # 3) Augment examples manually
    augmented_list = []
    for ex in ds:
        augmented_list.append(ex)
        augmented_list.extend(make_prefix_continuations(ex, num_augments=10))
    ds_aug = Dataset.from_list(augmented_list)
    print(f"Dataset size before: {len(ds)}, after augment: {len(ds_aug)}")

    # 4) Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(
        'm-a-p/ChatMusician-Base', trust_remote_code=True, timeout=60
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        'm-a-p/ChatMusician', trust_remote_code=True
    ).to(device)

    # 5) Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.0,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 6) Tokenize augmented dataset
    tokenized_dataset = ds_aug.map(
        lambda ex: tokenized(ex, tokenizer), batched=False
    )

    # 7) Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest',
        label_pad_token_id=-100,
        return_tensors='pt',
    )

    # 8) Training arguments
    training_args = TrainingArguments(
        output_dir='./lora-clean',
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to='none',
        overwrite_output_dir=True,
        # Save the logs for plotting
        logging_dir='./logs',
    )

    # 9) Trainer and training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 10) Save final adapter
    model.save_pretrained('lora-final')

    # 11) Save log history
    log_history = trainer.state.log_history
    with open('train_history.json', 'w') as f:
        json.dump(log_history, f, indent=2)
    print('Saved training history to train_history.json')

    # 12) Plot and save loss curve
    plot_loss_curve(log_history, output_path='loss_curve.png')
    print('Saved loss curve to loss_curve.png')

if __name__ == '__main__':
    main()
