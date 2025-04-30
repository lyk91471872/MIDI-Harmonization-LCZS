import json
import random
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
# Using LoKr instead of LoRA
from peft import LoKrConfig, get_peft_model, TaskType

# Config
CHECK = True # Set True for quick debugging on a small subset
MAX_LEN = 2048 # Max sequence length for model
MAX_SRC_LEN = 1536 # Max length for instruction + input part
LOG_DIR = 'logs_lokr'
OUTPUT_DIR = 'lokr-clean'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Reverted tokenize function to match the original LoRA script exactly
def tokenize(example, tokenizer, max_len=MAX_LEN, max_src_len=MAX_SRC_LEN):
    """
    Tokenize one example for causal LM fine-tuning.
    Masks out the prompt (instruction + input) and computes labels for the output.
    Uses global MAX_LEN and MAX_SRC_LEN unless overridden.
    """
    src_text = example['instruction'] + example['input']
    # Tokenize source and truncate
    src_tokens = tokenizer.tokenize(src_text)[:max_src_len]

    # Calculate max target length based on truncated source
    max_tgt_len = max_len - len(src_tokens) - 1 # Reserve space for potential EOS
    max_tgt_len = max(0, max_tgt_len) # Ensure non-negative

    # Tokenize target and truncate
    tgt_tokens = tokenizer.tokenize(example['output'])[:max_tgt_len]

    tokens = src_tokens + tgt_tokens
    # Append EOS if output doesn't end with it and space allows
    if not example['output'].strip().endswith(tokenizer.eos_token) and len(tokens) < max_len:
        tokens.append(tokenizer.eos_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Create labels: mask prompt tokens (-100), keep output tokens
    labels = [-100] * len(src_tokens) + input_ids[len(src_tokens):]

    # Final truncation if somehow still too long (should be rare)
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels    = labels[-max_len:]

    return {'input_ids': input_ids, 'labels': labels}


def make_prefix_continuations(example, num_augments=10):
    """Generate prefix-continuation variants by splitting output at spaces/newlines."""
    augments = []
    try:
        text = example['output']
        example_id = example['id'] # Assume 'id' exists and keep original
        instruction = example['instruction']
        input_text = example['input']
    except KeyError as e:
        # Minimal error handling as requested
        print(f"Warning: Missing key {e} in example ID {example.get('id', 'N/A')} for augmentation. Skipping.")
        return []

    split_positions = [m.start() for m in re.finditer(r"[ \n]", text)]
    if text and not text.isspace():
         split_positions.append(len(text))
    split_positions = sorted(list(set(split_positions)))

    count = 0
    attempts = 0
    max_attempts = num_augments * 3

    while count < num_augments and attempts < max_attempts:
        attempts += 1
        if not split_positions: break

        idx = random.choice(split_positions)
        prefix = text[:idx].rstrip()
        suffix = text[idx:].lstrip()

        # Add augmentation if prefix/suffix are non-empty (or edge case)
        if prefix and suffix:
            augments.append({
                'id': example_id,
                'instruction': instruction,
                'input': input_text + prefix,
                'output': suffix
            })
            count += 1
        elif idx == len(text) and prefix and not augments:
             augments.append({
                'id': example_id,
                'instruction': instruction,
                'input': input_text + prefix,
                'output': ""
            })
             count += 1

    return augments[:num_augments]


def compute_metrics(eval_pred):
    """Compute token-level accuracy, ignoring labels set to -100."""
    logits, labels = eval_pred
    preds = np.argmax(logits.astype(np.float32), axis=-1)
    mask = labels != -100
    labels_flat = labels[mask].flatten()
    preds_flat = preds[mask].flatten()
    if labels_flat.size == 0: return {'accuracy': 0.0}
    acc = accuracy_score(labels_flat, preds_flat)
    return {'accuracy': acc}


def plot_metrics(log_history, path='loss_curve_lokr.png'):
    """Plot training/validation loss from log history."""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    for rec in log_history: # Extract steps and metrics
        step = rec.get('step')
        if step is None: continue
        if 'loss' in rec and 'eval_loss' not in rec:
            train_steps.append(step)
            train_losses.append(rec['loss'])
        if 'eval_loss' in rec:
            val_steps.append(step)
            val_losses.append(rec['eval_loss'])

    plt.figure(figsize=(8, 5)) # Smaller figure
    if train_steps: plt.plot(train_steps, train_losses, label='train_loss', alpha=0.7)
    if val_steps: plt.plot(val_steps, val_losses, label='eval_loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('LoKr Training & Validation Loss')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved metrics plot to {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset splits
    dataset_path = 'data/dataset_jsb' # Assumes this has train/valid/test splits
    ds_splits = load_from_disk(dataset_path)
    print(f"Loaded dataset splits: {list(ds_splits.keys())}")
    train_ds, val_ds = ds_splits['train'], ds_splits['valid']

    if CHECK: # Select small subset for quick check
        train_ds = train_ds.select(range(min(20, len(train_ds))))
        val_ds   = val_ds.select(range(min(20, len(val_ds))))
        print(f"[CHECK MODE] Samples - Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Augment training data only
    print("Augmenting training data...")
    aug_records = []
    for ex in train_ds:
        aug_records.append(ex)
        aug_records.extend(make_prefix_continuations(ex, num_augments=10))
    # Create dataset without explicit features (relies on inference)
    train_aug = Dataset.from_list(aug_records)
    print(f"Augmented train size: {len(train_aug)}")

    # Load tokenizer & model
    model_name = 'm-a-p/ChatMusician'
    tokenizer_name = 'm-a-p/ChatMusician-Base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    base_model.gradient_checkpointing_enable() # Keep gradient checkpointing

    # LoKr Config (instead of LoRA)
    lokr_config = LoKrConfig(
        task_type=TaskType.CAUSAL_LM, r=16, alpha=32,
        target_modules=['q_proj', 'v_proj'], # Verify these modules
        rank_dropout=0.05, module_dropout=0.0, init_weights=True,
    )
    model = get_peft_model(base_model, lokr_config)
    model.config.use_cache = False # Required for gradient checkpointing
    model.print_trainable_parameters()

    # Tokenize datasets using the original tokenize function
    print("Tokenizing datasets...")
    train_tok = train_aug.map(lambda ex: tokenize(ex, tokenizer), batched=False, remove_columns=train_aug.column_names)
    val_tok   = val_ds.map(lambda ex: tokenize(ex, tokenizer), batched=False, remove_columns=val_ds.column_names)

    # Collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest', label_pad_token_id=-100)

    # Training Params
    if CHECK: per_device_bs, grad_accum, epochs = 1, 1, 1
    else: per_device_bs, grad_accum, epochs = 4, 8, 5 # Effective batch size 32

    # Training Arguments - Simplified logging, epoch-based saving/eval
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs * 2,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=2e-5, warmup_ratio=0.03, lr_scheduler_type='cosine',
        weight_decay=0.01, gradient_checkpointing=True,
        logging_steps=50, # Log less frequently
        save_strategy='epoch', evaluation_strategy='epoch',
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to='none', overwrite_output_dir=True, logging_dir=LOG_DIR,
        max_grad_norm=1.0,
    )

    # Trainer
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=collator, tokenizer=tokenizer, compute_metrics=compute_metrics
    )

    # Training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Evaluation
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_tok)
    print(f"Validation metrics: {val_metrics}")
    if 'eval_loss' in val_metrics: print(f"Validation perplexity: {np.exp(val_metrics['eval_loss']):.2f}")

    # Test Evaluation (if test split exists)
    if 'test' in ds_splits:
        print("Evaluating on test set...")
        test_ds = ds_splits['test']
        if CHECK: test_ds = test_ds.select(range(min(20, len(test_ds))))
        test_tok = test_ds.map(lambda ex: tokenize(ex, tokenizer), batched=False, remove_columns=test_ds.column_names)
        if len(test_tok) > 0:
             test_metrics = trainer.evaluate(eval_dataset=test_tok)
             print(f"Test metrics: {test_metrics}")
             if 'eval_loss' in test_metrics: print(f"Test perplexity: {np.exp(test_metrics['eval_loss']):.2f}")
        else: print("Test set empty after tokenization/filtering.")

    # Save final adapter, logs, plot
    final_adapter_path = os.path.join(OUTPUT_DIR, 'lokr-final-best')
    model.save_pretrained(final_adapter_path)
    print(f"Best adapter saved to {final_adapter_path}")
    log_history_path = os.path.join(OUTPUT_DIR, 'train_history_lokr.json')
    with open(log_history_path, 'w') as f: json.dump(trainer.state.log_history, f, indent=2)
    plot_metrics(trainer.state.log_history, path=os.path.join(OUTPUT_DIR, 'loss_curve_lokr.png'))

    print('Script finished successfully.')

if __name__ == '__main__':
    main()
