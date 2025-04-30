import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "m-a-p/ChatMusician-Base", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Base + LoRA:
    base = AutoModelForCausalLM.from_pretrained(
        "m-a-p/ChatMusician", trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(base, "lora-final").to(device)
    model.eval()

    # 3) Build prompt
    instruction = "Harmonize the below soprano melody into SATB form. Do NOT stop until you have four voices."
    # melody = "_g _g _a _b =c' =c' z =c' _d' _a _a =b _b _b _b _a _b _d' _d' =c' _d' _d' _g _g _a _b =c' =c' z =c' _d' _a _a =b _b _b z _a _b _d' _d' =c' _d' _d' z _a _a _a _b _a =b _b _b _b _a _a z _a _a _a _b _a =b _b _b _b _a _a z _d' _b _a _g _g z _d' _b _a _g _g _g _g _g _g" # jsb
    melody = "_a _b _d' _b =f' z =f' _e' z _a _b _d' _b _e' z _e' _d'"  # rickroll
    prompt = f"{instruction}\n{melody}{tokenizer.eos_token}"
    # prompt = 'Who is Mozart?' # chat instead of harmonize

    # 4) Tokenize **without truncation** to keep the entire prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        # don’t pass truncation=True here
        max_length=None,
    ).to(device)

    # 5) Compute how many new tokens we could generate
    ctx_len    = inputs["input_ids"].shape[-1]
    max_ctx    = model.config.max_position_embeddings
    max_new    = max_ctx - ctx_len

    # 6) Generate with larger max_new_tokens
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            min_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 7) Decode & strip prompt
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    harmonization = full[len(prompt):].strip()
    print(harmonization)

if __name__ == "__main__":
    main()

