import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or path")
    parser.add_argument("--adapter", type=str, default="../../final", help="Path to the LoRA adapter")
    parser.add_argument("--output", type=str, default="../../final_merged", help="Output directory for the merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Doing merge on CPU is usually safer for memory, but can use auto
    )

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading PEFT adapter from: {args.adapter}")
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging LoRA into base model...")
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()

    print(f"Saving full merged model to: {args.output}")
    # Save full model (this will be GBs)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True  # <-- this gives you .safetensors
    )

    tokenizer.save_pretrained(args.output)
    print("Model merging complete! Merged model saved at:", args.output)

if __name__ == "__main__":
    main()