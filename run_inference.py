import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Run inference with base model + optional LoRA adapter.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B", 
                        help="Base model path or huggingface ID")
    parser.add_argument("--adapter", type=str, default="outputs/sft/final", 
                        help="Path to the trained LoRA adapter (e.g., outputs/grpo/final)")
    parser.add_argument("--prompt", type=str, default=None, 
                        help="Single prompt to run. If not provided, starts interactive chat.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Inference Pipeline")
    print(f"  Base Model : {args.base_model}")
    print(f"  Adapter    : {args.adapter}")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if args.adapter and os.path.exists(args.adapter):
        print(f"Applying PEFT adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    else:
        print(f"WARNING: Adapter path '{args.adapter}' not found. Running base model only.")
        model.eval()

    print("\nModel loaded and ready!\n")

    def generate_response(user_text: str):
        # Format using the tokenizer's chat template
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Always format your code inside <python> tags and wrap explanations in <info> tags."},
            {"role": "user", "content": user_text}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated response (excluding the prompt sequence)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print("-" * 40)
        response = generate_response(args.prompt)
        print(response)
        print("-" * 40)
    else:
        print("Entering interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                user_input = input("\nUser> ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input.strip():
                    continue
                
                print("\nAssistant> ", end="", flush=True)
                response = generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
