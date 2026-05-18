from app.llama_models import LlamaCompressed, LlamaGenerator

if __name__ == '__main__':
    print("=" * 60)
    print("Llama Compressed Test")
    print("=" * 60)

    print("Initializing Llama compressed model...")
    model = LlamaCompressed(1024, 1, "cpu", False)

    generator = LlamaGenerator()
    prompt = "What is the capital of France ?"
    print(f"Prompt: {prompt}")

    print("Encoding tokens...")
    prompt_tokens, prompt_tensors = model.input_encoding(prompt)

    print("Generating answer...")
    generated_tokens = generator.generate(prompt_tensors, model)
    print(generated_tokens)

