# This code is from the oficial vLLM documentation: https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

from vllm import LLM, SamplingParams

prompts = [
    "What is the capital of France?",
    "What is the capital of the US?",
    "What is the difference between a cow and a horse?",
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=None)

llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Save output to a txt file
with open("output_batched.txt", "w") as f:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        f.write(f"Prompt: {prompt!r}\nGenerated text: {generated_text}\n\n")