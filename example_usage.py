"""
example_usage.py -- Demonstrates FastInferenceEngineClient.
"""
from client import FastInferenceEngineClient

def main():
    client = FastInferenceEngineClient()

    # --- Serverless tier: quick pay-per-token call ---
    result = client.infer(
        prompt="Explain the benefits of disaggregated LLM inference for enterprise workloads.",
        model_id="llama-3.3-70b",
        deployment_tier="serverless"
    )
    print("=== Serverless Inference ===")
    print(f"Response : {result['response_text']}")
    print(f"Latency  : {result['latency_ms']}ms")
    print(f"Tokens   : {result['tokens_used']}")
    print(f"Cost     : ${result['tier_cost_usd']}")

    print()

    # --- On-demand tier: with custom LoRA adapter ---
    result2 = client.infer(
        prompt="Generate a legal contract summary for a SaaS subscription agreement.",
        model_id="deepseek-v4",
        deployment_tier="on-demand",
        lora_adapter="legal-summarizer-v3"
    )
    print("=== On-Demand + LoRA Inference ===")
    print(f"Response : {result2['response_text']}")
    print(f"Latency  : {result2['latency_ms']}ms")
    print(f"Tokens   : {result2['tokens_used']}")
    print(f"Cost     : ${result2['tier_cost_usd']}")

if __name__ == "__main__":
    main()
