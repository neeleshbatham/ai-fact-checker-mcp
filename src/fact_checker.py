import sys
import asyncio
import aiohttp
import spacy
import importlib.util
import subprocess

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
    BartTokenizer
)
from mcp import ClientSession



def ensure_spacy_model(model_name="en_core_web_sm"):
    # Check if the model is already installed
    if importlib.util.find_spec(model_name) is None:
        print(f"Downloading spaCy model '{model_name}'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])




class ClaimExtractor:
    def __init__(self):
        ensure_spacy_model("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm")
    def extract_claims(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if any(tok.pos_ == "VERB" for tok in sent)]

class FactVerifier:
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
    def verify(self, claim, evidence_list):
        results = []
        for evidence in evidence_list:
            res = self.nlp(f"{claim} </s></s> {evidence}")[0]
            results.append((evidence, res['label'], float(res['score'])))
        return results

class OutputGenerator:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
    def generate(self, claim, verified_evidence):
        evidence_str = ""
        for ev, label, score in verified_evidence:
            evidence_str += f"Evidence: {ev}\nResult: {label} (Confidence: {score:.2f})\n"
        prompt = (
            f"Claim: {claim}\n{evidence_str}"
            "Based on the above, explain if the claim is supported, refuted, or uncertain. Cite evidence."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(**inputs, min_length=30, max_length=150)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


async def get_evidence_from_all(claim, endpoints):
    all_evidence = []
    for name, url in endpoints.items():
        tool = "search_wikipedia" if name == "wikipedia" else "search_x"
        # The tool call endpoint per MCP API: /tools/{tool}/call
        tool_url = f"{url}/tools/{tool}/call"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(tool_url, json={"query": claim}) as resp:
                    if resp.status == 200:
                        resp_json = await resp.json()
                        # MCP SDK returns a dict with 'result' and 'success' keys
                        if resp_json.get("success", False):
                            all_evidence.extend([f"[{name}] {e}" for e in resp_json.get("result", [])])
                        else:
                            all_evidence.append(f"[{name}] MCP error: {resp_json.get('error_message')}")
                    else:
                        all_evidence.append(f"[{name}] HTTP error: {resp.status}")
            except Exception as e:
                all_evidence.append(f"[{name}] Exception: {e}")
    return all_evidence

async def main():
    # Initialize components

# Ensure model is present
    extractor = ClaimExtractor()
    verifier = FactVerifier()
    generator = OutputGenerator()

    # Define MCP endpoints (adjust ports if you change them)
    MCP_ENDPOINTS = {
        "wikipedia": "http://localhost:8001",
        "x": "http://localhost:8002"
    }

    # Input text (can be replaced with user input)
    text = "The Eiffel Tower is located in Berlin. Twitter was founded in 2006."
    claims = extractor.extract_claims(text)

    for claim in claims:
        print("="*50)
        print(f"CLAIM: {claim}")
        evidence = await get_evidence_from_all(claim, MCP_ENDPOINTS)
        print("EVIDENCE:", evidence)
        verified = verifier.verify(claim, evidence)
        print("VERIFICATION:", verified)
        explanation = generator.generate(claim, verified)
        print("EXPLANATION:", explanation)

if __name__ == "__main__":
    asyncio.run(main())
