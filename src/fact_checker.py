import sys
import asyncio
import aiohttp
import spacy
import importlib.util
import subprocess
import uuid
import json
import re

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
    BartTokenizer
)
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SESSION_ID = str(uuid.uuid4())


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
            f"Claim: {claim}\n"
            f"{evidence_str}"
            "Given the above evidence and results, explain whether the claim is supported, refuted, or uncertain. "
            "Be concise and cite the most relevant evidence."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(**inputs, min_length=30, max_length=150)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


async def call_wikipedia_mcp(claim, mcp_url="http://localhost:8000/mcp"):
    """Call the Wikipedia MCP server using the MCP protocol and return search results for the claim."""
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Call the 'search_wikipedia' tool
            result = await session.call_tool("search_wikipedia", {"query": claim})
            return result

async def call_twitter_mcp(claim, mcp_url="http://localhost:8002/mcp"):
    """Call the Twitter MCP server using the MCP protocol and return search results for the claim."""
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Call the 'search_tweets' tool
            result = await session.call_tool("search_tweets", {"query": claim})
            return result


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)


async def main():
    # Initialize components
    extractor = ClaimExtractor()
    verifier = FactVerifier()
    generator = OutputGenerator()

    # Input text (can be replaced with user input)
    text = "The Eiffel Tower is located in Berlin. Twitter was founded in 2006."
    claims = extractor.extract_claims(text)

    for claim in claims:
        print("="*50)
        print(f"CLAIM: {claim}")
        
        # Get Wikipedia evidence
        wiki_result = await call_wikipedia_mcp(claim)
        wiki_evidence = []
        if wiki_result and wiki_result.content:
            wiki_json = json.loads(wiki_result.content[0].text)
            wiki_evidence = [
                f"{item.get('title', '')}: {clean_html(item.get('snippet', ''))}"
                for item in wiki_json.get("results", [])
            ]
            wiki_evidence = wiki_evidence[:5]
        
        # Get Twitter evidence
        twitter_result = await call_twitter_mcp(claim)
        twitter_evidence = []
        if twitter_result and twitter_result.content:
            twitter_json = json.loads(twitter_result.content[0].text)
            for tweet in twitter_json.get("results", []):
                author = tweet.get("author", {})
                author_name = author.get("name", "Unknown") if author else "Unknown"
                author_username = author.get("username", "Unknown") if author else "Unknown"
                twitter_evidence.append(
                    f"Tweet by {author_name} (@{author_username}): {tweet.get('text', '')}"
                )
            twitter_evidence = twitter_evidence[:5]
        
        # Combine evidence from both sources
        evidence = wiki_evidence + twitter_evidence
        
        # Verify claims
        verified = verifier.verify(claim, evidence)
        print("\nEVIDENCE & VERIFICATION:")
        for (ev, label, score) in verified:
            print(f"- {ev}: {label} ({score:.2f})")

        # Post-processing: Improved logic for verdict
        CONFIDENCE_THRESHOLD = 0.80
        NEUTRAL_HIGH_CONF = 0.90
        CONTRADICTION_THRESHOLD = 0.80  # Require very high confidence for contradictions

        # Check for entailment above threshold
        entailments = [v for v in verified if v[1] == 'entailment' and v[2] >= CONFIDENCE_THRESHOLD]
        contradictions = [v for v in verified if v[1] == 'contradiction' and v[2] >= CONTRADICTION_THRESHOLD]
        neutrals_high = [v for v in verified if v[1] == 'neutral' and v[2] >= NEUTRAL_HIGH_CONF]

        if contradictions and max(c[2] for c in contradictions) >= 0.95:  # Very strong contradiction
            most_confident = max(contradictions, key=lambda x: x[2])
            verdict = 'REFUTED'
        elif entailments:
            most_confident = max(entailments, key=lambda x: x[2])
            verdict = 'SUPPORTED'
        elif contradictions:
            most_confident = max(contradictions, key=lambda x: x[2])
            verdict = 'REFUTED'
        elif len(neutrals_high) >= 3:  # If we have 3 or more high-confidence neutral results
            most_confident = max(neutrals_high, key=lambda x: x[2])
            verdict = 'UNCERTAIN'  # Changed from SUPPORTED to UNCERTAIN
        else:
            # Fallback to the most confident overall (even if it's neutral)
            most_confident = max(verified, key=lambda x: x[2])
            if most_confident[1] == 'contradiction':
                verdict = 'REFUTED'
            elif most_confident[1] == 'entailment':
                verdict = 'SUPPORTED'
            else:
                verdict = 'UNCERTAIN'

        print(f"\nSUMMARY VERDICT: {verdict}")

        # Improved explanation
        if most_confident:
            ev, label, score = most_confident
            explanation = (
                f"The claim: '{claim}' is most strongly {label.upper()} (confidence: {score:.2f}) "
                f"based on the evidence: {ev}.\n"
                f"Overall verdict: {verdict}."
            )
        else:
            explanation = f"No strong evidence found for the claim: '{claim}'. Verdict: {verdict}."
        print("\nEXPLANATION:")
        print(explanation)

if __name__ == "__main__":
    asyncio.run(main())
