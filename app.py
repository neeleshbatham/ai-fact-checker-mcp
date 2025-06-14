import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio
import logging
from src.fact_checker import ClaimExtractor, FactVerifier, OutputGenerator, call_wikipedia_mcp, call_twitter_mcp
from mcp.shared.exceptions import McpError
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for fact checking
CONFIDENCE_THRESHOLD = 0.80  # Increased for better precision
NEUTRAL_HIGH_CONF = 0.90    # Increased to require stronger neutral evidence
CONTRADICTION_THRESHOLD = 0.85  # Increased for location claims
MIN_EVIDENCE_COUNT = 2  # Minimum number of pieces of evidence needed
WIKI_WEIGHT = 1.3  # Increased weight for Wikipedia evidence
TWITTER_WEIGHT = 0.7  # Decreased weight for Twitter evidence
LOCATION_CLAIM_THRESHOLD = 0.90  # New threshold for location-based claims

app = FastAPI(title="AI Fact Checker")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize components
extractor = ClaimExtractor()
verifier = FactVerifier()
generator = OutputGenerator()

# File to store history
HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {'history': []}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def clean_html(raw_html):
    """Clean HTML tags from text."""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

class FactCheckRequest(BaseModel):
    text: str

async def get_wikipedia_evidence(claim: str) -> List[str]:
    """Get Wikipedia evidence for a claim."""
    wiki_evidence = []
    try:
        wiki_result = await call_wikipedia_mcp(claim)
        if wiki_result and wiki_result.content:
            wiki_json = json.loads(wiki_result.content[0].text)
            for item in wiki_json.get("results", []):
                title = item.get('title', '')
                snippet = clean_html(item.get('snippet', ''))
                wiki_evidence.append(
                    f"Wikipedia - {title}: {snippet}"
                )
            wiki_evidence = wiki_evidence[:5]
            logger.info(f"Found {len(wiki_evidence)} pieces of Wikipedia evidence")
    except (McpError, Exception) as e:
        logger.warning(f"Failed to get Wikipedia evidence: {str(e)}")
    return wiki_evidence

async def get_twitter_evidence(claim: str) -> List[str]:
    """Get Twitter evidence for a claim."""
    twitter_evidence = []
    try:
        twitter_result = await call_twitter_mcp(claim)
        if twitter_result and twitter_result.content:
            twitter_json = json.loads(twitter_result.content[0].text)
            for tweet in twitter_json.get("results", []):
                author = tweet.get("author", {})
                author_name = author.get("name", "Unknown") if author else "Unknown"
                author_username = author.get("username", "Unknown") if author else "Unknown"
                timestamp = tweet.get("created_at", "")
                twitter_evidence.append(
                    f"Twitter - @{author_username} ({author_name}) - {timestamp}: {tweet.get('text', '')}"
                )
            twitter_evidence = twitter_evidence[:5]
            logger.info(f"Found {len(twitter_evidence)} pieces of Twitter evidence")
    except (McpError, Exception) as e:
        logger.warning(f"Failed to get Twitter evidence: {str(e)}")
    return twitter_evidence

async def process_claim(claim: str) -> Dict[str, Any]:
    """Process a single claim and return its verification results."""
    logger.info(f"Processing claim: {claim}")
    
    # Gather evidence concurrently
    wiki_evidence, twitter_evidence = await asyncio.gather(
        get_wikipedia_evidence(claim),
        get_twitter_evidence(claim)
    )
    
    # Combine evidence
    evidence = wiki_evidence + twitter_evidence
    
    # Verify claims
    verified = verifier.verify(claim, evidence)
    logger.info(f"Verification results: {verified}")
    
    # Process evidence with weights
    weighted_verified = []
    for ev, label, score in verified:
        weight = WIKI_WEIGHT if "Wikipedia" in ev else TWITTER_WEIGHT
        
        # Enhanced location-based claim detection
        if any(phrase in claim.lower() for phrase in ["in ", "located in", "part of", "within"]):
            # Extract locations from claim
            claim_parts = claim.lower().split()
            try:
                location_index = claim_parts.index("in")
                subject = " ".join(claim_parts[:location_index])
                location = " ".join(claim_parts[location_index+1:])
                
                # Check for explicit location contradictions
                if "located in" in ev.lower() or "in " in ev.lower():
                    # If evidence explicitly states the subject is in a different location
                    if subject in ev.lower() and location in ev.lower():
                        if f"{subject} in {location}" not in ev.lower():
                            label = 'contradiction'
                            score = max(score, 0.95)  # High confidence for explicit contradictions
                    
                    # If evidence states the subject is in a different location
                    if f"{subject} in" in ev.lower() and location not in ev.lower():
                        label = 'contradiction'
                        score = max(score, 0.9)
                    
                    # If evidence explicitly states the subject is not in the claimed location
                    if f"{subject} not in {location}" in ev.lower():
                        label = 'contradiction'
                        score = max(score, 0.95)
            
            except ValueError:
                pass  # No "in" found in claim
        
        weighted_score = score * weight
        weighted_verified.append((ev, label, weighted_score))

    # Get all evidence items
    all_evidence = [v for v in weighted_verified]
    
    # Check for explicit contradictions
    explicit_contradictions = []
    for ev, label, score in all_evidence:
        if "located in" in ev.lower():
            claim_location = claim.lower().split(" in ")[-1].strip()
            if claim_location not in ev.lower() and "in " in ev.lower():
                explicit_contradictions.append((ev, 'contradiction', max(score, 0.9)))

    # Combine contradictions
    contradictions = [v for v in weighted_verified if v[1] == 'contradiction' and v[2] >= CONTRADICTION_THRESHOLD]
    contradictions.extend(explicit_contradictions)
    
    entailments = [v for v in weighted_verified if v[1] == 'entailment' and v[2] >= CONFIDENCE_THRESHOLD]
    neutrals_high = [v for v in weighted_verified if v[1] == 'neutral' and v[2] >= NEUTRAL_HIGH_CONF]

    # Determine verdict
    if len(evidence) < MIN_EVIDENCE_COUNT:
        verdict = "UNKNOWN"
        explanation = "Not enough evidence to make a reliable determination."
    elif any(phrase in claim.lower() for phrase in ["in ", "located in", "part of", "within"]):
        # Special handling for location-based claims
        if contradictions:
            avg_contradiction_score = sum(score for _, _, score in contradictions) / len(contradictions)
            if avg_contradiction_score >= LOCATION_CLAIM_THRESHOLD:
                verdict = "FALSE"
                explanation = f"Found {len(contradictions)} pieces of strong evidence contradicting the location claim with high confidence."
            else:
                verdict = "LIKELY FALSE"
                explanation = f"Found {len(contradictions)} pieces of evidence contradicting the location claim, but with moderate confidence."
        elif entailments:
            avg_entailment_score = sum(score for _, _, score in entailments) / len(entailments)
            if avg_entailment_score >= LOCATION_CLAIM_THRESHOLD:
                verdict = "TRUE"
                explanation = f"Found {len(entailments)} pieces of strong evidence supporting the location claim with high confidence."
            else:
                verdict = "LIKELY TRUE"
                explanation = f"Found {len(entailments)} pieces of evidence supporting the location claim, but with moderate confidence."
        else:
            verdict = "FALSE"  # Default to FALSE for location claims without strong evidence
            explanation = "No strong evidence found to support the location claim."
    else:
        # Regular claim handling
        if contradictions:
            avg_contradiction_score = sum(score for _, _, score in contradictions) / len(contradictions)
            if avg_contradiction_score >= CONTRADICTION_THRESHOLD:
                verdict = "FALSE"
                explanation = f"Found {len(contradictions)} pieces of strong evidence contradicting the claim with high confidence."
            else:
                verdict = "LIKELY FALSE"
                explanation = f"Found {len(contradictions)} pieces of evidence contradicting the claim, but with moderate confidence."
        elif entailments:
            avg_entailment_score = sum(score for _, _, score in entailments) / len(entailments)
            if avg_entailment_score >= CONFIDENCE_THRESHOLD:
                verdict = "TRUE"
                explanation = f"Found {len(entailments)} pieces of strong evidence supporting the claim with high confidence."
            else:
                verdict = "LIKELY TRUE"
                explanation = f"Found {len(entailments)} pieces of evidence supporting the claim, but with moderate confidence."
        elif neutrals_high and len(neutrals_high) >= 3:
            verdict = "NEUTRAL"
            explanation = f"Found {len(neutrals_high)} pieces of evidence that neither strongly support nor contradict the claim."
        else:
            verdict = "UNCERTAIN"
            explanation = "The evidence is mixed or not strong enough to make a definitive determination."

    # Generate detailed explanation
    detailed_explanation = generator.generate(claim, weighted_verified)

    return {
        "claim": claim,
        "verdict": verdict,
        "explanation": detailed_explanation,
        "evidence": [
            {
                "text": ev,
                "label": label,
                "score": float(score)
            }
            for ev, label, score in weighted_verified
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check")
async def check_fact(request: FactCheckRequest):
    if not request.text:
        return JSONResponse(
            status_code=400,
            content={"error": "No text provided"}
        )

    logger.info(f"Received text: {request.text}")

    # Extract claims
    claims = extractor.extract_claims(request.text)
    logger.info(f"Extracted claims: {claims}")

    if not claims:
        # Clean and normalize the text
        cleaned_text = request.text.strip()
        cleaned_text = re.sub(r'^(i think|i believe|maybe|perhaps|probably|possibly)\s+', '', cleaned_text.lower())
        cleaned_text = re.sub(r'[.!?]+$', '', cleaned_text)
        logger.info("No claims extracted, treating cleaned text as a single claim")
        claims = [cleaned_text]

    # Process all claims concurrently
    results = await asyncio.gather(*[process_claim(claim) for claim in claims])

    # Save to history
    history = load_history()
    history['history'].append({
        'id': str(uuid.uuid4()),
        'text': request.text,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })
    save_history(history)

    return JSONResponse(content={"results": results})

@app.get("/history")
async def get_history():
    history = load_history()
    return JSONResponse(content=history)

@app.get("/history/{id}")
async def get_history_item(id: str):
    history = load_history()
    for item in history['history']:
        if item['id'] == id:
            return JSONResponse(content=item)
    return JSONResponse(status_code=404, content={"error": "History item not found"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 