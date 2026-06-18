"""
chain.py  —  5-Step Prompt Chaining Pipeline
============================================
Run the full pipeline from the command line:

    python chain.py sample_article.txt

Outputs:
    research_brief.md   — the final formatted brief with validation critique
    Terminal logs       — intermediate output from each step

Requirements:
    pip install ollama
    ollama pull llama3
    ollama serve          (in a separate terminal, if not running already)
"""

import json
import textwrap
import time
from pathlib import Path
from datetime import datetime

import ollama

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL       = "llama3"
OUTPUT_PATH = "research_brief.md"


# ── Helper Functions ───────────────────────────────────────────────────────────

def call_llm(user_prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to the local Ollama model and return the response text.
    Uses the messages array format so system and user roles are cleanly separated.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(model=MODEL, messages=messages)
    return response.message.content


def parse_json_response(raw: str) -> object:
    """
    Parse JSON from an LLM response, stripping markdown code fences if present.
    LLMs often add ```json ... ``` even when told not to — this strips them.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    return json.loads(cleaned)


def log_step(step_num: int, title: str, output: str) -> None:
    """
    Print a clearly labelled intermediate output.
    Inspecting intermediate outputs is the primary debugging tool in prompt chains.
    """
    border = "═" * 65
    print(f"\n{border}")
    print(f"  STEP {step_num}: {title}")
    print(border)
    for line in output.splitlines():
        print(textwrap.fill(line, width=70) if len(line) > 70 else line)
    print()


def load_article(path: str) -> str:
    """Load the raw article from a text file."""
    return Path(path).read_text(encoding="utf-8")


# ── Chain Steps ────────────────────────────────────────────────────────────────

def step1_extract(article: str) -> dict:
    """
    Step 1 — Entity Extraction.
    Input : Raw article text
    Output: Dict with keys: people, organisations, dates, locations
    """
    system = (
        "You are a precise named-entity extractor. "
        "You ONLY output valid JSON — no prose, no markdown fences, no explanations."
    )

    user = f"""Extract all named entities from the article below.

Return a single JSON object with exactly these four keys:
  "people"        : list of named individuals mentioned
  "organisations" : list of institutions, programmes, bodies
  "dates"         : list of specific dates, years, or time-ranges
  "locations"     : list of places, cities, or sites

Each list contains strings only. No nested objects.

ARTICLE:
{article}
"""

    raw = call_llm(user, system)
    entities = parse_json_response(raw)
    log_step(1, "EXTRACT — Named Entities (JSON)", json.dumps(entities, indent=2))
    return entities


def step2_analyse(article: str, entities: dict) -> list:
    """
    Step 2 — Claim Analysis.
    Input : Article + entities dict from Step 1
    Output: List of 3 dicts with keys: claim, confidence, justification
    """
    system = (
        "You are a rigorous fact-analyst. "
        "You ONLY output valid JSON — no prose, no markdown fences."
    )

    user = f"""You are analysing a historical article.
The following named entities have already been extracted:

{json.dumps(entities, indent=2)}

From the article below, identify the 3 MOST IMPORTANT factual claims.

Return a JSON array of exactly 3 objects, each with:
  "claim"          : a concise 1-sentence statement of the claim
  "confidence"     : one of: "high", "medium", or "low"
  "justification"  : 1-2 sentences explaining the confidence rating

Confidence guide:
  high   = well-documented historical fact, cross-verifiable
  medium = plausible but contains estimates or contested details
  low    = speculative, anecdotal, or single-source

ARTICLE:
{article}
"""

    raw = call_llm(user, system)
    claims = parse_json_response(raw)
    log_step(2, "ANALYSE — Top 3 Claims with Confidence (JSON)", json.dumps(claims, indent=2))
    return claims


def step3_synthesise(article: str, entities: dict, claims: list) -> str:
    """
    Step 3 — Synthesis.
    Input : Article + entities + claims
    Output: ~200-word prose summary
    """
    user = f"""You are a research analyst writing a concise briefing.

Using the information below, write a structured summary of approximately 200 words.
The summary must cover:
  1. What the article is about (context and significance)
  2. The key people and organisations involved
  3. The most important claims identified

Write in clear, professional prose. Do not use bullet points. Aim for ~200 words.

--- EXTRACTED ENTITIES ---
{json.dumps(entities, indent=2)}

--- KEY CLAIMS ---
{json.dumps(claims, indent=2)}

--- ORIGINAL ARTICLE ---
{article}
"""

    summary = call_llm(user)
    log_step(3, "SYNTHESISE — 200-Word Summary", summary)
    return summary


def step4_format(summary: str, entities: dict, claims: list) -> str:
    """
    Step 4 — Markdown Formatting.
    Input : Summary + entities + claims
    Output: Markdown string with four sections
    """
    user = f"""Convert the research content below into a structured Markdown brief.

The brief MUST contain exactly these four sections with these exact headings:

## Overview
[Insert the 200-word summary here, unchanged]

## Key Entities
[List all entities grouped by type: People, Organisations, Dates, Locations]

## Main Claims
[List the 3 claims as numbered items with their confidence rating in parentheses]

## Confidence Assessment
[A short paragraph discussing the overall reliability of the claims and any caveats]

Use proper Markdown formatting. Do not add any section not listed above.

--- 200-WORD SUMMARY ---
{summary}

--- ENTITIES ---
{json.dumps(entities, indent=2)}

--- CLAIMS ---
{json.dumps(claims, indent=2)}
"""

    brief = call_llm(user)
    log_step(4, "FORMAT — Markdown Research Brief", brief)
    return brief


def step5_validate(brief: str, article: str) -> str:
    """
    Step 5 — Validation / Critique.
    Input : Formatted brief + original article
    Output: Critique paragraph identifying unsupported or weak claims
    """
    user = f"""You are a sceptical academic fact-checker reviewing a research brief.
Your job is to compare the brief against the original source article and identify:
  1. Any claims in the brief that are NOT supported by the article
  2. Any important facts from the article that were omitted
  3. Any confidence ratings that seem too high or too low given the evidence
  4. Any vague, ambiguous, or potentially misleading statements

Write a single critique paragraph (3-5 sentences) addressed to the brief's author.
Be specific — cite the problematic claim or entity by name.
If the brief is fully accurate, say so and explain why you're satisfied.

--- RESEARCH BRIEF TO VALIDATE ---
{brief}

--- ORIGINAL SOURCE ARTICLE ---
{article}
"""

    critique = call_llm(user)
    log_step(5, "VALIDATE — Critique Paragraph", critique)
    return critique


# ── Assembly & Save ────────────────────────────────────────────────────────────

def assemble_and_save(brief: str, critique: str, output_path: str) -> str:
    """Combine brief and critique into a final Markdown document and save it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_document = f"""# Research Brief

> *Generated by a 5-step prompt chain using llama3 via Ollama*  
> *Generated at: {timestamp}*

---

{brief}

---

## Validation Critique

{critique}
"""
    Path(output_path).write_text(full_document, encoding="utf-8")
    return full_document


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(article_path: str, output_path: str = OUTPUT_PATH) -> dict:
    """
    Run the complete 5-step prompt chain.

    Parameters
    ----------
    article_path : Path to input .txt article
    output_path  : Path to write the output .md brief

    Returns
    -------
    dict  All intermediate outputs keyed by step name
    """
    print("🚀 Starting 5-step prompt chain pipeline")
    print("=" * 65)
    pipeline_start = time.time()

    article = load_article(article_path)
    print(f"📰 Article loaded: {len(article)} characters\n")

    t0 = time.time()
    print("⏳ Step 1: Extracting entities...")
    entities = step1_extract(article)
    print(f"   ✅ Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n⏳ Step 2: Analysing claims...")
    claims = step2_analyse(article, entities)
    print(f"   ✅ Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n⏳ Step 3: Synthesising summary...")
    summary = step3_synthesise(article, entities, claims)
    print(f"   ✅ Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n⏳ Step 4: Formatting Markdown brief...")
    brief = step4_format(summary, entities, claims)
    print(f"   ✅ Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n⏳ Step 5: Validating brief...")
    critique = step5_validate(brief, article)
    print(f"   ✅ Done in {time.time()-t0:.1f}s")

    document = assemble_and_save(brief, critique, output_path)

    total = time.time() - pipeline_start
    print(f"\n{'=' * 65}")
    print(f"🎉 Pipeline complete in {total:.1f}s")
    print(f"📄 Output saved to: {output_path}")

    return {
        "entities": entities,
        "claims":   claims,
        "summary":  summary,
        "brief":    brief,
        "critique": critique,
        "document": document,
    }


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chain.py <sample_article.txt>")
        sys.exit(1)

    run_pipeline(article_path=sys.argv[1], output_path=OUTPUT_PATH)