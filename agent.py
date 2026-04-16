"""
agent.py — Multimodal ASD Trait Detection AI Agent (v4)
---------------------------------------------------------
LangGraph multi-agent system with full companion capabilities.

ARCHITECTURE
============
  supervisor_agent    — Intent classification + routing (NO JSON bleed)
  questionnaire_agent — Q-CHAT-10 + XGBoost (98.6% accuracy)
  text_agent          — Free-text + BERT (88% accuracy) + trait extraction
  story_agent         — Personalised social story generation (NEW)
  schedule_agent      — Visual routine schedule generation (NEW)
  guidance_agent      — Warm, contextual post-result conversation

STATE FIELDS
============
  messages        — Full conversation history (LangGraph managed)
  answer          — Latest reply shown to user
  stage           — FSM stage name
  last_assessment — Full ML result context for guidance
  child_profile   — Name, age, sensory triggers, skills (NEW)
  session_history — All past assessments in this session (NEW)

FSM STAGES
==========
  idle                 → initial greeting, waiting for "yes"
  collect_profile      → gathering child name + age (NEW)
  choose_method        → waiting for method choice
  awaiting_answers     → questionnaire shown, waiting for 11 values
  awaiting_description → text prompt shown, waiting for description
  guidance             → post-result Q&A loop
  story_request        → waiting for situation description (NEW)
  schedule_request     → waiting for routine name (NEW)

FIXES FROM v3
=============
  ✓ JSON bleed eliminated — supervisor routes via ROUTE: tag, never shown to user
  ✓ "Another assessment" uses LLM intent classifier, never keyword-matching
  ✓ Intent detection uses LLM classification, not brittle if/elif
  ✓ Trait detection uses constrained LLM with taxonomy guard (not free-form)
  ✓ Child profile injects personal context into all guidance responses
  ✓ Session history enables progress review
"""

import re
import json
import numpy as np
import pandas as pd
import torch
import os

from datetime import datetime
from typing import Annotated, Literal, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    groq_api_key, BERT_MAX_LENGTH, FEATURE_COLS_DEFAULT,
    FEATURE_COLS_PATH, MATERIALS_PATH, SBERT_MODEL_NAME
)


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


# ── Load feature column order ─────────────────────────────────────────────────
def _load_feature_cols() -> list:
    try:
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)
    except Exception:
        return FEATURE_COLS_DEFAULT

FEATURE_COLS = _load_feature_cols()


# ══════════════════════════════════════════════════════════════════════════════
# State definition
# ══════════════════════════════════════════════════════════════════════════════

class ChildProfile(TypedDict):
    name            : str
    age             : str          # e.g. "4 years"
    triggers        : List[str]    # e.g. ["loud noises", "bright lights"]
    skills_working  : List[str]    # e.g. ["turn-taking", "eye contact"]

class State(TypedDict):
    messages        : Annotated[list, add_messages]
    answer          : str
    stage           : str
    last_assessment : Optional[dict]
    child_profile   : Optional[ChildProfile]
    session_history : List[dict]          # list of completed assessment dicts


# ══════════════════════════════════════════════════════════════════════════════
# Semantic retrieval engine (FAISS + sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalEngine:
    """
    Embeds a library of social stories and visual schedules.
    Supports semantic search so the agent can find the best match
    before generating a personalised version.
    """

    def __init__(self):
        self._ready    = False
        self._model    = None
        self._index    = None
        self._docs     = []

    def load(self):
        """Lazy-load — called once from build_agent."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self._model = SentenceTransformer(SBERT_MODEL_NAME)
            self._docs  = []

            # Load pre-built library
            if os.path.exists(MATERIALS_PATH):
                with open(MATERIALS_PATH) as f:
                    data = json.load(f)
                stories   = data.get("social_stories", [])
                schedules = data.get("visual_schedules", [])
            else:
                stories, schedules = _default_stories(), _default_schedules()

            for s in stories:
                self._docs.append({
                    "type"    : "social_story",
                    "title"   : s["title"],
                    "content" : s["content"],
                    "triggers": s.get("triggers", []),
                    "skills"  : s.get("skills", []),
                    "age"     : s.get("age_range", ""),
                })
            for sc in schedules:
                self._docs.append({
                    "type"   : "visual_schedule",
                    "title"  : sc["title"],
                    "content": " → ".join(sc["steps"]),
                    "context": sc.get("context", ""),
                    "age"    : sc.get("age_range", ""),
                })

            texts = [f"{d['title']}: {d['content']}" for d in self._docs]
            embs  = self._model.encode(texts).astype("float32")

            dim         = embs.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            self._index.add(embs)
            self._ready = True
            print(f"[RetrievalEngine] Loaded {len(self._docs)} materials.")
        except Exception as e:
            print(f"[RetrievalEngine] Could not load (FAISS/SBERT unavailable): {e}")
            self._ready = False

    def search(self, query: str, k: int = 2, doc_type: str = None) -> list:
        """Return up to k most similar documents."""
        if not self._ready:
            return []
        try:
            q_emb = self._model.encode([query]).astype("float32")
            distances, indices = self._index.search(q_emb, k * 3)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self._docs):
                    doc = self._docs[idx].copy()
                    if doc_type and doc["type"] != doc_type:
                        continue
                    doc["score"] = float(1.0 / (1.0 + dist))
                    results.append(doc)
                    if len(results) >= k:
                        break
            return results
        except Exception as e:
            print(f"[RetrievalEngine.search] {e}")
            return []

    def detect_triggers(self, text: str) -> list:
        """Return sensory trigger words found in text."""
        TRIGGER_WORDS = [
            "loud noise", "bright light", "strong smell", "crowded",
            "unexpected touch", "flickering", "sudden movement",
            "scratchy", "food texture", "buzzing", "clippers",
            "sirens", "hand dryer",
        ]
        found = [t for t in TRIGGER_WORDS if t in text.lower()]
        return found

retrieval_engine = RetrievalEngine()


# ══════════════════════════════════════════════════════════════════════════════
# Default library (fallback if asd_materials.json absent)
# ══════════════════════════════════════════════════════════════════════════════

def _default_stories():
    return [
        {"title": "Getting a Haircut",
         "content": "Today I am getting a haircut. The barber will use scissors. The scissors make a snipping sound. The clippers buzz. It might tickle a little. When they finish, my hair will look neat. I can ask for a break if I need one.",
         "age_range": "3-8", "triggers": ["buzzing sounds", "touch sensitivity"], "skills": ["grooming", "tolerance"]},
        {"title": "Going to the Grocery Store",
         "content": "Today I am going to the grocery store. The store has bright lights and many people. If it gets too loud, I can use my headphones. I can help by holding the list. When we finish, we go to the car.",
         "age_range": "4-6", "triggers": ["loud noises", "bright lights", "crowds"], "skills": ["shopping", "coping"]},
        {"title": "Calming Down When Upset",
         "content": "Sometimes I feel upset. My face gets hot and my heart beats fast. When this happens, I take 3 deep breaths. I count to 10 slowly. I go to my quiet corner. When I feel calm I can use my words.",
         "age_range": "4-10", "triggers": ["emotional overwhelm", "sensory overload"], "skills": ["self-regulation"]},
        {"title": "Making a New Friend",
         "content": "I see someone I want to be friends with. I can say Hi, my name is. What is your name? I can ask what they like to play. I listen and then tell them what I like. If they do not want to talk that is okay.",
         "age_range": "6-8", "triggers": ["rejection", "social uncertainty"], "skills": ["greeting", "conversation"]},
        {"title": "Eating Lunch at School",
         "content": "At lunchtime I line up with my class. I get my tray and find a seat. The cafeteria is noisy. I will try to eat my lunch. If it is too loud I can ask to eat in a quieter place.",
         "age_range": "5-10", "triggers": ["loud noises", "strong smells"], "skills": ["eating", "self-advocacy"]},
    ]

def _default_schedules():
    return [
        {"title": "Morning Routine", "age_range": "4-10", "context": "home",
         "steps": ["Wake up","Use the bathroom","Wash face and hands","Get dressed","Eat breakfast","Brush teeth","Pack backpack","Put on shoes","Go to school"]},
        {"title": "Bedtime Routine", "age_range": "3-8", "context": "home",
         "steps": ["Take a bath","Put on pajamas","Brush teeth","Read a story","Say goodnight","Turn off lights","Go to sleep"]},
        {"title": "School Day", "age_range": "5-12", "context": "school",
         "steps": ["Morning meeting","Reading time","Snack","Math","Lunch","Recess","Science","Art","Clean up","Go home"]},
        {"title": "Grocery Trip", "age_range": "4-8", "context": "community",
         "steps": ["Get the shopping list","Put on shoes","Drive to the store","Get a cart","Find items on the list","Pay at the counter","Return the cart","Drive home","Put groceries away"]},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Trait taxonomy + severity scoring
# ══════════════════════════════════════════════════════════════════════════════

TRAIT_TAXONOMY = [
    "Eye Contact", "Attention Response", "Follow Pointing",
    "Word Repetition", "Repetitive Behavior", "Focused Attention",
    "Emotional Empathy", "Sharing Interest", "Sign Communication",
    "Change Reaction", "Noise Sensitivity", "Finger Movements",
    "Toy Arranging", "Tiptoe Flapping",
]

HIGH_CONCERN     = {"Eye Contact","Attention Response","Follow Pointing",
                    "Word Repetition","Emotional Empathy","Sharing Interest"}
MODERATE_CONCERN = {"Repetitive Behavior","Change Reaction",
                    "Sign Communication","Focused Attention"}
MONITOR_CONCERN  = {"Noise Sensitivity","Finger Movements",
                    "Toy Arranging","Tiptoe Flapping"}

TRAIT_DESCRIPTIONS = {
    "Eye Contact"       : "avoiding or rarely making eye contact with others",
    "Attention Response": "not responding when their name is called",
    "Follow Pointing"   : "not following when someone points to something",
    "Word Repetition"   : "repeating words or phrases (echolalia)",
    "Repetitive Behavior": "doing repetitive movements or actions",
    "Focused Attention" : "unusually intense focus on specific objects or topics",
    "Emotional Empathy" : "limited response to other people's emotions",
    "Sharing Interest"  : "not pointing to share things with others",
    "Sign Communication": "limited use of gestures to communicate",
    "Change Reaction"   : "strong distress in response to changes in routine",
    "Noise Sensitivity" : "being distressed by certain sounds or noise levels",
    "Finger Movements"  : "unusual hand or finger movements",
    "Toy Arranging"     : "lining up or arranging toys repetitively",
    "Tiptoe Flapping"   : "walking on tiptoes or flapping hands",
}


def compute_concern_level(traits_present: list) -> dict:
    high     = [t for t in traits_present if t in HIGH_CONCERN]
    moderate = [t for t in traits_present if t in MODERATE_CONCERN]
    monitor  = [t for t in traits_present if t in MONITOR_CONCERN]

    if len(high) >= 2:
        level     = "High"
        rationale = f"{len(high)} core social-communication traits detected — professional evaluation is recommended."
    elif len(high) == 1 or len(moderate) >= 2:
        level     = "Moderate"
        rationale = f"{len(high)} high-concern and {len(moderate)} moderate-concern traits noted — monitoring advised."
    elif traits_present:
        level     = "Monitor"
        rationale = f"{len(traits_present)} sensory/motor trait(s) noted — continue observation."
    else:
        level     = "Low"
        rationale = "No specific behavioural traits clearly identified in the description."

    return {"level": level, "rationale": rationale,
            "high": high, "moderate": moderate, "monitor": monitor}


TRAIT_EXTRACTION_SYSTEM = """
You are a clinical behavioural analysis assistant for ASD screening.
A parent has described their child's behaviour. Identify which of the 14 named
traits are PRESENT, ABSENT, or UNCERTAIN.

RULES:
- Only mark traits directly stated or clearly implied in the text
- Do NOT invent traits not mentioned
- Use EXACT trait names as listed — spelling must match precisely
- If briefly mentioned, mark as present. If unclear, mark as uncertain.

THE 14 TRAITS (use these exact names only):
1.  Eye Contact           — avoids or rarely makes eye contact
2.  Attention Response    — does not respond when name is called
3.  Follow Pointing       — does not follow when someone points
4.  Word Repetition       — repeats words or phrases (echolalia)
5.  Repetitive Behavior   — repetitive movements or actions
6.  Focused Attention     — unusually intense focus on specific things
7.  Emotional Empathy     — little response to others' emotions
8.  Sharing Interest      — does not point to share things
9.  Sign Communication    — limited use of gestures
10. Change Reaction       — reacts strongly to changes in routine
11. Noise Sensitivity     — distressed by certain sounds
12. Finger Movements      — unusual hand or finger movements
13. Toy Arranging         — lines up or arranges toys repetitively
14. Tiptoe Flapping       — walks on tiptoes or flaps hands

Reply ONLY with valid JSON — no other text, no markdown backticks:
{"traits_present": ["Exact Trait Name", ...], "traits_absent": [], "traits_uncertain": ["Exact Trait Name", ...]}
"""


def extract_traits(description: str) -> dict:
    """
    Identify which of the 14 behavioural traits appear in the text.
    The LLM is constrained to the fixed taxonomy and cannot invent new traits.
    Case-insensitive fallback normalisation prevents minor capitalisation drops.
    """
    trait_lower_map = {t.lower(): t for t in TRAIT_TAXONOMY}

    def normalise(raw_list: list) -> list:
        valid = set(TRAIT_TAXONOMY)
        result = []
        for t in raw_list:
            if t in valid:
                result.append(t)
            elif t.lower() in trait_lower_map:
                result.append(trait_lower_map[t.lower()])
        return result

    try:
        raw  = llm.invoke([
            SystemMessage(content=TRAIT_EXTRACTION_SYSTEM),
            HumanMessage(content=f"Description: {description}"),
        ])
        text = raw.content.strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        text = m.group(0) if m else text
        data = json.loads(text)
        return {
            "traits_present"  : normalise(data.get("traits_present",   [])),
            "traits_absent"   : normalise(data.get("traits_absent",    [])),
            "traits_uncertain": normalise(data.get("traits_uncertain", [])),
        }
    except Exception as e:
        print(f"[trait extraction error] {e}")
        return {"traits_present": [], "traits_absent": [], "traits_uncertain": []}


def format_trait_report(trait_result: dict, concern: dict) -> str:
    present   = trait_result.get("traits_present",   [])
    uncertain = trait_result.get("traits_uncertain", [])

    lines = ["\n\n---\n### 🔍 Behavioural Trait Analysis\n"]

    if present:
        lines.append(f"**{len(present)} trait(s) identified:**\n")
        for t in present:
            if t in HIGH_CONCERN:
                tag = "🔴 High concern"
            elif t in MODERATE_CONCERN:
                tag = "🟡 Moderate concern"
            else:
                tag = "🔵 Monitor"
            desc = TRAIT_DESCRIPTIONS.get(t, "")
            lines.append(f"- **{t}** ({tag}) — *{desc}*")
    else:
        lines.append("- No specific traits clearly identified in the description.")

    if uncertain:
        lines.append(f"\n**Needs more detail:** {', '.join(uncertain)}")

    lines.append(f"\n**Overall concern level: {concern['level']}**  \n*{concern['rationale']}*")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Intent classifier — replaces all brittle keyword matching
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM = """
Classify the user's message into exactly ONE of these intents.
Reply with ONLY the intent label — no other text.

Intents:
  START_SCREENING   — user wants to begin ASD screening (says yes, start, screen, assess, check, begin, ok, sure)
  QUESTIONNAIRE     — user wants the structured questionnaire / Q-CHAT-10
  TEXT_ANALYSIS     — user wants to type a behaviour description
  SOCIAL_STORY      — user wants a social story (mentions story, narrative, situation)
  VISUAL_SCHEDULE   — user wants a visual schedule or routine (morning, bedtime, steps, schedule, routine)
  RESTART           — user wants another / new assessment after getting results
  SHOW_HISTORY      — user wants to see past results / history / progress
  EXIT              — user says goodbye / exit / quit / done / thank you
  FOLLOW_UP         — user is asking a question about results or ASD in general
  OFF_TOPIC         — completely unrelated message
"""

def classify_intent(user_text: str, stage: str) -> str:
    """
    Use LLM to classify intent into a fixed set of labels.
    Returns one of the intent labels above. This replaces ALL keyword matching.
    """
    try:
        resp = llm.invoke([
            SystemMessage(content=INTENT_SYSTEM),
            SystemMessage(content=f"Current conversation stage: {stage}"),
            HumanMessage(content=user_text),
        ])
        intent = resp.content.strip().upper()
        valid  = {"START_SCREENING","QUESTIONNAIRE","TEXT_ANALYSIS",
                  "SOCIAL_STORY","VISUAL_SCHEDULE","RESTART",
                  "SHOW_HISTORY","EXIT","FOLLOW_UP","OFF_TOPIC"}
        return intent if intent in valid else "FOLLOW_UP"
    except Exception as e:
        print(f"[intent classifier error] {e}")
        return "FOLLOW_UP"


# ══════════════════════════════════════════════════════════════════════════════
# Core prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def questionnaire_predict(xgboost_model, features: list) -> tuple:
    sample_df  = pd.DataFrame([features], columns=FEATURE_COLS)
    proba_raw  = xgboost_model.predict_proba(sample_df)[0]
    p_non      = float(proba_raw[0]) * 100
    p_asd      = float(proba_raw[1]) * 100
    pred       = int(np.argmax(proba_raw))
    label      = "ASD" if pred == 1 else "Non-ASD"
    confidence = float(proba_raw[pred]) * 100

    result_md = (
        f"### ✅ Questionnaire Analysis Complete\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Assessment** | {label} |\n"
        f"| **Confidence** | {confidence:.1f}% |\n"
        f"| ASD probability | {p_asd:.1f}% |\n"
        f"| Non-ASD probability | {p_non:.1f}% |\n\n"
        f"> ⚠️ **Disclaimer:** This is a screening tool only, not a medical diagnosis. "
        f"Always consult a qualified healthcare professional."
    )
    return result_md, label, confidence, p_asd, p_non


def text_predict(bert_tokenizer, bert_model_obj, description: str) -> tuple:
    inputs = bert_tokenizer(
        description, return_tensors="pt",
        truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
    )
    with torch.no_grad():
        logits = bert_model_obj(**inputs).logits
        proba  = torch.softmax(logits, dim=1)[0]
        pred   = torch.argmax(proba).item()

    label      = "ASD" if pred == 1 else "Non-ASD"
    p_non      = float(proba[0].item()) * 100
    p_asd      = float(proba[1].item()) * 100
    confidence = float(proba[pred].item()) * 100
    preview    = description[:200] + ("..." if len(description) > 200 else "")

    result_md = (
        f"### ✅ Text Analysis Complete\n\n"
        f"> *\"{preview}\"*\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Assessment** | {label} |\n"
        f"| **Confidence** | {confidence:.1f}% |\n"
        f"| ASD probability | {p_asd:.1f}% |\n"
        f"| Non-ASD probability | {p_non:.1f}% |\n\n"
        f"> ⚠️ **Disclaimer:** This is a screening tool only, not a medical diagnosis. "
        f"Always consult a qualified healthcare professional."
    )
    return result_md, label, confidence, p_asd, p_non


def parse_answers(raw: str):
    try:
        values = [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        return "Please use only 0 or 1 separated by commas.\nExample: 0,1,0,1,1,0,0,0,1,0,0"
    if len(values) != 11:
        return f"I need exactly 11 values but got {len(values)}. Please re-enter all 11 answers."
    bad = [v for v in values if v not in (0, 1)]
    if bad:
        return f"All values must be 0 or 1. Found: {bad}. Please re-enter."
    return values


def is_valid_description(text: str) -> bool:
    check = llm.invoke([
        SystemMessage(content=(
            "You are a strict input filter for a medical screening tool.\n"
            "Reply ONLY with yes or no — nothing else, no punctuation.\n"
            "Is the following text a description of a child's behaviour, "
            "development, communication, or social traits?"
        )),
        HumanMessage(content=text),
    ])
    return check.content.strip().lower().startswith("yes")


# ══════════════════════════════════════════════════════════════════════════════
# Profile extraction helper
# ══════════════════════════════════════════════════════════════════════════════

PROFILE_EXTRACTION_SYSTEM = """
Extract the child's profile from the user's message.
Return ONLY valid JSON with these fields (use empty string / empty list if unknown):
{
  "name": "child's first name or empty string",
  "age": "age as string e.g. '4 years' or empty string",
  "triggers": ["list of sensory triggers if mentioned"],
  "skills_working": ["list of skills or goals being worked on if mentioned"]
}
"""

def extract_profile(text: str) -> dict:
    try:
        raw  = llm.invoke([
            SystemMessage(content=PROFILE_EXTRACTION_SYSTEM),
            HumanMessage(content=text),
        ])
        t = raw.content.strip().strip("`")
        if t.startswith("json"):
            t = t[4:].strip()
        m = re.search(r"\{.*\}", t, re.DOTALL)
        t = m.group(0) if m else t
        d = json.loads(t)
        return {
            "name"          : d.get("name", ""),
            "age"           : d.get("age", ""),
            "triggers"      : d.get("triggers", []),
            "skills_working": d.get("skills_working", []),
        }
    except Exception as e:
        print(f"[profile extraction error] {e}")
        return {"name": "", "age": "", "triggers": [], "skills_working": []}


# ══════════════════════════════════════════════════════════════════════════════
# System prompts
# ══════════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """
You are a warm, knowledgeable ASD screening assistant.

Your job is to greet parents and guide them through screening for their child.
Speak naturally, like a helpful expert. Never use JSON. Never mention routing.

When a new conversation starts:
  - Greet warmly
  - Ask if they would like to begin a screening
  - Offer to collect brief info about their child first

Capabilities you can mention:
  1. ASD screening — questionnaire (Q-CHAT-10) or free-text description
  2. Social stories — personalised narratives for difficult situations
  3. Visual schedules — step-by-step routine breakdowns
  4. Progress tracking — reviewing past screening results

Always be warm, non-alarmist, and reassuring.
Always recommend professional consultation for any concerning result.
"""

GUIDANCE_SYSTEM = """
You are a compassionate ASD screening support assistant.
A screening assessment has just been completed. You have full context below.

Your role:
1. Answer the parent's questions about results in simple, friendly language
2. Explain what each identified trait means for a child of this age
3. Suggest practical activities or observations to try at home
4. Advise when and how to seek a professional developmental assessment
5. Provide emotional support — warm, reassuring, non-alarmist

Rules:
- NEVER upgrade a screening result to a clinical diagnosis
- ALWAYS recommend professional consultation for concerning results
- ONLY discuss traits identified in THIS specific assessment
- Stay factually grounded — do not speculate beyond the data
- Be conversational, not a bulleted list of facts
- Mention the child by name if the profile has one

Assessment context:
{context}

Child profile:
{profile}
"""

STORY_SYSTEM = """
You generate personalised social stories for children with ASD.

A social story is a short, first-person narrative that describes a situation
step-by-step. It uses simple sentences, reassuring language, and coping strategies.

Rules:
- Write in first person ("I am going to...")
- Use short sentences (max 15 words each)
- Include what will happen, what the child can do, and a positive outcome
- Mention specific sensory experiences the child might encounter
- If triggers are known, include coping strategies for them
- End with a positive affirmation

Format:
## 📖 Social Story: {title}

[Body — 8-12 short first-person sentences]

---
*This story can be read together before the situation occurs.*
"""

SCHEDULE_SYSTEM = """
You generate clear, numbered visual schedules for children with ASD.

Rules:
- Use simple, action-first language for each step ("Wash hands", not "You should wash your hands")
- Include 6-12 steps appropriate for the routine
- Add brief sensory notes where helpful (e.g., "Brush teeth (2 minutes, timer helps)")
- Keep it positive and encouraging

Format:
## 🗓️ Visual Schedule: {title}

1. Step one
2. Step two
...

---
*You can print this and display it where your child can see it.*
"""


# ══════════════════════════════════════════════════════════════════════════════
# Agent factory — all nodes as closures with ML models in scope
# ══════════════════════════════════════════════════════════════════════════════

def build_supervisor_fns(xgboost_model, bert_tokenizer, bert_model_obj):

    # ── Helper: build profile string for LLM context ─────────────────────────

    def profile_context(profile: dict) -> str:
        if not profile:
            return "No profile collected yet."
        parts = []
        if profile.get("name"):
            parts.append(f"Child's name: {profile['name']}")
        if profile.get("age"):
            parts.append(f"Age: {profile['age']}")
        if profile.get("triggers"):
            parts.append(f"Known sensory triggers: {', '.join(profile['triggers'])}")
        if profile.get("skills_working"):
            parts.append(f"Skills being practiced: {', '.join(profile['skills_working'])}")
        return "\n".join(parts) if parts else "No profile details available."

    # ── Supervisor ────────────────────────────────────────────────────────────

    def supervisor_agent(state: State) -> State:
        stage    = state.get("stage", "idle")
        messages = state.get("messages", [])
        profile  = state.get("child_profile") or {}

        # Fast-path: skip LLM for data-collection stages
        if stage in ("awaiting_answers", "awaiting_description",
                     "story_request", "schedule_request"):
            state["answer"] = ""
            return state

        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        # ── Intent classification (replaces all keyword matching) ─────────────
        intent = "FOLLOW_UP"
        if last_human:
            intent = classify_intent(last_human, stage)

        # ── Route on intent ───────────────────────────────────────────────────

        if intent == "EXIT":
            name   = profile.get("name", "")
            suffix = f" Give {name} a hug from me." if name else ""
            reply  = (f"Thank you for using the ASD Screening Assistant.{suffix} "
                      f"Take care, and don't hesitate to consult a healthcare "
                      f"professional if you have concerns. Goodbye! 💙")
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "idle"
            return state

        if intent == "RESTART":
            reply = (
                "Of course! Let's run another screening.\n\n"
                "Which method would you like to use?\n\n"
                "- **Questionnaire** — Answer 11 structured Q-CHAT-10 questions (0 or 1)\n"
                "- **Text description** — Describe your child's behaviour in your own words\n\n"
                "Type **questionnaire** or **text** to begin."
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "choose_method"
            return state

        if intent == "SHOW_HISTORY":
            history = state.get("session_history", [])
            if not history:
                reply = "No assessments have been completed yet in this session."
            else:
                lines = ["### 📊 Session History\n"]
                for i, h in enumerate(history, 1):
                    lines.append(
                        f"**Assessment {i}** — {h.get('date','')}\n"
                        f"- Method: {h.get('method','').capitalize()}\n"
                        f"- Result: **{h.get('label','')}** ({h.get('confidence',0):.1f}% confidence)\n"
                        + (f"- Concern level: {h.get('concern_level','')}\n" if h.get('concern_level') else "")
                        + (f"- Traits: {', '.join(h.get('traits_present', [])) or 'None identified'}\n")
                    )
                reply = "\n".join(lines)
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            return state

        if intent == "SOCIAL_STORY":
            reply = (
                "I'd be happy to create a personalised social story! 📖\n\n"
                "Please describe the **situation** your child finds difficult. "
                "For example: *'My son panics at haircuts'* or "
                "*'My daughter gets upset when we go to loud places.'*\n\n"
                "**Describe the situation:**"
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "story_request"
            return state

        if intent == "VISUAL_SCHEDULE":
            reply = (
                "Great! I'll create a visual schedule. 🗓️\n\n"
                "Which routine would you like? For example:\n"
                "*morning routine, bedtime routine, school day, grocery trip, "
                "getting ready for bath, dentist visit...*\n\n"
                "**Name the routine:**"
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "schedule_request"
            return state

        if intent == "OFF_TOPIC":
            reply = (
                "I'm sorry, I can only assist with ASD-related topics. "
                "I'm here to help with:\n\n"
                "- 🧪 **ASD screening** — questionnaire or text description\n"
                "- 📖 **Social stories** — for difficult situations\n"
                "- 🗓️ **Visual schedules** — step-by-step routine guides\n\n"
                "Would you like to **start a screening**, or can I help with something ASD-related?"
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            return state

        if intent in ("QUESTIONNAIRE", "TEXT_ANALYSIS") and stage in ("choose_method", "collect_profile", "idle"):
            # If no profile yet, collect it first — unless user is explicitly skipping straight to screening
            if not profile.get("name") and not profile.get("age") and stage != "choose_method":
                # Store intent so we can route after profile
                if intent == "QUESTIONNAIRE":
                    state["stage"] = "awaiting_answers"
                else:
                    state["stage"] = "awaiting_description"
                state["answer"] = ""
                return state
            # Route directly to the appropriate specialist agent
            if intent == "QUESTIONNAIRE":
                state["stage"] = "awaiting_answers"
            else:
                state["stage"] = "awaiting_description"
            state["answer"] = ""
            return state

        if intent == "START_SCREENING" or (stage == "idle" and last_human):
            # Check if we have a profile; if not, ask for one first
            if not profile.get("name") and not profile.get("age"):
                reply = (
                    "I'd love to help! Before we begin, it helps to know a little "
                    "about your child so I can personalise the guidance for you.\n\n"
                    "**What is your child's name and age?** "
                    "*(You can also mention any known sensory triggers — e.g. loud noises, bright lights)*"
                )
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "collect_profile"
                return state
            else:
                name  = profile.get("name", "your child")
                reply = (
                    f"Let's begin the screening for **{name}**.\n\n"
                    "Which method would you prefer?\n\n"
                    "- **Questionnaire** — Answer 11 structured Q-CHAT-10 questions (0 or 1)\n"
                    "- **Text description** — Describe your child's behaviour in your own words\n\n"
                    "Type **questionnaire** or **text** to continue."
                )
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "choose_method"
                return state

        if stage == "collect_profile" and last_human:
            # If user skips profile and directly asks for screening method, honour it
            if intent in ("QUESTIONNAIRE", "TEXT_ANALYSIS"):
                if intent == "QUESTIONNAIRE":
                    state["stage"] = "awaiting_answers"
                else:
                    state["stage"] = "awaiting_description"
                state["answer"] = ""
                return state

            extracted = extract_profile(last_human)
            state["child_profile"] = extracted
            name  = extracted.get("name", "your child")
            age   = extracted.get("age", "")
            intro = f"Thank you! I'll keep **{name}**" + (f" ({age})" if age else "") + " in mind throughout our session.\n\n"
            reply = (
                intro
                + "Which method would you like to use for screening?\n\n"
                "- **Questionnaire** — Answer 11 structured Q-CHAT-10 questions (0 or 1)\n"
                "- **Text description** — Describe your child's behaviour in your own words\n\n"
                "You can also ask me to create a **social story** or a **visual schedule** at any time.\n\n"
                "Type **questionnaire** or **text** to begin."
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "choose_method"
            return state

        # ── Default: supervisor speaks (idle greeting or off-topic) ───────────
        if stage == "idle" and not last_human:
            # Opening greeting — no human message yet
            reply = (
                "👋 **Hello! Welcome to the ASD Screening & Companion Assistant.**\n\n"
                "I'm here to help parents and caregivers with:\n\n"
                "- 🧪 **ASD screening** using a structured questionnaire or free-text description\n"
                "- 📖 **Social stories** — personalised narratives for difficult situations\n"
                "- 🗓️ **Visual schedules** — step-by-step routine guides\n"
                "- 📊 **Progress tracking** — review past screening results\n\n"
                "⚠️ *This tool is for screening purposes only — not a clinical diagnosis.*\n\n"
                "**Would you like to begin a screening?** Or type what you need and I'll help."
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "idle"
            return state

        # Guidance-stage follow-up or unknown intent → guidance agent handles it
        if stage in ("guidance", "choose_method") or intent == "FOLLOW_UP":
            state["answer"] = ""  # let guidance agent respond
            return state

        # Fallback reply
        reply = (
            "I'm here to help with ASD screening, social stories, or visual schedules. "
            "Would you like to **start a screening**, get a **social story**, "
            "or create a **visual schedule**?"
        )
        state["messages"] = messages + [AIMessage(content=reply)]
        state["answer"]   = reply
        return state

    # ── Questionnaire agent ───────────────────────────────────────────────────

    def questionnaire_agent(state: State) -> State:
        messages   = state.get("messages", [])
        profile    = state.get("child_profile") or {}
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        child_name = profile.get("name", "your child")

        QUESTIONS = (
            f"### 📋 Q-CHAT-10 Questionnaire\n\n"
            f"Answer **0** (No / Female) or **1** (Yes / Male) for each question about **{child_name}**:\n\n"
            " 1.  Does your child look at you when you call his/her name?\n"
            " 2.  How easy is it to get eye contact with your child?\n"
            " 3.  Does your child point to indicate they want something?\n"
            " 4.  Does your child point to share interest with you?\n"
            " 5.  Does your child pretend? (e.g. care for dolls, toy phone)\n"
            " 6.  Does your child follow where you're looking?\n"
            " 7.  If someone is upset, does your child try to comfort them?\n"
            " 8.  Would you describe your child's first words as normal?\n"
            " 9.  Does your child use simple gestures? (e.g. wave goodbye)\n"
            "10.  Does your child stare at nothing with no apparent purpose?\n"
            "11.  Child biological sex (0=Female, 1=Male)\n\n"
            "Enter all 11 answers as comma-separated numbers: A1,A2,...,A10,Sex\n"
            "**Example:** `0,1,0,1,1,0,0,0,1,0,0`\n\n"
            "**Your answers:**"
        )

        if state.get("stage") == "awaiting_answers" and last_human and "," in last_human:
            result = parse_answers(last_human)
            if isinstance(result, str):
                reply = f"❌ {result}\n\n{QUESTIONS}"
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "awaiting_answers"
            else:
                result_md, label, conf, p_asd, p_non = questionnaire_predict(
                    xgboost_model, result
                )
                trait_note = (
                    "\n\n---\n"
                    "**ℹ️ Note:** Detailed behavioural trait analysis is available "
                    "when using the **text description method**. The questionnaire "
                    "provides a validated screening score.\n\n"
                    "You can ask me what the result means, what to do next, "
                    "request a **social story** or **visual schedule**, "
                    "type **another** for a new assessment, or **exit** to end."
                )
                answer = result_md + trait_note

                assessment = {
                    "date"          : datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "method"        : "questionnaire",
                    "label"         : label,
                    "confidence"    : float(conf),
                    "proba_asd"     : float(p_asd),
                    "proba_non"     : float(p_non),
                    "traits_present": [],
                    "concern_level" : "",
                    "features"      : {k: int(v) for k, v in zip(FEATURE_COLS, result)},
                }
                state["last_assessment"] = assessment
                history = list(state.get("session_history", []))
                history.append(assessment)
                state["session_history"] = history

                state["messages"] = messages + [AIMessage(content=answer)]
                state["answer"]   = answer
                state["stage"]    = "guidance"
            return state

        # First visit — show questions
        state["messages"] = messages + [AIMessage(content=QUESTIONS)]
        state["answer"]   = QUESTIONS
        state["stage"]    = "awaiting_answers"
        return state

    # ── Text agent ────────────────────────────────────────────────────────────

    def text_agent(state: State) -> State:
        messages   = state.get("messages", [])
        profile    = state.get("child_profile") or {}
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        child_name = profile.get("name", "your child")

        PROMPT = (
            f"### 📝 Text Description Method\n\n"
            f"Describe **{child_name}'s** behaviour in your own words.\n"
            "Include details about:\n"
            "- Social interactions\n"
            "- Communication patterns\n"
            "- Repetitive behaviours\n"
            "- Response to surroundings\n\n"
            "**Example:** *'My 3-year-old rarely makes eye contact and does not "
            "respond to his name. He lines up his toys and gets upset if moved.'*\n\n"
            "**Your description:**"
        )

        if state.get("stage") == "awaiting_description" and last_human and len(last_human.split()) >= 3:
            if not is_valid_description(last_human):
                reply = (
                    "❌ That doesn't look like a behavioural description.\n\n"
                    "Please describe the **child's behaviour**, communication, or social traits.\n\n"
                    "**Example:** *'My 3-year-old rarely makes eye contact.'*\n\n"
                    "**Your description:**"
                )
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "awaiting_description"
                return state

            # BERT inference
            result_md, label, conf, p_asd, p_non = text_predict(
                bert_tokenizer, bert_model_obj, last_human
            )

            # Trait extraction (constrained to fixed taxonomy)
            trait_result = extract_traits(last_human)

            # Severity scoring
            concern      = compute_concern_level(trait_result["traits_present"])
            trait_report = format_trait_report(trait_result, concern)

            # Trigger detection from retrieval engine
            triggers_found = retrieval_engine.detect_triggers(last_human)
            trigger_note   = ""
            if triggers_found:
                trigger_note = (
                    f"\n\n> 🔔 **Sensory triggers detected in description:** "
                    f"{', '.join(triggers_found)}. "
                    f"Consider requesting a **social story** that includes coping strategies for these."
                )

            answer = result_md + trait_report + trigger_note + (
                "\n\n---\n"
                "You can now ask what any trait means, what to do next, request a "
                "**social story** or **visual schedule**, type **another** for a "
                "new assessment, or **exit** to end."
            )

            assessment = {
                "date"             : datetime.now().strftime("%Y-%m-%d %H:%M"),
                "method"           : "text",
                "label"            : label,
                "confidence"       : float(conf),
                "proba_asd"        : float(p_asd),
                "proba_non"        : float(p_non),
                "description"      : last_human,
                "traits_present"   : trait_result["traits_present"],
                "traits_absent"    : trait_result["traits_absent"],
                "traits_uncertain" : trait_result["traits_uncertain"],
                "concern_level"    : concern["level"],
                "concern_rationale": concern["rationale"],
                "high_traits"      : concern["high"],
                "moderate_traits"  : concern["moderate"],
                "monitor_traits"   : concern["monitor"],
            }
            state["last_assessment"] = assessment
            history = list(state.get("session_history", []))
            history.append(assessment)
            state["session_history"] = history

            state["messages"] = messages + [AIMessage(content=answer)]
            state["answer"]   = answer
            state["stage"]    = "guidance"
            return state

        state["messages"] = messages + [AIMessage(content=PROMPT)]
        state["answer"]   = PROMPT
        state["stage"]    = "awaiting_description"
        return state

    # ── Story agent ───────────────────────────────────────────────────────────

    def story_agent(state: State) -> State:
        messages   = state.get("messages", [])
        profile    = state.get("child_profile") or {}
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        if state.get("stage") == "story_request" and last_human and len(last_human.split()) >= 2:
            # Retrieve similar existing stories as context
            similar = retrieval_engine.search(last_human, k=2, doc_type="social_story")
            context_block = ""
            if similar:
                context_block = "\n\nFor reference, here are similar stories:\n"
                for s in similar:
                    context_block += f"\n**{s['title']}:** {s['content'][:200]}...\n"

            child_name = profile.get("name", "the child")
            child_age  = profile.get("age", "")
            triggers   = profile.get("triggers", [])

            profile_note = f"Child's name: {child_name}"
            if child_age:
                profile_note += f", Age: {child_age}"
            if triggers:
                profile_note += f", Known triggers: {', '.join(triggers)}"

            system = STORY_SYSTEM.format(title=last_human)
            prompt = (
                f"Situation: {last_human}\n"
                f"Profile: {profile_note}\n"
                f"{context_block}\n\n"
                f"Now write a personalised social story for this child and situation."
            )

            resp  = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            story = resp.content.strip()

            # Detect triggers in generated story and add advisory
            triggers_in_story = retrieval_engine.detect_triggers(story)
            if triggers_in_story:
                known = profile.get("triggers", [])
                overlap = [t for t in triggers_in_story if any(k.lower() in t for k in known)]
                if overlap:
                    story += (
                        f"\n\n> ⚠️ **Sensory note:** This story mentions "
                        f"{', '.join(overlap)}, which are known triggers for {child_name}. "
                        f"Consider reviewing the story together before the situation occurs."
                    )

            reply = (
                story + "\n\n---\n"
                "You can ask me to modify this story, create a **visual schedule**, "
                "or go back to screening with **another**."
            )
            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "guidance"
            return state

        # Shouldn't reach here (supervisor sets up the prompt)
        state["answer"] = ""
        return state

    # ── Schedule agent ────────────────────────────────────────────────────────

    def schedule_agent(state: State) -> State:
        messages   = state.get("messages", [])
        profile    = state.get("child_profile") or {}
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        if state.get("stage") == "schedule_request" and last_human and len(last_human.split()) >= 1:
            # Retrieve similar existing schedules
            similar = retrieval_engine.search(last_human, k=2, doc_type="visual_schedule")

            child_name = profile.get("name", "the child")
            child_age  = profile.get("age", "")

            # Always personalise with LLM; use close template match as a base
            system = SCHEDULE_SYSTEM.format(title=last_human)
            profile_note = f"Child: {child_name}" + (f", Age: {child_age}" if child_age else "")
            triggers = profile.get("triggers", [])
            if triggers:
                profile_note += f", Sensory considerations: {', '.join(triggers)}"

            template_note = ""
            if similar and similar[0].get("score", 0) > 0.4:
                match = similar[0]
                base_steps = match["content"].replace(" → ", "\n")
                template_note = f"\n\nBase template steps to personalise:\n{base_steps}"

            prompt = (
                f"Routine: {last_human}\n"
                f"Profile: {profile_note}"
                f"{template_note}\n\n"
                f"Write a warm, personalised visual schedule. Use the child's name naturally. "
                f"Include short sensory tips in brackets if triggers are listed."
            )
            resp     = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            schedule = resp.content.strip()
            reply    = (
                schedule + "\n\n---\n"
                "You can ask me to adjust any step, request a **social story**, "
                "or type **another** for a new screening."
            )

            state["messages"] = messages + [AIMessage(content=reply)]
            state["answer"]   = reply
            state["stage"]    = "guidance"
            return state

        state["answer"] = ""
        return state

    # ── Guidance agent ────────────────────────────────────────────────────────

    def guidance_agent(state: State) -> State:
        messages        = state.get("messages", [])
        last_assessment = state.get("last_assessment") or {}
        profile         = state.get("child_profile") or {}

        # Check intent to handle RESTART/EXIT reliably without keyword matching
        last_human = next(
            (m.content.strip() for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human:
            intent = classify_intent(last_human, "guidance")
            if intent == "RESTART":
                state["stage"]  = "choose_method"
                state["answer"] = ""
                return state
            if intent == "EXIT":
                name  = profile.get("name", "")
                reply = (
                    f"Thank you for using the ASD Screening Assistant."
                    + (f" Give {name} a big hug from me! 💙" if name else " 💙")
                    + " Always consult a qualified professional for any concerns. Goodbye!"
                )
                state["messages"] = messages + [AIMessage(content=reply)]
                state["answer"]   = reply
                state["stage"]    = "idle"
                return state
            if intent == "SOCIAL_STORY":
                state["stage"]  = "story_request"
                state["answer"] = ""
                return state
            if intent == "VISUAL_SCHEDULE":
                state["stage"]  = "schedule_request"
                state["answer"] = ""
                return state

        # Build assessment context string
        ctx = []
        if last_assessment:
            ctx.append(f"Method:          {last_assessment.get('method','unknown')}")
            ctx.append(f"Result:          {last_assessment.get('label','unknown')}")
            ctx.append(f"Confidence:      {last_assessment.get('confidence',0):.1f}%")
            ctx.append(f"Concern level:   {last_assessment.get('concern_level','N/A')}")
            traits = last_assessment.get("traits_present", [])
            if traits:
                ctx.append(f"Traits found:    {', '.join(traits)}")
                ctx.append(f"High concern:    {', '.join(last_assessment.get('high_traits', [])) or 'None'}")
                ctx.append(f"Moderate:        {', '.join(last_assessment.get('moderate_traits', [])) or 'None'}")
                ctx.append(f"Monitor:         {', '.join(last_assessment.get('monitor_traits', [])) or 'None'}")
            else:
                ctx.append("Traits found:    None specifically identified")
            desc = last_assessment.get("description", "")
            if desc:
                ctx.append(f"Description:     {desc[:300]}")
        else:
            ctx.append("No assessment completed yet.")

        sys_prompt = GUIDANCE_SYSTEM.format(
            context="\n".join(ctx),
            profile=profile_context(profile),
        )

        history = [SystemMessage(content=sys_prompt)]
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                history.append(msg)

        response = llm.invoke(history)
        reply    = response.content.strip()

        state["messages"] = messages + [AIMessage(content=reply)]
        state["answer"]   = reply
        state["stage"]    = "guidance"
        return state

    return (supervisor_agent, questionnaire_agent, text_agent,
            story_agent, schedule_agent, guidance_agent)


# ══════════════════════════════════════════════════════════════════════════════
# Routing logic — maps state to next node
# ══════════════════════════════════════════════════════════════════════════════

def routing_logic(state: State) -> Literal[
    "questionnaire_agent","text_agent","guidance_agent",
    "story_agent","schedule_agent","end"
]:
    stage = state.get("stage", "idle")

    if stage == "awaiting_answers":
        return "questionnaire_agent"
    if stage == "awaiting_description":
        return "text_agent"
    if stage == "story_request":
        return "story_agent"
    if stage == "schedule_request":
        return "schedule_agent"
    if stage in ("guidance", "choose_method"):
        # If supervisor set answer to "" it means guidance should respond
        if state.get("answer", "") == "":
            return "guidance_agent"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(xgboost_model, bert_tokenizer, bert_model_obj):
    """
    Build and compile the LangGraph agent with MemorySaver.
    Returns (compiled_agent, MemorySaver_instance).
    """
    # Load retrieval engine lazily
    retrieval_engine.load()

    supervisor, questionnaire, text, story, schedule, guidance = build_supervisor_fns(
        xgboost_model, bert_tokenizer, bert_model_obj
    )

    workflow = StateGraph(State)

    workflow.add_node("supervisor_agent",    supervisor)
    workflow.add_node("questionnaire_agent", questionnaire)
    workflow.add_node("text_agent",          text)
    workflow.add_node("story_agent",         story)
    workflow.add_node("schedule_agent",      schedule)
    workflow.add_node("guidance_agent",      guidance)

    workflow.add_edge(START, "supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        routing_logic,
        {
            "questionnaire_agent": "questionnaire_agent",
            "text_agent"         : "text_agent",
            "story_agent"        : "story_agent",
            "schedule_agent"     : "schedule_agent",
            "guidance_agent"     : "guidance_agent",
            "end"                : END,
        },
    )

    workflow.add_edge("questionnaire_agent", END)
    workflow.add_edge("text_agent",          END)
    workflow.add_edge("story_agent",         END)
    workflow.add_edge("schedule_agent",      END)
    workflow.add_edge("guidance_agent",      END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory), memory
