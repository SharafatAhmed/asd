"""
----------------------------------------------------------------
A professional caregiver-facing interface with:
  • Warm onboarding — child profile collection before screening
  • ASD screening via Q-CHAT-10 questionnaire (XGBoost) or free-text (BERT)
  • Social story generation for difficult real-life situations
  • Visual schedule generation for daily routines
  • Session history — review all assessments in the current session
  • Progress chart — confidence trend across assessments
  • Concern level badges with colour coding
  • MemorySaver thread per session — true multi-turn memory
"""

import uuid
import pickle
import json
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.messages import HumanMessage

from config import XGBOOST_MODEL_PATH, BERT_MODEL_PATH
from agent import build_agent


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Trait Detection Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge-high     { background:#FF4B4B; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-moderate { background:#FFA500; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-monitor  { background:#1C83E1; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-low      { background:#21C354; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.profile-card   { background:#f0f4ff; border-radius:8px; padding:12px; margin-bottom:8px; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading (cached — once per Streamlit process)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_xgboost_model():
    try:
        with open(XGBOOST_MODEL_PATH, "rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, f"XGBoost model not found at: {XGBOOST_MODEL_PATH}"
    except Exception as e:
        return None, f"Error loading XGBoost: {e}"


@st.cache_resource
def load_bert_model():
    try:
        tok   = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        model.eval()
        return tok, model, None
    except Exception as e:
        return None, None, f"Error loading BERT: {e}"


@st.cache_resource
def load_agent(_xgb, _tok, _bert):
    return build_agent(_xgb, _tok, _bert)


xgboost_model,              xgb_error  = load_xgboost_model()
bert_tokenizer, bert_model, bert_error = load_bert_model()
models_ready = (xgboost_model is not None) and (bert_model is not None)

if models_ready:
    agents, memory = load_agent(xgboost_model, bert_tokenizer, bert_model)
else:
    agents, memory = None, None


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def init_session(trigger_greeting: bool = True):
    st.session_state.messages        = []
    st.session_state.thread_id       = str(uuid.uuid4())
    st.session_state.greeted         = False
    st.session_state.child_profile   = {}
    st.session_state.session_history = []

    if trigger_greeting and agents:
        config  = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial = {
            "messages"       : [],
            "answer"         : "",
            "stage"          : "idle",
            "last_assessment": None,
            "child_profile"  : None,
            "session_history": [],
        }
        result = agents.invoke(initial, config=config)
        greeting = result.get("answer", "")
        if greeting:
            st.session_state.messages.append(("assistant", greeting))
        st.session_state.greeted = True

        # Sync profile and history from agent state
        st.session_state.child_profile   = result.get("child_profile") or {}
        st.session_state.session_history = result.get("session_history") or []


if "thread_id" not in st.session_state:
    init_session(trigger_greeting=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core: send a message through the LangGraph agent
# ══════════════════════════════════════════════════════════════════════════════

def send_message(user_text: str):
    if not agents:
        st.session_state.messages.append(
            ("assistant", "❌ Models not loaded. Cannot process request.")
        )
        return

    st.session_state.messages.append(("user", user_text))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = agents.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )

    reply = result.get("answer", "")
    if reply:
        st.session_state.messages.append(("assistant", reply))

    # Sync profile and history so sidebar updates live
    profile = result.get("child_profile")
    if profile:
        st.session_state.child_profile = profile

    history = result.get("session_history")
    if history:
        st.session_state.session_history = history


# ══════════════════════════════════════════════════════════════════════════════
# Progress chart helper
# ══════════════════════════════════════════════════════════════════════════════

def render_progress_chart(history: list):
    """Render a confidence trend chart from session history."""
    if len(history) < 2:
        return None

    labels  = [f"#{i+1} {h.get('method','').capitalize()}" for i, h in enumerate(history)]
    confs   = [h.get("confidence", 0) for h in history]
    colours = ["#FF4B4B" if h.get("label") == "ASD" else "#21C354" for h in history]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.bar(labels, confs, color=colours, alpha=0.85, width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Confidence %", fontsize=9)
    ax.set_title("Screening Confidence History", fontsize=10, fontweight="bold")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    for bar, val in zip(bars, confs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Concern level badge helper
# ══════════════════════════════════════════════════════════════════════════════

def concern_badge(level: str) -> str:
    cls_map = {
        "High": "badge-high", "Moderate": "badge-moderate",
        "Monitor": "badge-monitor", "Low": "badge-low",
    }
    cls = cls_map.get(level, "badge-low")
    return f'<span class="{cls}">{level}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# UI — main layout
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧠 ASD Screening Agent")
        st.markdown("---")

        # Model status
        st.subheader("Model Status")
        c1, c2 = st.columns(2)
        with c1:
            if xgboost_model:
                st.success("✅ XGBoost")
            else:
                st.error("❌ XGBoost")
        with c2:
            if bert_model:
                st.success("✅ BERT")
            else:
                st.error("❌ BERT")

        st.markdown("---")

        # Child profile card (live, updates as agent collects info)
        profile = st.session_state.get("child_profile", {})
        if profile.get("name") or profile.get("age"):
            st.subheader("👶 Child Profile")
            card_html = '<div class="profile-card">'
            if profile.get("name"):
                card_html += f"<b>Name:</b> {profile['name']}<br>"
            if profile.get("age"):
                card_html += f"<b>Age:</b> {profile['age']}<br>"
            if profile.get("triggers"):
                card_html += f"<b>Triggers:</b> {', '.join(profile['triggers'])}<br>"
            if profile.get("skills_working"):
                card_html += f"<b>Goals:</b> {', '.join(profile['skills_working'])}"
            card_html += "</div>"
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📋 Questionnaire", use_container_width=True):
                send_message("questionnaire")
                st.rerun()
        with c2:
            if st.button("📝 Text Analysis", use_container_width=True):
                send_message("text")
                st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("📖 Social Story", use_container_width=True):
                send_message("social story")
                st.rerun()
        with c4:
            if st.button("🗓️ Schedule", use_container_width=True):
                send_message("visual schedule")
                st.rerun()

        if st.button("📊 Show History", use_container_width=True):
            send_message("show my history")
            st.rerun()

        st.markdown("---")

        # Session history summary
        history = st.session_state.get("session_history", [])
        if history:
            st.subheader("📋 Session Summary")
            for i, h in enumerate(history, 1):
                label = h.get("label", "")
                conf  = h.get("confidence", 0)
                cl    = h.get("concern_level", "")
                badge = concern_badge(cl) if cl else ""
                st.markdown(
                    f"**#{i}** {h.get('method','').capitalize()} → "
                    f"**{label}** ({conf:.0f}%) {badge}",
                    unsafe_allow_html=True,
                )

            # Progress chart if 2+ assessments
            fig = render_progress_chart(history)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")

        # New conversation
        if st.button("🔄 New Conversation", type="secondary", use_container_width=True):
            init_session(trigger_greeting=True)
            st.rerun()

        st.markdown("---")

        # How to use
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Screening methods**
            - **Questionnaire** — 11 Q-CHAT-10 questions (enter 0 or 1 for each)
            - **Text description** — Describe behaviour freely, get BERT-based analysis

            **Companion features**
            - **Social story** — Describe a situation your child finds difficult
            - **Visual schedule** — Name a routine (morning, bedtime, school day...)
            - **History** — Review all screenings in this session

            **After a result**
            - Ask about any trait, next steps, or what to watch at home
            - Type **another** to run a new assessment
            - Type **exit** to end the session
            """)

        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**

        This is a screening tool only — not a clinical diagnostic instrument.
        Always consult a qualified healthcare professional for diagnosis and treatment.
        """)

        if "thread_id" in st.session_state:
            st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("Multimodal AI Agent Framework for Autism Trait Detection in Children")


    # Chat history
    chat_container = st.container(height=520)
    with chat_container:
        if not st.session_state.messages:
            st.info("Loading agent greeting...")
        for role, message in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(message, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        with st.spinner("Thinking..."):
            send_message(prompt)
        st.rerun()

    # Footer
    st.markdown("---")
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.caption(

            "For screening purposes only — not a clinical diagnostic tool"
        )


if __name__ == "__main__":
    main()
