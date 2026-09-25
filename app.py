
import os, re, json, time, uuid, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = Any

try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None


# ============================================================
# PromptPilot Enterprise — Streamlit-native reference app
# Layers:
# UI -> Application -> Orchestration -> Intelligence -> Providers
# -> Persistence/Telemetry
# ============================================================

st.set_page_config(
    page_title="PromptPilot Enterprise",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Design System -----------------------------

CSS = """
<style>
:root {
  --pp-blue: #1457ff;
  --pp-blue-2: #0b3dcc;
  --pp-blue-50: #eff5ff;
  --pp-blue-100: #dce8ff;
  --pp-green: #0b8f63;
  --pp-green-50: #ecfbf5;
  --pp-ink: #10213f;
  --pp-muted: #63728b;
  --pp-line: #dce4f0;
  --pp-bg: #f5f8fc;
  --pp-white: #ffffff;
  --pp-danger: #c83232;
  --pp-radius: 18px;
}
.stApp { background: var(--pp-bg); color: var(--pp-ink); }
.block-container { max-width: 1480px; padding-top: 1.25rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--pp-line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
h1,h2,h3,h4 { color: var(--pp-ink) !important; letter-spacing: -0.02em; }
p, label, .stCaption { color: var(--pp-muted); }
.stButton > button {
  border-radius: 11px !important;
  border: 1px solid var(--pp-line) !important;
  min-height: 42px;
  font-weight: 700;
}
.stButton > button[kind="primary"] {
  background: var(--pp-blue) !important;
  border-color: var(--pp-blue) !important;
  color: white !important;
}
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
  border-radius: 12px !important;
  border-color: var(--pp-line) !important;
}
.pp-topbar {
  background: white; border: 1px solid var(--pp-line); border-radius: 16px;
  padding: 13px 18px; display:flex; justify-content:space-between; align-items:center;
  margin-bottom: 18px; box-shadow: 0 5px 18px rgba(30, 60, 100, .04);
}
.pp-brand { font-size: 18px; font-weight: 850; color: var(--pp-ink); }
.pp-brand span { color: var(--pp-blue); }
.pp-status { display:flex; gap:8px; align-items:center; font-size:12px; font-weight:700; color:var(--pp-green); }
.pp-dot { width:8px; height:8px; background:var(--pp-green); border-radius:50%; display:inline-block; }
.pp-hero {
  background: linear-gradient(135deg, #ffffff 0%, #f2f7ff 100%);
  border: 1px solid var(--pp-blue-100); border-radius: 24px;
  padding: 28px; margin-bottom: 18px;
}
.pp-eyebrow { color:var(--pp-blue); text-transform:uppercase; letter-spacing:.12em; font-size:11px; font-weight:850; }
.pp-hero-title { font-size: 35px; font-weight: 900; line-height: 1.05; margin: 7px 0 8px; }
.pp-hero-sub { color: var(--pp-muted); font-size: 15px; max-width: 800px; }
.pp-card {
  background: white; border:1px solid var(--pp-line); border-radius:var(--pp-radius);
  padding: 18px; box-shadow: 0 7px 24px rgba(30, 60, 100, .035); margin-bottom: 14px;
}
.pp-card-title { font-size:15px; font-weight:850; color:var(--pp-ink); margin-bottom:5px; }
.pp-card-sub { font-size:12px; color:var(--pp-muted); margin-bottom:13px; }
.pp-metric {
  background:#fff; border:1px solid var(--pp-line); border-radius:15px; padding:15px;
}
.pp-metric-label { color:var(--pp-muted); font-size:11px; font-weight:750; }
.pp-metric-value { color:var(--pp-ink); font-size:24px; font-weight:900; margin-top:4px; }
.pp-tag {
  display:inline-block; background:var(--pp-blue-50); color:var(--pp-blue-2);
  border:1px solid var(--pp-blue-100); border-radius:999px; padding:4px 8px;
  font-size:11px; font-weight:750; margin:2px;
}
.pp-output {
  background:#0d1b35; color:#eaf1ff; border-radius:15px; padding:17px;
  border:1px solid #20365f; white-space:pre-wrap; line-height:1.55; font-size:13px;
}
.pp-side-note { font-size:11px; color:var(--pp-muted); line-height:1.45; }
div[data-testid="stExpander"] { border:1px solid var(--pp-line); border-radius:14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------- Configuration -----------------------------

def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

GROQ_API_KEY = secret("GROQ_API_KEY")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_KEY = secret("SUPABASE_KEY")
ELEVENLABS_API_KEY = secret("ELEVENLABS_API_KEY")
FAST_MODEL = secret("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
QUALITY_MODEL = secret("GROQ_QUALITY_MODEL", "openai/gpt-oss-120b")
ELEVEN_VOICE_ID = secret("ELEVENLABS_VOICE_ID", "")

# ----------------------------- Session State -----------------------------

defaults = {
    "page": "Prompt Studio",
    "prompt": "",
    "result": "",
    "plan": None,
    "analysis": None,
    "history": [],
    "last_run_ms": 0,
    "request_id": "",
    "error": "",
    "role": "AI Assistant",
    "mode": "Fast",
    "voice": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- Domain Models -----------------------------

@dataclass
class PromptSignals:
    word_count: int
    is_question: bool
    code: bool
    research: bool
    creative: bool
    business: bool
    data: bool
    technical: bool

@dataclass
class PromptAnalysis:
    intent: str
    complexity: float
    completeness: int
    confidence: int
    signals: Dict[str, Any]
    missing: List[str]
    route: str

@dataclass
class PromptPlan:
    title: str
    objective: str
    context: str
    instructions: List[str]
    constraints: List[str]
    output_format: str
    assumptions: List[str]
    optimized_prompt: str


# ----------------------------- Provider Layer -----------------------------

@st.cache_resource(show_spinner=False)
def get_groq():
    if not GROQ_API_KEY or Groq is None:
        return None
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource(show_spinner=False)
def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY or create_client is None:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_elevenlabs():
    if not ELEVENLABS_API_KEY or ElevenLabs is None:
        return None
    try:
        return ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception:
        return None


# ----------------------------- Intelligence Layer -----------------------------

STOPWORDS = {
    "the","a","an","is","are","to","of","and","or","for","in","on","with",
    "this","that","it","i","you","me","my","we","can","please","make","give"
}

def sanitize_prompt(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:12000]

def extract_signals(text: str) -> PromptSignals:
    t = text.lower()
    return PromptSignals(
        word_count=len(text.split()),
        is_question="?" in text,
        code=bool(re.search(r"\b(code|python|javascript|java|sql|api|function|debug|bug|algorithm)\b", t)),
        research=bool(re.search(r"\b(research|paper|experiment|hypothesis|literature|citation|study)\b", t)),
        creative=bool(re.search(r"\b(story|poem|design|creative|caption|image|brand|logo)\b", t)),
        business=bool(re.search(r"\b(strategy|business|market|sales|customer|product|enterprise|roi|kpi)\b", t)),
        data=bool(re.search(r"\b(data|dataset|analytics|statistics|model|regression|forecast)\b", t)),
        technical=bool(re.search(r"\b(architecture|system|database|backend|frontend|cloud|docker|kubernetes|security)\b", t)),
    )

def completeness_score(text: str) -> tuple[int, List[str]]:
    t = text.lower()
    score = 15
    missing = []
    checks = [
        ("clear task", bool(re.search(r"\b(create|build|write|explain|analyze|design|fix|compare|generate|develop|optimize)\b", t)), 25),
        ("context", len(text.split()) >= 12, 20),
        ("constraints", bool(re.search(r"\b(must|should|avoid|under|within|only|using|without)\b", t)), 15),
        ("audience", bool(re.search(r"\b(for|audience|user|customer|student|developer|manager|team)\b", t)), 10),
        ("output format", bool(re.search(r"\b(table|json|steps|bullet|report|code|email|list|markdown|format)\b", t)), 15),
    ]
    for name, ok, points in checks:
        if ok:
            score += points
        else:
            missing.append(name)
    return min(100, score), missing

def estimate_complexity(signals: PromptSignals) -> float:
    score = 0.15
    score += min(signals.word_count / 500, 0.25)
    score += 0.12 if signals.code else 0
    score += 0.12 if signals.research else 0
    score += 0.10 if signals.business else 0
    score += 0.10 if signals.data else 0
    score += 0.10 if signals.technical else 0
    return round(min(score, 1.0), 3)

def infer_intent(s: PromptSignals) -> str:
    if s.code or s.technical:
        return "Technical / Engineering"
    if s.research:
        return "Research / Analysis"
    if s.business:
        return "Business / Product"
    if s.creative:
        return "Creative / Content"
    if s.data:
        return "Data / Analytics"
    return "General Assistance"

def analyze_prompt(text: str, mode: str) -> PromptAnalysis:
    s = extract_signals(text)
    comp, missing = completeness_score(text)
    complexity = estimate_complexity(s)
    route = QUALITY_MODEL if mode == "Quality" or complexity >= 0.58 else FAST_MODEL
    confidence = max(50, min(98, 55 + int(comp * 0.35) + int(complexity * 12)))
    return PromptAnalysis(
        intent=infer_intent(s),
        complexity=complexity,
        completeness=comp,
        confidence=confidence,
        signals=asdict(s),
        missing=missing,
        route=route,
    )

def deterministic_plan(text: str, analysis: PromptAnalysis, role: str) -> PromptPlan:
    objective = text.strip()
    constraints = [
        "Preserve factual accuracy and state assumptions when information is missing.",
        "Use a clear, structured response appropriate for the requested task.",
    ]
    if analysis.signals.get("code"):
        constraints.append("Prefer production-quality code with validation and error handling.")
    if analysis.signals.get("research"):
        constraints.append("Separate established evidence from hypotheses or assumptions.")
    output = "Provide a concise, structured answer with actionable details."
    optimized = (
        f"Role: {role}\n\n"
        f"Objective:\n{objective}\n\n"
        "Execution requirements:\n"
        "- Understand the user's actual goal before responding.\n"
        "- Use the supplied context and avoid inventing missing facts.\n"
        "- Make the answer structured, actionable, and easy to verify.\n\n"
        f"Constraints:\n" + "\n".join(f"- {x}" for x in constraints) +
        f"\n\nOutput format:\n{output}"
    )
    return PromptPlan(
        title="Optimized Prompt",
        objective=objective,
        context="User-provided request with deterministic signal analysis.",
        instructions=[
            "Understand the goal and identify the required deliverable.",
            "Use relevant context and explicit constraints.",
            "Return the requested output in a structured format.",
        ],
        constraints=constraints,
        output_format=output,
        assumptions=[],
        optimized_prompt=optimized,
    )


# ----------------------------- Orchestration Layer -----------------------------

PLANNER_SYSTEM = """You are PromptPilot's prompt-planning engine.
Transform a user's rough request into a precise, reusable instruction.
Return ONLY valid JSON matching the supplied schema.
Do not invent facts. Preserve the user's intent.
"""

def call_planner(text: str, analysis: PromptAnalysis, role: str) -> PromptPlan:
    client = get_groq()
    if client is None:
        return deterministic_plan(text, analysis, role)

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "objective": {"type": "string"},
            "context": {"type": "string"},
            "instructions": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "output_format": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "optimized_prompt": {"type": "string"},
        },
        "required": [
            "title","objective","context","instructions","constraints",
            "output_format","assumptions","optimized_prompt"
        ],
        "additionalProperties": False,
    }

    try:
        r = client.chat.completions.create(
            model=analysis.route,
            temperature=0.2,
            max_tokens=1600,
            messages=[
                {"role":"system","content":PLANNER_SYSTEM},
                {"role":"user","content":json.dumps({
                    "role": role,
                    "request": text,
                    "analysis": asdict(analysis),
                })},
            ],
            response_format={"type":"json_schema","json_schema":{"name":"prompt_plan","schema":schema}},
        )
        raw = r.choices[0].message.content
        data = json.loads(raw)
        return PromptPlan(**data)
    except Exception:
        return deterministic_plan(text, analysis, role)

def execute_prompt(plan: PromptPlan, analysis: PromptAnalysis, role: str) -> str:
    client = get_groq()
    if client is None:
        return (
            "Provider is not configured. The deterministic PromptPilot pipeline "
            "completed successfully.\n\nOptimized prompt:\n" + plan.optimized_prompt
        )

    system = f"""You are PromptPilot's execution engine.
Role: {role}
Answer the optimized prompt directly.
Be accurate, structured, useful, and do not claim actions you did not perform.
"""
    try:
        r = client.chat.completions.create(
            model=analysis.route,
            temperature=0.35,
            max_tokens=3000,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":plan.optimized_prompt},
            ],
        )
        return r.choices[0].message.content or "No output returned."
    except Exception as e:
        return "Execution fallback: " + plan.optimized_prompt

def synthesize_voice(text: str) -> Optional[bytes]:
    client = get_elevenlabs()
    if client is None or not ELEVEN_VOICE_ID:
        return None
    try:
        audio = client.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=text[:5000],
        )
        return b"".join(audio)
    except Exception:
        return None


# ----------------------------- Persistence / Observability -----------------------------

def log_run(payload: Dict[str, Any]) -> None:
    client = get_supabase()
    if client is None:
        return
    try:
        client.table("promptpilot_runs").insert(payload).execute()
    except Exception:
        pass

def run_pipeline(user_prompt: str, role: str, mode: str, execute: bool, voice: bool):
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    text = sanitize_prompt(user_prompt)
    if not text:
        raise ValueError("Please enter a prompt first.")

    analysis = analyze_prompt(text, mode)
    plan = call_planner(text, analysis, role)
    result = execute_prompt(plan, analysis, role) if execute else ""

    elapsed = int((time.perf_counter() - started) * 1000)
    audio = synthesize_voice(result) if voice and result else None

    payload = {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "role": role,
        "intent": analysis.intent,
        "complexity": analysis.complexity,
        "completeness": analysis.completeness,
        "latency_ms": elapsed,
        "input_hash": hashlib.sha256(text.encode()).hexdigest(),
    }
    log_run(payload)
    return request_id, analysis, plan, result, audio, elapsed


# ----------------------------- UI Layer -----------------------------

with st.sidebar:
    st.markdown("## ✦ PromptPilot")
    st.caption("Enterprise prompt intelligence")
    st.divider()

    pages = ["Prompt Studio", "History", "Architecture", "Settings"]
    st.session_state.page = st.radio(
        "Workspace", pages,
        index=pages.index(st.session_state.page)
    )

    st.divider()
    st.markdown("**Runtime**")
    provider = "Groq connected" if get_groq() else "Deterministic fallback"
    st.write("● " + provider)
    st.caption("Supabase: " + ("Connected" if get_supabase() else "Optional"))
    st.caption("Voice: " + ("Enabled" if get_elevenlabs() else "Optional"))

    st.divider()
    st.markdown('<div class="pp-side-note">Production note: configure secrets in Streamlit Cloud. Never place API keys in source code.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pp-topbar">
  <div class="pp-brand">Prompt<span>Pilot</span> <small>Enterprise</small></div>
  <div class="pp-status"><span class="pp-dot"></span> AI workspace ready</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.page == "Prompt Studio":
    st.markdown("""
    <div class="pp-hero">
      <div class="pp-eyebrow">Prompt Intelligence Platform</div>
      <div class="pp-hero-title">Turn rough ideas into production-ready AI instructions.</div>
      <div class="pp-hero-sub">Analyze intent, estimate complexity, route to the right model, structure the prompt, execute it, and capture measurable run telemetry.</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, label, value in [
        (c1,"Pipeline","6 layers"),
        (c2,"Routing","Adaptive"),
        (c3,"Output","Structured"),
        (c4,"Telemetry","Ready"),
    ]:
        with col:
            st.markdown(f'<div class="pp-metric"><div class="pp-metric-label">{label}</div><div class="pp-metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.25, .75], gap="large")

    with left:
        st.markdown('<div class="pp-card-title">Prompt Studio</div><div class="pp-card-sub">Describe what you want in natural language. PromptPilot handles the structure.</div>', unsafe_allow_html=True)

        role = st.selectbox(
            "AI role",
            ["AI Assistant","Senior Software Engineer","Data Scientist","Research Analyst","Product Strategist","Technical Writer","Creative Director"],
            index=["AI Assistant","Senior Software Engineer","Data Scientist","Research Analyst","Product Strategist","Technical Writer","Creative Director"].index(st.session_state.role),
        )
        st.session_state.role = role

        prompt = st.text_area(
            "Your request",
            value=st.session_state.prompt,
            height=210,
            placeholder="Example: Build a production-ready PostgreSQL schema for a university management system with RBAC, audit logging and analytics.",
            label_visibility="visible",
        )
        st.session_state.prompt = prompt

        a,b,c = st.columns([1,1,1])
        with a:
            mode = st.radio("Intelligence", ["Fast","Quality"], horizontal=True, index=0 if st.session_state.mode=="Fast" else 1)
            st.session_state.mode = mode
        with b:
            voice = st.toggle("Voice output", value=st.session_state.voice)
            st.session_state.voice = voice
        with c:
            st.caption("Fast = lower latency\nQuality = stronger reasoning route")

        b1,b2,b3 = st.columns([1.2,1,1])
        with b1:
            preview = st.button("Analyze & Optimize", type="primary", use_container_width=True)
        with b2:
            run = st.button("Optimize & Run", use_container_width=True)
        with b3:
            clear = st.button("Clear", use_container_width=True)

        if clear:
            st.session_state.prompt = ""
            st.session_state.result = ""
            st.session_state.plan = None
            st.session_state.analysis = None
            st.rerun()

        if preview or run:
            try:
                with st.spinner("Running PromptPilot intelligence pipeline..."):
                    rid, analysis, plan, result, audio, elapsed = run_pipeline(
                        prompt, role, mode, execute=run, voice=voice
                    )
                st.session_state.request_id = rid
                st.session_state.analysis = analysis
                st.session_state.plan = plan
                st.session_state.result = result
                st.session_state.last_run_ms = elapsed
                st.session_state.error = ""
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "request_id": rid,
                    "intent": analysis.intent,
                    "mode": mode,
                    "latency": elapsed,
                    "prompt": prompt[:100],
                    "result": result,
                })
                if audio:
                    st.session_state.audio = audio
            except Exception as e:
                st.session_state.error = str(e)

        if st.session_state.error:
            st.error(st.session_state.error)

        if st.session_state.plan:
            st.markdown("### Optimized Prompt")
            st.code(st.session_state.plan.optimized_prompt, language="text")

            with st.expander("Planner details"):
                p = st.session_state.plan
                st.write("**Objective:**", p.objective)
                st.write("**Context:**", p.context)
                st.write("**Instructions:**")
                for x in p.instructions: st.write("• " + x)
                st.write("**Constraints:**")
                for x in p.constraints: st.write("• " + x)
                st.write("**Output format:**", p.output_format)
                if p.assumptions:
                    st.write("**Assumptions:**", p.assumptions)

        if st.session_state.result:
            st.markdown("### Execution Result")
            st.markdown(f'<div class="pp-output">{st.session_state.result}</div>', unsafe_allow_html=True)
            st.download_button(
                "Download result",
                st.session_state.result,
                file_name="promptpilot-result.txt",
                mime="text/plain",
            )
            if st.session_state.get("audio"):
                st.audio(st.session_state.audio, format="audio/mpeg")

    with right:
        st.markdown('<div class="pp-card-title">Live Intelligence</div><div class="pp-card-sub">Deterministic analysis runs before model execution.</div>', unsafe_allow_html=True)
        a = st.session_state.analysis
        if a:
            metrics = [
                ("Intent", a.intent),
                ("Completeness", f"{a.completeness}%"),
                ("Confidence", f"{a.confidence}%"),
                ("Complexity", f"{a.complexity:.2f}"),
                ("Route", "Quality" if a.route == QUALITY_MODEL else "Fast"),
                ("Latency", f"{st.session_state.last_run_ms} ms"),
            ]
            for label, value in metrics:
                st.markdown(f'<div class="pp-metric" style="margin-bottom:9px"><div class="pp-metric-label">{label}</div><div class="pp-metric-value" style="font-size:18px">{value}</div></div>', unsafe_allow_html=True)

            st.markdown("#### Signals")
            for key, value in a.signals.items():
                if isinstance(value, bool) and value:
                    st.markdown(f'<span class="pp-tag">{key}</span>', unsafe_allow_html=True)
            if a.missing:
                st.warning("Could improve: " + ", ".join(a.missing))
        else:
            st.info("Enter a request and run the pipeline to see intent, complexity, completeness, routing and confidence.")

        st.markdown("#### Quick templates")
        templates = {
            "Enterprise SQL": "Design a production-ready PostgreSQL database for a university management system with RBAC, audit logging, indexes, constraints, reporting and cloud deployment guidance.",
            "Research": "Help me formulate a novel AI research hypothesis, experimental design, baselines, evaluation metrics, ablations and reproducibility plan.",
            "API": "Design a secure FastAPI service with authentication, validation, observability, rate limiting, error handling, testing and deployment architecture.",
            "Data Science": "Analyze this business problem and propose a data science workflow including data quality checks, feature engineering, model selection and evaluation.",
        }
        for name, value in templates.items():
            if st.button(name, use_container_width=True):
                st.session_state.prompt = value
                st.rerun()

elif st.session_state.page == "History":
    st.title("Run History")
    st.caption("Session history is local to this Streamlit session. Supabase telemetry can provide persistent history.")
    if not st.session_state.history:
        st.info("No runs yet.")
    else:
        for item in st.session_state.history[:25]:
            with st.expander(f"{item['time']} · {item['intent']} · {item['latency']} ms"):
                st.write("Request ID:", item["request_id"])
                st.write("Prompt:", item["prompt"])
                if item["result"]:
                    st.markdown(item["result"])

elif st.session_state.page == "Architecture":
    st.title("Enterprise Architecture")
    st.caption("PromptPilot is structured as a layered application so providers can be replaced without rewriting the UI.")
    layers = [
        ("01 · Experience", "Streamlit UI, navigation, workspace state, responsive design system."),
        ("02 · Application", "User actions, session orchestration, validation and workflow control."),
        ("03 · Intelligence", "Signal extraction, completeness scoring, complexity estimation and model routing."),
        ("04 · Orchestration", "Structured prompt planning, execution, fallback and voice synthesis."),
        ("05 · Providers", "Groq, optional ElevenLabs, optional Supabase persistence."),
        ("06 · Observability", "Request IDs, latency, hashes, mode, intent and run telemetry."),
    ]
    for title, desc in layers:
        st.markdown(f'<div class="pp-card"><div class="pp-card-title">{title}</div><div class="pp-card-sub">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("### Processing flow")
    st.code("Input → Sanitize → Signals → Completeness → Complexity → Model Route → Structured Plan → Execute → Voice → Telemetry", language="text")

elif st.session_state.page == "Settings":
    st.title("Settings")
    st.caption("Streamlit Cloud: use App → Settings → Secrets for production credentials.")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Providers")
        st.write("Groq:", "Connected" if get_groq() else "Not configured")
        st.write("Supabase:", "Connected" if get_supabase() else "Not configured")
        st.write("ElevenLabs:", "Connected" if get_elevenlabs() else "Not configured")
    with cols[1]:
        st.markdown("#### Models")
        st.code(f"Fast: {FAST_MODEL}\nQuality: {QUALITY_MODEL}", language="text")
    st.info("API keys are intentionally never displayed by this UI.")

st.caption("PromptPilot Enterprise · Streamlit architecture · Blue/white enterprise UI")
