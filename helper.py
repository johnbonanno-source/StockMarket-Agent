import yfinance as yf
import os
import re
import json
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from prompts import methods, SYSTEM_PROMPT, EXTRACT_ACTION_AND_TICKER_PROMPT, EXTRACT_RELEVANT_METHOD_PROMPT

COMPANY_ALIASES = {
    "micron": "MU",
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META",
    "google": "GOOGL",
    "alphabet": "GOOGL",
}
LEADING_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good morning|good afternoon)\b(?:[\s,!.?:;-]+|$)",
    re.IGNORECASE,
)
EXPLICIT_TICKER_PATTERNS = (
    re.compile(r"\$([A-Z]{1,5})\b"),
    re.compile(r"\b(?:ticker|symbol)\s+([A-Z]{1,5})\b", re.IGNORECASE),
)
PERIOD_RULES = (
    ("5d", ("today", "right now", "intraday", "this week", "weekly")),
    ("1mo", ("this month", "monthly")),
    ("1y", ("this year", "ytd")),
)
INTENT_METHOD_RULES = (
    ("history", ("price", "stock", "doing", "performance", "trend", "chart")),
    ("get_earnings_history", ("earnings",)),
    ("get_news", ("news",)),
)

def _default_plan(user_text: str) -> dict:
    return {"ticker": None, "action": user_text, "methods": [], "history": {"period": "5d"}}

def _strip_leading_greeting(user_text: str) -> str:
    return LEADING_GREETING_RE.sub("", user_text, count=1).strip()

def _find_alias_matches(user_text: str) -> list[tuple[str, str]]:
    matches = []
    lowered = user_text.lower()
    for company, ticker in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(company)}\b", lowered):
            matches.append((company, ticker))
    return matches

def _find_explicit_tickers(user_text: str) -> list[str]:
    matches = []
    for pattern in EXPLICIT_TICKER_PATTERNS:
        matches.extend(match.group(1).upper() for match in pattern.finditer(user_text))
    return matches

def _choose_period(user_text: str) -> str:
    lowered = user_text.lower()
    for period, keywords in PERIOD_RULES:
        if any(keyword in lowered for keyword in keywords):
            return period
    return "1mo"

def _choose_methods(user_text: str) -> list[str]:
    lowered = user_text.lower()
    selected = []
    for method_name, keywords in INTENT_METHOD_RULES:
        if any(keyword in lowered for keyword in keywords):
            selected.append(method_name)
    return selected or ["history"]

def try_fast_plan(user_text: str) -> dict | None:
    cleaned_text = _strip_leading_greeting(user_text)
    alias_matches = _find_alias_matches(cleaned_text)
    explicit_tickers = _find_explicit_tickers(cleaned_text)

    candidate_tickers = {ticker for _, ticker in alias_matches}
    candidate_tickers.update(explicit_tickers)
    if len(alias_matches) > 1 or len(explicit_tickers) > 1 or len(candidate_tickers) != 1:
        return None

    ticker = next(iter(candidate_tickers), None)
    if not ticker:
        return None

    plan = {
        "ticker": ticker,
        "action": user_text,
        "methods": _choose_methods(cleaned_text),
        "history": {"period": _choose_period(cleaned_text)},
    }
    print(f"[DEBUG] try_fast_plan matched | plan={plan}")
    return plan

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    print("[DEBUG] get_llm called")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",             # Model identifier.
        api_key=os.environ.get("GEMINI_API_KEY"),   # Read API key from the environment.
    )
    return llm

def get_ticker_and_action_from_query(user_text: str) -> dict:
    """Extract stock request plan from user query."""
    print(f"[DEBUG] get_ticker_and_action_from_query called | user_text={user_text!r}")
    fast_plan = try_fast_plan(user_text)
    if fast_plan is not None:
        return fast_plan
    llm = get_llm()
    prompt = [("system", EXTRACT_ACTION_AND_TICKER_PROMPT), ("user", user_text)]
    resp = llm.invoke(prompt)
    try:
        content = json.loads(resp.text.strip())
    except json.JSONDecodeError:
        return _default_plan(user_text)
    if not isinstance(content, dict):
        return _default_plan(user_text)

    content.setdefault("ticker", None)
    content.setdefault("action", user_text)
    content.setdefault("methods", [])
    content.setdefault("history", {})

    if not isinstance(content["methods"], list):
        content["methods"] = []
    if not isinstance(content["history"], dict):
        content["history"] = {}
    content["history"].setdefault("period", "5d")
    return content


def get_specialized_methods_from_llm(action: str, all_methods:list)->list:
    """Of all methods callable on a specific stock ticker, return which of these methods relate to the action requested by the user, in a list format"""
    print(f"[DEBUG] get_specialized_methods_from_llm called | action={action!r}, all_methods_count={len(all_methods)}")
    llm = get_llm()
    prompt = [("system", EXTRACT_RELEVANT_METHOD_PROMPT), ("user", f"Action: {action}\nAllowed methods: {', '.join(all_methods)}")]    
    resp = llm.invoke(prompt)
    try:
        content = json.loads(resp.text)
    except (json.JSONDecodeError, AttributeError):
        return []
    return content if isinstance(content, list) else []

def choose_interval(period: str) -> str:
    p = (period or "5d").lower().strip()
    # Yahoo intraday intervals are only available for limited lookback windows.
    if p.endswith("y") or p.endswith("mo"):
        return "1d"
    return "1h"

@st.cache_data(ttl=60)
def yahoo_finance(ticker_symbol: str, method_list: list, history_cfg: dict | None = None) -> dict:
    """For each method in method list, call the method and store in a dictionary defined as methodName:methodOutput"""
    print(
        f"[DEBUG] yahoo_finance called | ticker_symbol={ticker_symbol!r}, "
        f"method_list={method_list!r}, history_cfg={history_cfg!r}"
    )
    output = dict()
    history_cfg = history_cfg or {}
    history_period = history_cfg.get("period", "5d")
    history_interval = choose_interval(history_period)

    if ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        for method_name in method_list:
            try:
                if method_name.startswith("history"):
                    output["history"] = ticker.history(period=history_period, interval=history_interval).tail(50)
                    continue

                if method_name == "live" or method_name.startswith("live("):
                    output["live"] = "Skipped: streaming method disabled"
                    continue

                if method_name not in methods:
                    output[method_name] = "Skipped: Disallowed Method"
                    continue

                method = getattr(ticker, method_name, None)
                if callable(method):
                    output[method_name] = method()
                else:
                    output[method_name] = "Skipped: Not callable"
            except Exception as e:
                output[method_name] = f"Error: {e}"
    return output

def display_stock_chart(ticker: str, yfi_output: dict) -> None:
    """Render stock close price over time."""
    history_df = yfi_output.get("history") if isinstance(yfi_output, dict) else None
    if history_df is None or "Close" not in history_df.columns:
        return

    st.subheader(f"{ticker} - Stock Performance")
    st.line_chart(history_df["Close"])

def generate_final_response(history: list, yfi_output: dict) -> str:
    """Generate final LLM response with Yahoo Finance context."""
    print(f"[DEBUG] generate_final_response called | history_len={len(history)}, has_yfi_output={bool(yfi_output)}")
    llm = get_llm()
    messages = [("system", SYSTEM_PROMPT)] + history
    
    if yfi_output:
        messages.append(("system", f"Yahoo Finance tool output (JSON):\n{json.dumps(yfi_output, default=str, separators=(',', ':'))}"))
    resp = llm.invoke(messages)
    return re.sub(r'\*+', '', resp.text).strip()

def summarizeHistory(history: list) -> list:
    """Trim conversation history by summarizing older messages."""
    print(f"[DEBUG] summarizeHistory called | history_len={len(history)}")
    N = len(history)
    toSummarize = history[:N-5]
    remaining = history[N-5:]
    chunk = "\n".join(f"{role}: {text}" for role, text in toSummarize)
    prompt = [
        ("system", "Update the running conversation summary. Return ONLY the updated summary."),
        ("user", chunk),
    ]
    summary = ("assistant", get_llm().invoke(prompt).text.strip())
    return [summary, *remaining]

