import yfinance as yf
import os
import re
import json
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import altair as alt

from prompts import methods, SYSTEM_PROMPT, EXTRACT_ACTION_AND_TICKER_PROMPT, EXTRACT_RELEVANT_METHOD_PROMPT

EXPLICIT_TICKER_PATTERNS = (
    re.compile(r"\$([A-Z]{1,5})\b"),
    re.compile(r"\b(?:ticker|symbol)\s+([A-Z]{1,5})\b", re.IGNORECASE),
)
QUOTE_TYPE_PRIORITY = {"EQUITY": 0, "ETF": 1, "INDEX": 2}
US_EXCHANGES = {"NMS", "NYQ", "NGM"}
MAJOR_INDEX_ALIASES = {
    "s&p 500": "^GSPC",
    "s&p": "^GSPC",
    "sp500": "^GSPC",
    "dow jones": "^DJI",
    "dow": "^DJI",
    "nasdaq composite": "^IXIC",
    "nasdaq": "^IXIC",
    "russell 2000": "^RUT",
}

def _default_plan(user_text: str) -> dict:
    return {"ticker": None, "company": None, "action": user_text, "methods": [], "history": {"period": "5d"}}

def _find_explicit_tickers(user_text: str) -> list[str]:
    matches = []
    for pattern in EXPLICIT_TICKER_PATTERNS:
        matches.extend(match.group(1).upper() for match in pattern.finditer(user_text))
    return matches

def _with_explicit_ticker_fallback(plan: dict, user_text: str) -> dict:
    if plan.get("ticker"):
        return plan

    explicit_tickers = set(_find_explicit_tickers(user_text))
    if len(explicit_tickers) == 1:
        plan["ticker"] = next(iter(explicit_tickers))
    return plan

def _quote_rank(quote: dict) -> tuple:
    symbol = quote.get("symbol", "")
    return (
        QUOTE_TYPE_PRIORITY.get(quote.get("quoteType"), 99),
        "." in symbol,
        quote.get("exchange") not in US_EXCHANGES,
    )

def resolve_major_index(query: str) -> str | None:
    q = query.lower()
    for name, symbol in MAJOR_INDEX_ALIASES.items():
        if name in q and has_market_data(symbol):
            return symbol
    return None

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
    llm = get_llm()
    prompt = [("system", EXTRACT_ACTION_AND_TICKER_PROMPT), ("user", user_text)]
    resp = llm.invoke(prompt)
    try:
        content = json.loads(resp.text.strip())
    except json.JSONDecodeError:
        return _with_explicit_ticker_fallback(_default_plan(user_text), user_text)
    if not isinstance(content, dict):
        return _with_explicit_ticker_fallback(_default_plan(user_text), user_text)

    content.setdefault("ticker", None)
    content.setdefault("company", None)
    content.setdefault("action", user_text)
    content.setdefault("methods", [])
    content.setdefault("history", {})

    if not isinstance(content["methods"], list):
        content["methods"] = []
    if not isinstance(content["history"], dict):
        content["history"] = {}
    content["history"].setdefault("period", "5d")
    return _with_explicit_ticker_fallback(content, user_text)

@st.cache_data(ttl=60)
def has_market_data(ticker_symbol: str) -> bool:
    try:
        return not yf.Ticker(ticker_symbol).history(period="5d").empty
    except Exception:
        return False

@st.cache_data(ttl=300)
def resolve_ticker(query: str) -> str | None:
    major_index = resolve_major_index(query)
    if major_index:
        return major_index

    try:
        search = yf.Search(
            query,
            max_results=6,
            news_count=0,
            lists_count=0,
            include_cb=False,
            timeout=10,
            raise_errors=False,
        )
    except Exception:
        return None

    quotes = sorted(getattr(search, "quotes", []), key=_quote_rank)
    for quote in quotes:
        symbol = quote.get("symbol")
        quote_type = quote.get("quoteType")
        if symbol and quote_type in QUOTE_TYPE_PRIORITY and has_market_data(symbol):
            return symbol
    return None

def validate_or_resolve_ticker(ticker_symbol: str | None, search_query: str | None, user_text: str) -> str | None:
    explicit_tickers = set(_find_explicit_tickers(user_text))
    if ticker_symbol and ticker_symbol in explicit_tickers and has_market_data(ticker_symbol):
        return ticker_symbol

    resolved_ticker = resolve_ticker(search_query or user_text)
    if resolved_ticker:
        return resolved_ticker

    if ticker_symbol and not search_query and has_market_data(ticker_symbol):
        return ticker_symbol
    return None


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
    if p == "1d":
        return "5m"
    if p == "5d":
        return "30m"
    if p.endswith("y") or p.endswith("mo"):
        return "1d"
    return "1d"

def make_cache_safe(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError, RecursionError):
        return str(value)

@st.cache_data(ttl=60)
def yahoo_finance(ticker_symbol: str, method_list: tuple, history_cfg: dict | None = None) -> dict:
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
                    output[method_name] = make_cache_safe(method())
                else:
                    output[method_name] = "Skipped: Not callable"
            except Exception as e:
                output[method_name] = f"Error: {e}"
    return output

def display_stock_chart(ticker: str, yfi_output: dict) -> None:
    """Render stock close price over time."""
    history_df = yfi_output.get("history") if isinstance(yfi_output, dict) else None
    if not hasattr(history_df, "columns") or "Close" not in history_df.columns:
        return

    close = history_df["Close"].dropna()
    if close.empty:
        return

    padding = max((close.max() - close.min()) * 0.15, close.mean() * 0.002)
    chart_df = close.reset_index()
    chart_df.columns = ["Date", "Close"]
    chart = alt.Chart(chart_df).mark_line().encode(
        x="Date:T",
        y=alt.Y("Close:Q", scale=alt.Scale(domain=[close.min() - padding, close.max() + padding])),
    )

    st.subheader(f"{ticker} - Stock Performance")
    st.altair_chart(chart, use_container_width=True)

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
