import yfinance as yf
import os
import re
import json
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from prompts import methods, SYSTEM_PROMPT, EXTRACT_ACTION_AND_TICKER_PROMPT, EXTRACT_RELEVANT_METHOD_PROMPT

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",             # Model identifier.
        api_key=os.environ.get("GEMINI_API_KEY"),   # Read API key from the environment.
    )
    return llm

def get_ticker_and_action_from_query(user_text: str) -> tuple:
    """Extract a stock ticker and a action the user would like performed, based on their input query"""
    llm = get_llm()
    prompt = [("system", EXTRACT_ACTION_AND_TICKER_PROMPT), ("user", user_text)]
    resp = llm.invoke(prompt)
    try:
        content = json.loads(resp.text.strip())
    except json.JSONDecodeError:
        return (None, user_text)
    if not isinstance(content, dict):
        return (None, user_text)
    return (content.get("ticker"), content.get("action"))


def get_specialized_methods_from_llm(action: str, all_methods:list)->list:
    """Of all methods callable on a specific stock ticker, return which of these methods relate to the action requested by the user, in a list format"""
    llm = get_llm()
    prompt = [("system", EXTRACT_RELEVANT_METHOD_PROMPT), ("user", f"Action: {action}\nAllowed methods: {', '.join(all_methods)}")]    
    resp = llm.invoke(prompt)
    content = json.loads(resp.text)
    return content if content else []

def yahoo_finance(ticker_symbol:str, method_list:list) -> dict:
    """For each method in method list, call the method and store in a dictionary defined as methodName:methodOutput"""
    output = dict()
    if ticker_symbol and ticker_symbol is not None:
        ticker = yf.Ticker(ticker_symbol)
        for method_name in method_list:
            try:
                if method_name.startswith("history"):
                    output["history"] = eval(f"ticker.{method_name}")
                    continue
                method = getattr(ticker, method_name, None)
                if method in methods and callable(method):
                    if method_name != 'live' and not method_name.startswith("live("):
                        output[method_name] = method()
                else:
                    output[method_name] = "Skipped: Not callable"
            except Exception as e:
                    output[method_name] = f"Error: {e}"
    return output

def display_stock_chart(ticker: str, yfi_output: dict) -> None:
    """Generate a stock chart based on historical data"""
   
    history_df = yfi_output.get("history") if yfi_output else None

    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return

    close = history_df["Close"].dropna()
    if close.empty:
        return

    ymin = float(close.min())
    ymax = float(close.max())
    yrange = ymax - ymin
    pad = max(yrange * 0.02, 1e-6)
    y_domain = [ymin - pad, ymax + pad]

    df = history_df.reset_index()
    time_col = "Date" if "Date" in df.columns else "Datetime"  # yfinance uses one of these
    if time_col not in df.columns:
        return

    spec = {
        "width": 900,
        "height": 420,
        "mark": {"type": "line", "interpolate": "linear", "strokeWidth": 2},
        "encoding": {
            "x": {
                "field": time_col,
                "type": "temporal",
                "axis": {"grid": True, "tickCount": 12, "labelAngle": 0},
            },
            "y": {
                "field": "Close",
                "type": "quantitative",
                "scale": {"domain": y_domain, "nice": False},
                "axis": {"grid": True, "tickCount": 8},
            },
        },
    }
    st.subheader(f"{ticker} - Stock Performance")
    st.vega_lite_chart(df[[time_col, "Close"]], spec)

def generate_final_response(history: list, yfi_output: dict) -> str:
    """Generate final LLM response with Yahoo Finance context."""
    llm = get_llm()
    messages = [("system", SYSTEM_PROMPT)] + history
    if yfi_output:
        messages.append(("system", f"Yahoo Finance tool output (JSON):\n{yfi_output}"))
    resp = llm.invoke(messages)
    return re.sub(r'\*+', '', resp.text).strip()

def summarizeHistory(history: list) -> list:
    """Trim conversation history by summarizing older messages."""
    N = len(history)
    toSummarize = history[:N-5]
    remaining = history[N-5:]
    chunk = "\n".join(f"{role}: {text}" for role, text in toSummarize)
    prompt = [
        ("system", "Update the running conversation summary. Return ONLY the updated summary."),
        ("user", chunk),
    ]
    remaining.append(("assistant",get_llm().invoke(prompt).text.strip()))
    return remaining
