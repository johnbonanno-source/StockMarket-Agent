import streamlit as st
from helper import get_ticker_and_action_from_query, validate_or_resolve_ticker, yahoo_finance, display_stock_chart, generate_final_response, summarizeHistory

def main():
    st.title("Stock Market Agent")

    if "history" not in st.session_state:
        st.session_state.history = []
    if len(st.session_state.history) >= 20:
        st.session_state.history = summarizeHistory(st.session_state.history)

    for role, text in st.session_state.history:
        st.chat_message(role).text(text)
    user_text = st.chat_input("")
    if not user_text:
        return
    st.session_state.history.append(("user", user_text))
    st.chat_message("user").text(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            request_plan = get_ticker_and_action_from_query(user_text)
            print(f"[DEBUG] request_plan={request_plan}")
            ticker = None
            if request_plan.get("ticker") or request_plan.get("company"):
                ticker = validate_or_resolve_ticker(
                    request_plan.get("ticker"),
                    request_plan.get("company"),
                    user_text,
                )
            yfi_methods = request_plan.get("methods", [])
            history_cfg = request_plan.get("history", {})
            yfi_output = None
            if ticker:
                if not yfi_methods:
                    yfi_methods = ["history"]
                print(yfi_methods)
                yfi_output = yahoo_finance(ticker, tuple(yfi_methods), history_cfg)
                # plot the chart
                display_stock_chart(ticker, yfi_output)
            resp = generate_final_response(st.session_state.history, yfi_output)
        st.session_state.history.append(("assistant",resp))
        st.text(resp)
if __name__ == "__main__":
    st.set_page_config(page_title="Chatbot", page_icon="💬")
    main()
