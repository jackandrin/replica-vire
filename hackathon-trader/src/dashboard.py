"""Streamlit dashboard.  streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import ROOT, cfg, env, query_df  # noqa: E402
from src.predict import latest_pivot  # noqa: E402
from src.strategy import select_agreement  # noqa: E402

st.set_page_config(page_title="Hackathon Trader", layout="wide")
st.title("ML paper trader")
st.caption("Alpaca PAPER trading. Research prototype, not financial advice.")


@st.cache_data(ttl=60)
def piv():
    return latest_pivot()


@st.cache_data(ttl=60)
def orders():
    return query_df("SELECT created_at, symbol, side, notional, weight, mu_ridge, mu_mlp, status "
                    "FROM orders ORDER BY created_at DESC LIMIT 100")


@st.cache_data(ttl=60)
def runlog():
    return query_df("SELECT run_at, step, ok, detail FROM run_log ORDER BY run_at DESC LIMIT 30")


t1, t2, t3, t4 = st.tabs(["Predictions & agreement", "Positions & orders", "Honest results", "Pipeline health"])

with t1:
    p = piv()
    if p.empty:
        st.info("No predictions yet. Run: python -m src.predict")
    else:
        picks = select_agreement(p, cfg()["strategy"])
        p = p.assign(selected=p.symbol.isin(picks.symbol), conviction=p.mu_mlp / p.sigma_mlp)
        st.subheader(f"As of {p.asof.iloc[0]}  |  {len(picks)} names pass the agreement filter")
        st.dataframe(p.sort_values("mu_mlp", ascending=False), use_container_width=True)
        st.bar_chart(p.set_index("symbol")[["mu_ridge", "mu_mlp"]])
        st.write("Selected:", picks)

with t2:
    try:
        from alpaca.trading.client import TradingClient
        tc = TradingClient(env("ALPACA_API_KEY"), env("ALPACA_API_SECRET"), paper=True)
        a = tc.get_account()
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"${float(a.equity):,.0f}")
        c2.metric("Cash", f"${float(a.cash):,.0f}")
        c3.metric("Market open", str(tc.get_clock().is_open))
        pos = pd.DataFrame([{"symbol": x.symbol, "qty": float(x.qty), "market_value": float(x.market_value),
                             "unrealized_pl": float(x.unrealized_pl)} for x in tc.get_all_positions()])
        st.subheader("Positions")
        st.dataframe(pos, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Alpaca unavailable: {e}")
    st.subheader("Order log")
    st.dataframe(orders(), use_container_width=True)

with t3:
    mp = ROOT / "results/metrics.json"
    if not mp.exists():
        st.info("Run: python -m src.train")
    else:
        m = json.loads(mp.read_text())
        rows = [{"model": k, "IC": v["ic_mean"], "hit_rate": v["hit_rate"], "naive_t": v["ic_tstat_naive"],
                 "n_days": v["n_days"]} for k, v in m.items() if k.startswith("mu_")]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        bt = pd.DataFrame(m["backtest"]).T[["total_return", "sharpe", "max_drawdown", "n_periods"]]
        st.dataframe(bt, use_container_width=True)
        cv = ROOT / "results/backtest_curve.csv"
        if cv.exists():
            curve = pd.read_csv(cv, index_col=0)
            st.line_chart((1 + curve[["ridge", "mlp", "agreement", "spy", "spy_at_gross"]]).cumprod())
        st.warning(m["meta"]["caveat"])
        st.caption(f"Train {m['meta']['train_dates']} | Test {m['meta']['test_dates']}")

with t4:
    st.dataframe(runlog(), use_container_width=True)
