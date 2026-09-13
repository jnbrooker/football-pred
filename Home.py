"""
Home.py — landing page for the football match-model suite (multi-page Streamlit app).

Run with:
    streamlit run Home.py

Each stat lives on its own page (see the sidebar). They all share the data
and model layer in `footsim/`, and they all feed the Match Simulation page.
"""

import streamlit as st

st.set_page_config(page_title="Football Model Suite", page_icon="⚽", layout="wide")

st.title("⚽ Football Monte Carlo Model Suite")
st.write(
    "Pick a page from the sidebar. Each one simulates a Premier League match thousands "
    "of times from distributions fitted to real shot-by-shot data, adjusts for the "
    "opponent and the venue, and shows the full distribution of outcomes plus the "
    "chance of beating a line."
)

st.subheader("Pages")
st.markdown(
    "- **Shots** — minutes → team shots → the player's share; the reference chain.\n"
    "- **Shots on Target** — each shot on target at his own rate, moved by the opponent.\n"
    "- **Goals** — shots × conversion (xG per shot × finishing), penalties by taker; anytime-scorer odds.\n"
    "- **Assists** — key passes → converted, scaled by the team's expected goals.\n"
    "- **Goalkeeper Saves** — the opponent's on-target shots minus the goals conceded; clean-sheet odds.\n"
    "- **Cards** — yellows per 90, regressed toward the position.\n"
    "- **Team Strength** — attack and defence solved from shots, xG per shot and finishing; home advantage; the base of the match model.\n"
    "- **Match Simulation** — pick a fixture: score distribution, 1X2, goal lines, both box scores, every identity holding in every simulation.\n"
    "- **Fantasy Projections** — every player on the gameweek's slate scored on FPL rules per simulation, with floor, ceiling and a breakdown.\n"
    "- **Backtest** — every model scored out of sample, match date by match date, against the closing market and a trailing average.\n"
)

st.subheader("How it works")
st.write(
    "Everything is self-contained: priors come from Understat shot data and the FPL "
    "squad feed, and the match model forms its own view of team strength rather than "
    "anchoring to a bookmaker. Recent matches count more than old ones (the sidebar's "
    "recency control), the squad layer knows who is available, and every constant "
    "that could be fitted was fitted on out-of-sample residuals. See `README.md` for "
    "the model and `ROADMAP.md` for how it got here and what the data said."
)
st.caption("For research and entertainment.")
