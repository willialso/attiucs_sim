import pandas as pd
import random
from datetime import datetime
import xlsxwriter

# === BEHAVIOR PROFILES ===
BEHAVIOR_PROFILES = {
    "A": {"population": 250, "sessions_per_day": (2, 8), "trades_per_session": (2, 8), "seconds_between_trades": (6, 15), "first_trade_up": (0.45, 0.55), "subsequent_same": (0.55, 0.75), "trade_amount": (25, 75)},
    "B": {"population": 250, "sessions_per_day": (2, 8), "trades_per_session": (2, 8), "seconds_between_trades": (6, 20), "first_trade_up": (0.45, 0.55), "subsequent_same": (0.60, 0.80), "trade_amount": (5, 25)},
    "C": {"population": 100, "sessions_per_day": (4, 12), "trades_per_session": (2, 8), "seconds_between_trades": (6, 20), "first_trade_up": (0.48, 0.58), "subsequent_same": (0.65, 0.85), "trade_amount": (5, 15)},
    "D": {"population": 250, "sessions_per_day": (5, 25), "trades_per_session": (2, 8), "seconds_between_trades": (6, 20), "first_trade_up": (0.50, 0.62), "subsequent_same": (0.70, 0.90), "trade_amount": (1, 10)}
}

NC_PUSH_FEE = 0.25
UP_PAYOUT_ODDS = -120
DOWN_PAYOUT_ODDS = -120

# === LOAD BTC DATA ===
df = pd.read_csv("analyzed_price_changes.csv")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.rename(columns={"up_/_down_/_no_change": "direction"})
df['direction'] = df['direction'].str.lower().str.strip()
df['date'] = df['initial_time'].str[:10]

# === SIMULATE ===
trades = []
unique_dates = sorted(df['date'].unique())[:7]

for date in unique_dates:
    df_day = df[df['date'] == date].reset_index(drop=True)
    trader_bet_amounts = {}
    for trader_type, config in BEHAVIOR_PROFILES.items():
        for i in range(config["population"]):
            trader_id = f"{trader_type}_Trader_{i+1}"
            trader_bet_amounts[trader_id] = random.randint(*config["trade_amount"])

    for trader_id, fixed_bet_amount in trader_bet_amounts.items():
        trader_type = trader_id.split("_")[0]
        config = BEHAVIOR_PROFILES[trader_type]
        num_sessions = random.randint(*config["sessions_per_day"])

        for session_id in range(num_sessions):
            num_trades = random.randint(*config["trades_per_session"])
            behavior_same = random.uniform(*config["subsequent_same"])
            last_valid_actual = None

            if len(df_day) < num_trades + 10:
                continue

            trade_indices = random.sample(range(len(df_day) - 10), num_trades)

            for trade_num, idx in enumerate(trade_indices):
                row = df_day.iloc[idx]
                actual = row['direction']
                session_key = f"{trader_id}_{date}_session{session_id + 1}"

                if actual == 'no change':
                    trades.append({
                        "trader_id": trader_id,
                        "trader_type": trader_type,
                        "session_id": session_id + 1,
                        "session_key": session_key,
                        "trade_id": trade_num + 1,
                        "date": date,
                        "timestamp_trade_placed": row["initial_time"],
                        "strike_time": row["strike_time"],
                        "price_at_trade": row["initial_price"],
                        "price_after_5_sec": row["strike_price"],
                        "predicted_direction": last_valid_actual.title() if last_valid_actual else random.choice(["Up", "Down"]),
                        "actual_direction": "No Change",
                        "trade_result": "No Change",
                        "trade_amount": fixed_bet_amount,
                        "trade_strategy": "No Change",
                        "up_price": UP_PAYOUT_ODDS,
                        "down_price": DOWN_PAYOUT_ODDS,
                        "nc_fee": NC_PUSH_FEE,
                        "revenue": NC_PUSH_FEE
                    })
                    continue

                if trade_num == 0 or last_valid_actual is None:
                    pred = random.choices(["up", "down"], weights=[config["first_trade_up"][1], config["first_trade_up"][0]])[0]
                    strategy = "Initial"
                else:
                    strategy = "Last Bet Outcome"
                    pred = last_valid_actual if random.random() < behavior_same else ("down" if last_valid_actual == "up" else "up")

                result = "Win Up" if pred == "up" and actual == "up" else \
                         "Win Down" if pred == "down" and actual == "down" else \
                         "Loss"

                if result == "Win Up":
                    revenue = -round(fixed_bet_amount / (abs(UP_PAYOUT_ODDS) / 100), 2)
                elif result == "Win Down":
                    revenue = -round(fixed_bet_amount / (abs(DOWN_PAYOUT_ODDS) / 100), 2)
                elif result == "Loss":
                    revenue = fixed_bet_amount

                trades.append({
                    "trader_id": trader_id,
                    "trader_type": trader_type,
                    "session_id": session_id + 1,
                    "session_key": session_key,
                    "trade_id": trade_num + 1,
                    "date": date,
                    "timestamp_trade_placed": row["initial_time"],
                    "strike_time": row["strike_time"],
                    "price_at_trade": row["initial_price"],
                    "price_after_5_sec": row["strike_price"],
                    "predicted_direction": pred.title(),
                    "actual_direction": actual.title(),
                    "trade_result": result,
                    "trade_amount": fixed_bet_amount,
                    "trade_strategy": strategy,
                    "up_price": UP_PAYOUT_ODDS,
                    "down_price": DOWN_PAYOUT_ODDS,
                    "nc_fee": 0.00,
                    "revenue": revenue
                })

# === EXPORT ===
df_trades = pd.DataFrame(trades)

# Trader Summary
df_valid = df_trades[df_trades["trade_result"].isin(["Win Up", "Win Down", "Loss"])]
df_trader_summary = df_valid.groupby("trader_id").agg(
    trader_type=("trader_type", "first"),
    total_sessions=("session_key", "nunique"),
    trades=("trade_id", "count"),
    total_wins=("trade_result", lambda x: x.str.startswith("Win").sum()),
    total_losses=("trade_result", lambda x: (x == "Loss").sum()),
    win_rate_percent=("trade_result", lambda x: round(x.str.startswith("Win").mean() * 100, 2)),
    up_count=("predicted_direction", lambda x: (x == "Up").sum()),
    down_count=("predicted_direction", lambda x: (x == "Down").sum()),
    trade_amount=("trade_amount", "mean")
).reset_index()

# Session Summary
session_summaries = []
for session_key, group in df_trades.groupby("session_key"):
    group = group.sort_values("trade_id").reset_index(drop=True)
    if group.empty:
        continue

    first_bet_up = group.loc[0, "predicted_direction"] == "Up"
    trader_id = group.loc[0, "trader_id"]
    trader_type = group.loc[0, "trader_type"]
    date = group.loc[0, "date"]
    total_trades = len(group)
    tag = f"{trader_id}_{date}_Session{group.loc[0, 'session_id']}"

    up_after_up = down_after_up = up_after_down = down_after_down = skipped = 0
    for i in range(1, total_trades):
        prev_actual = group.loc[i - 1, "actual_direction"]
        curr_pred = group.loc[i, "predicted_direction"]

        if prev_actual not in ["Up", "Down"]:
            skipped += 1
            continue

        if prev_actual == "Up":
            if curr_pred == "Up":
                up_after_up += 1
            else:
                down_after_up += 1
        elif prev_actual == "Down":
            if curr_pred == "Up":
                up_after_down += 1
            else:
                down_after_down += 1

    session_summaries.append({
        "date": date,
        "session_key": session_key,
        "trader_id": trader_id,
        "trader_type": trader_type,
        "trades": total_trades,
        "first_bet_up": first_bet_up,
        "session_tag": tag,
        "non_first_down_given_last_win_up": down_after_up,
        "non_first_up_given_last_win_down": up_after_down,
        "non_first_down_given_last_win_down": down_after_down,
        "non_first_nc": skipped
    })

session_summary_df = pd.DataFrame(session_summaries)

with pd.ExcelWriter("TypeA-D_Sim_FullTraders_v3_Output.xlsx", engine="xlsxwriter") as writer:
    df_trades.to_excel(writer, sheet_name="Full Trade Log", index=False)
    df_trader_summary.to_excel(writer, sheet_name="Trader Summary", index=False)
    session_summary_df.to_excel(writer, sheet_name="Session Summary", index=False)

print("✅ Type A–D simulation with payout-based revenue and summaries completed.")
