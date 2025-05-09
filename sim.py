import pandas as pd
import numpy as np
import random
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

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

class Simulation:
    def __init__(self, sim_id):
        self.sim_id = sim_id
        self.trades = []
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and prepare BTC data"""
        self.df = pd.read_csv("analyzed_price_changes.csv")
        self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]
        self.df = self.df.rename(columns={"up_/_down_/_no_change": "direction"})
        self.df['direction'] = self.df['direction'].str.lower().str.strip()
        self.df['date'] = self.df['initial_time'].str[:10]
        
    def run_simulation(self):
        """Run a single simulation"""
        try:
            unique_dates = sorted(self.df['date'].unique())[:7]
            
            for date in unique_dates:
                df_day = self.df[self.df['date'] == date].reset_index(drop=True)
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

                            trade_data = self.process_trade(
                                trader_id, trader_type, session_id, session_key,
                                trade_num, date, row, actual, last_valid_actual,
                                fixed_bet_amount, config, behavior_same
                            )
                            
                            self.trades.append(trade_data)
                            if actual != 'no change':
                                last_valid_actual = actual

            return self.generate_summaries()
        except Exception as e:
            logging.error(f"Error in simulation {self.sim_id}: {str(e)}")
            return None

    def process_trade(self, trader_id, trader_type, session_id, session_key, trade_num, date, row, actual, last_valid_actual, fixed_bet_amount, config, behavior_same):
        """Process a single trade"""
        if actual == 'no change':
            return {
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
                "revenue": NC_PUSH_FEE,
                "sim_id": self.sim_id
            }

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
        else:
            revenue = fixed_bet_amount

        return {
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
            "revenue": revenue,
            "sim_id": self.sim_id
        }

    def generate_summaries(self):
        """Generate summary DataFrames"""
        df_trades = pd.DataFrame(self.trades)
        
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
        df_trader_summary['sim_id'] = self.sim_id

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
                "non_first_nc": skipped,
                "sim_id": self.sim_id
            })

        session_summary_df = pd.DataFrame(session_summaries)
        
        return {
            'trades': df_trades,
            'trader_summary': df_trader_summary,
            'session_summary': session_summary_df
        }

def run_multiple_simulations(num_sims=10):
    """Run multiple simulations and collect results"""
    all_results = {
        'trades': [],
        'trader_summary': [],
        'session_summary': []
    }
    
    for sim_id in range(1, num_sims + 1):
        logging.info(f"Running simulation {sim_id}")
        sim = Simulation(sim_id)
        results = sim.run_simulation()
        
        if results:
            for key in all_results:
                all_results[key].append(results[key])
    
    # Combine results from all simulations
    combined_results = {
        key: pd.concat(dfs, ignore_index=True) 
        for key, dfs in all_results.items()
    }
    
    return combined_results

def save_to_excel(results, output_file='Rev_Sim_Output_Tables.xlsx'):
    """Save simulation results to Excel with multiple sheets"""
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save each DataFrame to its own sheet
            results['trades'].to_excel(writer, sheet_name="Full Trade Log", index=False)
            results['trader_summary'].to_excel(writer, sheet_name="Trader Summary", index=False)
            results['session_summary'].to_excel(writer, sheet_name="Session Summary", index=False)
            
            # Create pivot tables for analysis across simulations
            pivot_trades = pd.pivot_table(
                results['trades'],
                values=['trade_amount', 'revenue'],
                index=['sim_id', 'trader_type'],
                aggfunc={'trade_amount': 'mean', 'revenue': ['sum', 'mean']}
            ).reset_index()
            pivot_trades.to_excel(writer, sheet_name="Simulation Analysis", index=False)
        
        logging.info(f"Results saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving to Excel: {str(e)}")
        return False

def main():
    logging.info("Starting simulation batch")
    
    # Run simulations
    results = run_multiple_simulations(num_sims=10)
    
    # Save results
    if results:
        save_to_excel(results)
    else:
        logging.error("No results to save")
    
    logging.info("Simulation batch completed")

if __name__ == "__main__":
    main() 