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
        
        # Convert timestamps to datetime for time calculations
        df_trades['timestamp_trade_placed'] = pd.to_datetime(df_trades['timestamp_trade_placed'], format='ISO8601')
        df_trades['strike_time'] = pd.to_datetime(df_trades['strike_time'], format='ISO8601')
        
        return {
            'trades': df_trades
        }

def calculate_session_metrics(df_trades):
    """Calculate session-level metrics for Tab 1"""
    metrics = {}
    
    for sim_id in df_trades['sim_id'].unique():
        df_sim = df_trades[df_trades['sim_id'] == sim_id].copy()
        sim_metrics = {}
        
        for trader_type in ['A', 'B', 'C', 'D']:
            df_type = df_sim[df_sim['trader_type'] == trader_type].copy()
            
            # Calculate metrics
            total_traders = df_type['trader_id'].nunique()
            total_sessions = df_type['session_key'].nunique()
            
            # Avg trades per session
            trades_per_session = df_type.groupby('session_key').size()
            avg_trades_per_session = trades_per_session.mean()
            
            # Time between trades
            df_type = df_type.sort_values(['session_key', 'trade_id'])
            df_type['next_trade_time'] = df_type.groupby('session_key')['timestamp_trade_placed'].shift(-1)
            df_type['time_between_trades'] = (df_type['next_trade_time'] - df_type['strike_time']).dt.total_seconds()
            avg_seconds_between_trades = df_type['time_between_trades'].mean()
            
            # First trade up percentage
            first_trades = df_type[df_type['trade_id'] == 1].copy()
            first_trade_up_pct = (first_trades['predicted_direction'] == 'Up').mean() * 100
            
            # Subsequent trade direction matching
            subsequent_trades = df_type[df_type['trade_id'] > 1].copy()
            subsequent_trades.loc[:, 'prev_direction'] = subsequent_trades.groupby('session_key')['predicted_direction'].shift(1)
            direction_match_pct = (subsequent_trades['predicted_direction'] == subsequent_trades['prev_direction']).mean() * 100
            
            # Average trade amount
            avg_trade_amount = df_type['trade_amount'].mean()
            
            # Total revenue
            total_revenue = df_type['revenue'].sum()
            
            sim_metrics[trader_type] = {
                'Total Traders': total_traders,
                'Total Sessions': total_sessions,
                'Avg Trades per Session': avg_trades_per_session,
                'Avg Seconds Between Trades': avg_seconds_between_trades,
                'First Trade Up %': first_trade_up_pct,
                'Subsequent Direction Match %': direction_match_pct,
                'Avg Trade Amount': avg_trade_amount,
                'Total Revenue': total_revenue
            }
        
        metrics[sim_id] = sim_metrics
    
    return metrics

def calculate_trader_metrics(df_trades):
    """Calculate trader-level metrics for Tab 2"""
    metrics = {}
    
    for sim_id in df_trades['sim_id'].unique():
        df_sim = df_trades[df_trades['sim_id'] == sim_id].copy()
        sim_metrics = {}
        
        for trader_type in ['A', 'B', 'C', 'D']:
            df_type = df_sim[df_sim['trader_type'] == trader_type].copy()
            
            # Calculate metrics
            total_traders = df_type['trader_id'].nunique()
            total_trades = len(df_type)
            
            # Winning percentage
            winning_trades = df_type[df_type['trade_result'].str.startswith('Win')]
            winning_pct = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            
            # Average trade amount
            avg_trade_amount = df_type['trade_amount'].mean()
            
            # Total revenue
            total_revenue = df_type['revenue'].sum()
            
            sim_metrics[trader_type] = {
                'Total Traders': total_traders,
                'Total Trades': total_trades,
                'Winning %': winning_pct,
                'Avg Trade Amount': avg_trade_amount,
                'Total Revenue': total_revenue
            }
        
        metrics[sim_id] = sim_metrics
    
    return metrics

def calculate_revenue_summary(df_trades):
    """Calculate revenue summary for Tab 3"""
    metrics = {}
    
    for sim_id in df_trades['sim_id'].unique():
        df_sim = df_trades[df_trades['sim_id'] == sim_id].copy()
        sim_metrics = {}
        
        # Calculate revenue for each trader type
        for trader_type in ['A', 'B', 'C', 'D']:
            df_type = df_sim[df_sim['trader_type'] == trader_type].copy()
            total_revenue = df_type['revenue'].sum()
            
            sim_metrics[trader_type] = {
                'Total Revenue': total_revenue
            }
        
        # Calculate Atticus revenue (sum of all trader revenues)
        atticus_revenue = df_sim['revenue'].sum()
        sim_metrics['Atticus'] = {
            'Total Revenue': atticus_revenue
        }
        
        metrics[sim_id] = sim_metrics
    
    return metrics

def create_formatted_dataframe(metrics, metric_names):
    """Create a formatted DataFrame with trader types as column headers"""
    # Create multi-level columns
    columns = pd.MultiIndex.from_product([['Type A', 'Type B', 'Type C', 'Type D'], metric_names])
    
    # Create empty DataFrame with multi-level columns
    df = pd.DataFrame(index=range(len(metrics)), columns=columns)
    
    # Fill in the data
    for sim_id, sim_metrics in metrics.items():
        for trader_type, type_metrics in sim_metrics.items():
            col_prefix = f'Type {trader_type}'
            for metric_name in metric_names:
                if metric_name in type_metrics:
                    df.loc[sim_id-1, (col_prefix, metric_name)] = type_metrics[metric_name]
    
    return df

def save_to_excel(df_trades, output_file='Rev_Sim_Output_Tables.xlsx'):
    """Save simulation results to Excel with three specific tabs"""
    try:
        # Calculate metrics for each tab
        session_metrics = calculate_session_metrics(df_trades)
        trader_metrics = calculate_trader_metrics(df_trades)
        revenue_metrics = calculate_revenue_summary(df_trades)
        
        # Define metric names for each tab
        session_metric_names = [
            'Total Traders', 'Total Sessions', 'Avg Trades per Session',
            'Avg Seconds Between Trades', 'First Trade Up %',
            'Subsequent Direction Match %', 'Avg Trade Amount', 'Total Revenue'
        ]
        
        trader_metric_names = [
            'Total Traders', 'Total Trades', 'Winning %',
            'Avg Trade Amount', 'Total Revenue'
        ]
        
        revenue_metric_names = ['Total Revenue']
        
        # Create formatted DataFrames
        session_df = create_formatted_dataframe(session_metrics, session_metric_names)
        trader_df = create_formatted_dataframe(trader_metrics, trader_metric_names)
        
        # Special handling for revenue summary (includes Atticus)
        revenue_columns = pd.MultiIndex.from_product([
            ['Type A', 'Type B', 'Type C', 'Type D', 'Atticus'],
            revenue_metric_names
        ])
        revenue_df = pd.DataFrame(index=range(len(revenue_metrics)), columns=revenue_columns)
        
        for sim_id, sim_metrics in revenue_metrics.items():
            for trader_type, type_metrics in sim_metrics.items():
                col_prefix = f'Type {trader_type}' if trader_type != 'Atticus' else 'Atticus'
                for metric_name in revenue_metric_names:
                    if metric_name in type_metrics:
                        revenue_df.loc[sim_id-1, (col_prefix, metric_name)] = type_metrics[metric_name]
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Tab 1: Session Metrics
            session_df.to_excel(writer, sheet_name="Session Metrics")
            
            # Tab 2: Trader Metrics
            trader_df.to_excel(writer, sheet_name="Trader Metrics")
            
            # Tab 3: Revenue Summary
            revenue_df.to_excel(writer, sheet_name="Revenue Summary")
        
        logging.info(f"Results saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving to Excel: {str(e)}")
        return False

def run_multiple_simulations(num_sims=10):
    """Run multiple simulations and collect results"""
    all_trades = []
    
    for sim_id in range(1, num_sims + 1):
        logging.info(f"Running simulation {sim_id}")
        sim = Simulation(sim_id)
        results = sim.run_simulation()
        
        if results:
            all_trades.append(results['trades'])
    
    # Combine results from all simulations
    combined_trades = pd.concat(all_trades, ignore_index=True)
    
    return combined_trades

def main():
    logging.info("Starting simulation batch")
    
    # Run simulations
    df_trades = run_multiple_simulations(num_sims=10)
    
    # Save results
    if df_trades is not None:
        save_to_excel(df_trades)
    else:
        logging.error("No results to save")
    
    logging.info("Simulation batch completed")

if __name__ == "__main__":
    main() 