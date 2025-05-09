import pandas as pd
import numpy as np
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

class Simulation:
    def __init__(self, sim_id):
        self.sim_id = sim_id
        self.results = {}
        
    def run(self):
        """Run a single simulation"""
        try:
            # Read the template Excel file to get all columns
            template_file = 'Rev_Sim_Output_Tables.xlsx'
            if not os.path.exists(template_file):
                raise FileNotFoundError(f"Template file {template_file} not found")
            
            # Read all sheets from the template
            excel_file = pd.ExcelFile(template_file)
            sheet_names = excel_file.sheet_names
            
            # Initialize results dictionary with all columns from all sheets
            self.results = {'sim_id': self.sim_id, 'timestamp': datetime.now()}
            
            for sheet_name in sheet_names:
                df = pd.read_excel(template_file, sheet_name=sheet_name)
                for column in df.columns:
                    # Generate data for each column (replace with your actual simulation logic)
                    if column not in ['sim_id', 'timestamp']:
                        # Example: Generate random data based on column name
                        if 'price' in column.lower():
                            self.results[f"{sheet_name}_{column}"] = np.random.normal(100, 10)
                        elif 'volume' in column.lower():
                            self.results[f"{sheet_name}_{column}"] = np.random.normal(1000, 100)
                        else:
                            self.results[f"{sheet_name}_{column}"] = np.random.normal(50, 5)
            
            logging.info(f"Simulation {self.sim_id} completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error in simulation {self.sim_id}: {str(e)}")
            return False

def run_multiple_simulations(num_sims=10):
    """Run multiple simulations and collect results"""
    all_results = []
    
    for sim_id in range(1, num_sims + 1):
        sim = Simulation(sim_id)
        if sim.run():
            all_results.append(sim.results)
    
    return all_results

def save_to_excel(results, output_file='Rev_Sim_Output_Tables_Results.xlsx'):
    """Save simulation results to Excel with multiple sheets matching the template structure"""
    try:
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Read the template file to get sheet structure
        template_file = 'Rev_Sim_Output_Tables.xlsx'
        excel_file = pd.ExcelFile(template_file)
        sheet_names = excel_file.sheet_names
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save full results to first sheet
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Create sheets matching the template structure
            for sheet_name in sheet_names:
                # Get columns that belong to this sheet
                sheet_columns = [col for col in df.columns if col.startswith(f"{sheet_name}_")]
                if sheet_columns:
                    # Create a new DataFrame with just the relevant columns
                    sheet_df = df[['sim_id', 'timestamp'] + sheet_columns].copy()
                    # Rename columns to remove sheet name prefix
                    sheet_df.columns = ['sim_id', 'timestamp'] + [col.replace(f"{sheet_name}_", "") for col in sheet_columns]
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
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