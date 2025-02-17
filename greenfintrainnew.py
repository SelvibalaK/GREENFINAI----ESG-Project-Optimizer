import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

def reshape_data(data):
    # Melt the dataframe to convert years to rows
    melted_df = pd.melt(
        data,
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        var_name='Year',
        value_name='Value'
    )
    
    # Clean year column
    melted_df['Year'] = melted_df['Year'].str.extract('(\d{4})').astype(int)
    
    # Pivot to get features as columns
    reshaped_df = melted_df.pivot_table(
        index=['Country Name', 'Country Code', 'Year'],
        columns='Series Code',
        values='Value'
    ).reset_index()
    
    return reshaped_df

class ESGProjectPredictor:
    def __init__(self):
        self.projects_features = {
            "Renewable Energy Expansion": [
                "EG.ELC.RNEW.ZS",  # Renewable electricity output
                "EG.FEC.RNEW.ZS",  # Renewable energy consumption
                "EG.USE.PCAP.KG.OE",  # Energy use
                "EG.USE.COMM.FO.ZS",  # Fossil fuel consumption
                "EN.ATM.CO2E.PC"  # CO2 emissions
            ],
            "Smart Grid and Energy Efficiency": [
                "EG.ELC.ACCS.ZS",  # Access to electricity
                "EG.EGY.PRIM.PP.KD",  # Energy intensity
                "EG.USE.PCAP.KG.OE",  # Energy use
                "GE.EST",  # Government effectiveness
                "RQ.EST"  # Regulatory quality
            ],
            "Reforestation and Deforestation Control": [
                "AG.LND.FRST.ZS",  # Forest area
                "AG.LND.FRLS.HA",  # Tree cover loss
                "EN.CLC.GHGR.MT.CE"  # GHG emissions
            ],
            "Water Conservation and Management": [
                "ER.H2O.FWTL.ZS",  # Freshwater withdrawals
                "EN.H2O.BDYS.ZS"  # Water quality
            ],
            "Sustainable Agriculture": [
                "SN.ITK.DEFC.ZS",  # Undernourishment
                "EN.CLC.GHGR.MT.CE"  # GHG emissions
            ],
            "Circular Economy Initiative": [
                "EN.ATM.CO2E.PC",  # Used as proxy for waste efficiency
                "GB.XPD.RSDV.GD.ZS"  # R&D expenditure
            ],
            "Urban Green Space": [
                "AG.LND.FRST.ZS",  # Used as proxy for green space
                "EN.ATM.CO2E.PC"  # Air pollution proxy
            ],
            "Public Transport and E-Mobility": [
                "EG.USE.COMM.FO.ZS",  # Fossil fuel consumption
                "EN.ATM.CO2E.PC"  # Emissions
            ],
            "Climate Resilience": [
                "EN.CLC.SPEI.XD",  # Climate index
                "GE.EST"  # Government effectiveness
            ],
            "Sustainable Fishing": [
                "EN.H2O.BDYS.ZS",  # Water quality
                "ER.H2O.FWTL.ZS"  # Water withdrawals
            ],
            "Corporate ESG Investment": [
                "SE.XPD.TOTL.GB.ZS",  # Education expenditure
                "GB.XPD.RSDV.GD.ZS",  # R&D expenditure
                "RQ.EST"  # Regulatory quality
            ],
            "Environmental Education": [
                "SE.ADT.LITR.ZS",  # Literacy rate
                "VA.EST",  # Voice and accountability
                "GE.EST"  # Government effectiveness
            ],
            "Green Building": [
                "EG.EGY.PRIM.PP.KD",  # Energy intensity
                "EG.USE.COMM.FO.ZS"  # Fossil fuel consumption
            ],
            "Air Pollution Reduction": [
                "EN.ATM.CO2E.PC",  # CO2 emissions
                "EG.USE.COMM.FO.ZS"  # Fossil fuel consumption
            ],
            "ESG Governance": [
                "RL.EST",  # Rule of law
                "PV.EST",  # Political stability
                "CC.EST",  # Control of corruption
                "IC.LGL.CRED.XQ"  # Legal rights
            ]
        }
        
        self.models = {}
        self.scalers = {}
    
    def preprocess_data(self, data):
        # Reshape the data first
        reshaped_data = reshape_data(data)
        print("Available features after reshaping:", reshaped_data.columns.tolist())
        
        processed_data = {}
        for project, features in self.projects_features.items():
            available_features = [f for f in features if f in reshaped_data.columns]
            if not available_features:
                print(f"Warning: No features found for {project}")
                continue
                
            scaler = StandardScaler()
            X = reshaped_data[available_features]
            X_scaled = scaler.fit_transform(X)
            processed_data[project] = {
                'data': X_scaled,
                'years': reshaped_data['Year'],
                'countries': reshaped_data['Country Name']
            }
            self.scalers[project] = scaler
        return processed_data

    def train_models(self, data):
        processed_data = self.preprocess_data(data)
        
        for project, data_dict in processed_data.items():
            X = data_dict['data']
            
            # Generate synthetic targets based on actual data dimensions
            y_esg = np.random.uniform(60, 95, X.shape[0])
            y_risk = np.random.uniform(10, 40, X.shape[0])
            y_cost = np.random.uniform(500000, 5000000, X.shape[0])
            
            # Train models
            esg_model = RandomForestRegressor(n_estimators=100, random_state=42)
            risk_model = RandomForestRegressor(n_estimators=100, random_state=42)
            cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            esg_model.fit(X, y_esg)
            risk_model.fit(X, y_risk)
            cost_model.fit(X, y_cost)
            
            self.models[project] = {
                'esg': esg_model,
                'risk': risk_model,
                'cost': cost_model
            }
    
    def save_models(self, base_path='models/'):
        os.makedirs(base_path, exist_ok=True)
        
        # Save models
        with open(r'D:\User1\GREENFIN\greenfinproject_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        
        # Save scalers
        with open(r'D:\User1\GREENFIN\greenfinscalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature mappings
        with open(r'D:\User1\GREENFIN\greenfinfeature_mappings.pkl', 'wb') as f:
            pickle.dump(self.projects_features, f)

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv(r"D:\User1\ESG_REAL20.csv")
    
    # Initialize and train models
    predictor = ESGProjectPredictor()
    predictor.train_models(data)
    
    # Save all necessary files
    predictor.save_models()

