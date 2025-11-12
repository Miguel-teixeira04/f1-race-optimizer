# F1 Race Optimizer 🏎️

Advanced car setup optimization tool with machine learning and iterative testing workflow.

## Features

- 🎯 Multi-stage optimization algorithm (exploration → exploitation → refinement)
- 🧠 Ensemble ML models (Random Forest + Gradient Boosting)
- 📊 Practice data integration with dynamic weighting
- 🔄 Iterative testing suggestions
- 🔐 User authentication system
- 📈 Fuel & tyre type controls for accuracy

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run race_optimizer_ui.py
```

## Default Login Credentials

- **Demo User**: username `demo`, password `demo123`
- **Admin User**: username `admin`, password `admin123`

## How to Use

1. **Login** with your credentials
2. **Select Track** from dropdown
3. **Upload Practice Data** (optional, but recommended)
4. **Enter Laps Remaining** for the practice session
5. **View Recommendations**:
   - More than 5 laps → Testing suggestions
   - 5 or fewer laps → Final optimal setup

## Data Files Required

- `simulator_data (2).csv` - Baseline simulation data
- `track_data.csv` - Track characteristics
- `practice_data.csv` - Your actual practice lap data

## Deployment to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repository
4. Make sure all CSV files are included!

## Algorithm

**Multi-Stage Optimization:**
1. **Stage 1**: Broad exploration (5000 samples)
2. **Stage 2**: Focused exploitation around top 10 candidates (5000 samples)
3. **Stage 3**: Lower Confidence Bound selection (uncertainty quantification)
4. **Stage 4**: Gradient-based local refinement

**Dynamic Weighting:**
- Practice data influence increases with more laps
- Maintains all simulator data, never discards information
- Balanced approach: 15%-45% practice influence depending on lap count

## Author

Created for Big Data course - Msc Business Analytics
Nova SBE - 2025
