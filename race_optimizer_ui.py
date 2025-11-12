"""
RACE OPTIMIZER - USER INTERFACE
================================
Interactive interface for car setup optimization with iterative testing suggestions
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import authentication
from auth import verify_user, create_user, initialize_default_users

# Page configuration
st.set_page_config(
    page_title="F1 Race Optimizer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #000000 0%, #E10600 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #E10600;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #E10600;
    }
    .suggestion-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .final-setup-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize authentication
initialize_default_users()

# ===== AUTHENTICATION SYSTEM =====
def login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🏎️ F1 RACE OPTIMIZER</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if verify_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.success("✓ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email (optional)")
                new_password = st.text_input("Choose Password", type="password")
                new_password2 = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("Register", use_container_width=True)
                
                if register:
                    if not new_username or not new_password:
                        st.error("Username and password are required")
                    elif new_password != new_password2:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = create_user(new_username, new_password, new_email)
                        if success:
                            st.success(f"✓ {message}! You can now login.")
                        else:
                            st.error(f"❌ {message}")
        
        st.markdown("---")
        st.info("**Default accounts:**\n\n🔹 Username: `demo` | Password: `demo123`\n\n🔹 Username: `admin` | Password: `admin123`")

# Check authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_page()
    st.stop()

# ===== MAIN APP (after login) =====

# Title
st.markdown('<div class="main-header">🏎️ F1 RACE OPTIMIZER</div>', unsafe_allow_html=True)

# Sidebar: User info
with st.sidebar:
    st.markdown(f"### 👤 Welcome, **{st.session_state['username']}**!")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.rerun()
    st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load all necessary data files"""
    simulator_data = pd.read_csv('simulator_data (2).csv')
    practice_data = pd.read_csv('practice_data.csv')
    track_data = pd.read_csv('track_data.csv')
    return simulator_data, practice_data, track_data

try:
    simulator_data, practice_data, track_data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Setup parameters (what we want to optimize)
# NOTE: Practice data uses 'Brake' while simulator uses 'Brake Balance'
SETUP_PARAMS = ['Rear Wing', 'Engine', 'Front Wing', 'Brake Balance', 'Differential', 'Suspension']
SETUP_PARAMS_PRACTICE = ['Rear Wing', 'Engine', 'Front Wing', 'Brake', 'Differential', 'Suspension']

# Track features
TRACK_FEATURES = ['Cornering', 'Inclines', 'Camber', 'Grip', 'Wind (Avg. Speed)', 
                  'Temperature', 'Humidity', 'Air Density', 'Air Pressure', 
                  'Wind (Gusts)', 'Altitude', 'Roughness', 'Width', 'Lap Distance']

# Control variables (affect lap time but NOT part of setup optimization)
# These need to be included in training to avoid confounding the setup effects
CONTROL_FEATURES = ['Fuel']  # Fuel affects lap time (less fuel = lighter = faster)

# Tyre type (categorical) - needs encoding
TYRE_FEATURE = 'Tyre Choice'  # Extra Soft, Soft, Medium, Hard

# Model training function
@st.cache_resource
def train_models(X_train, y_train, _use_fuel_tyre_controls=False):
    """Train ensemble models for setup optimization
    
    Args:
        X_train: Training features
        y_train: Training target (lap times)
        _use_fuel_tyre_controls: Whether to include Fuel/Tyre as control variables
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=20, 
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models, scaler, _use_fuel_tyre_controls

def predict_lap_time(models, scaler, track_features, setup_values, use_controls=False, fuel_tyre_values=None):
    """Predict lap time using ensemble of models
    
    Args:
        models: Trained models
        scaler: Feature scaler
        track_features: Track characteristic values
        setup_values: Setup parameter values
        use_controls: Whether model was trained with fuel/tyre controls
        fuel_tyre_values: [fuel, tyre_remaining] if using controls
    """
    if use_controls and fuel_tyre_values is not None:
        features = track_features + setup_values + fuel_tyre_values
    else:
        features = track_features + setup_values
    
    features_scaled = scaler.transform([features])
    
    predictions = []
    for model in models.values():
        predictions.append(model.predict(features_scaled)[0])
    
    return np.mean(predictions)

def optimize_setup(models, scaler, track_features, sample_size=10000, use_controls=False, fuel_tyre_values=None, practice_setups=None):
    """Find optimal setup configuration - ADVANCED MULTI-STAGE OPTIMIZATION
    
    Strategy:
    1. Stage 1: Grid-based exploration (broad search)
    2. Stage 2: Exploitation around best practice data (local search)
    3. Stage 3: Gradient-guided refinement (fine-tuning)
    4. Stage 4: Ensemble agreement verification (confidence check)
    
    Args:
        models: Trained models
        scaler: Feature scaler
        track_features: Track characteristic values
        sample_size: Number of configurations to test
        use_controls: Whether model uses fuel/tyre controls
        fuel_tyre_values: [fuel, tyre_remaining] for standardized conditions
        practice_setups: DataFrame with actual practice setups (to focus search nearby)
    """
    best_lap_time = float('inf')
    best_setup = None
    
    np.random.seed(42)
    
    # Helper function for batch prediction with uncertainty
    def predict_with_uncertainty(candidates):
        """Predict lap times and return mean + std (uncertainty)"""
        if len(candidates.shape) == 1:
            candidates = candidates.reshape(1, -1)
        
        track_matrix = np.tile(track_features, (len(candidates), 1))
        
        if use_controls and fuel_tyre_values is not None:
            fuel_tyre_matrix = np.tile(fuel_tyre_values, (len(candidates), 1))
            full_features = np.hstack([track_matrix, candidates, fuel_tyre_matrix])
        else:
            full_features = np.hstack([track_matrix, candidates])
        
        full_features_scaled = scaler.transform(full_features)
        
        # Get predictions from all models
        all_predictions = np.array([model.predict(full_features_scaled) for model in models.values()])
        
        # Mean prediction and uncertainty (std between models)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return mean_pred, std_pred
    
    np.random.seed(42)
    
    # ==== STAGE 1: BROAD EXPLORATION ====
    # Get reasonable ranges from simulator data
    setup_distributions = {}
    for param in SETUP_PARAMS:
        setup_distributions[param] = simulator_data[param].values
    
    stage1_candidates = []
    stage1_size = sample_size // 2  # 50% for exploration
    
    # If we have practice data, use it intelligently!
    if practice_setups is not None and len(practice_setups) > 0:
        practice_laps = len(practice_setups)
        
        # Dynamic focus based on amount of practice data
        if practice_laps < 20:
            # Few practices: 40% focused, 60% exploration
            focused_pct = 0.4
            variation_range = 50  # Wider exploration
            top_pct = 0.2  # Top 20%
        elif practice_laps < 60:
            # Medium practices: 60% focused, 40% exploration
            focused_pct = 0.6
            variation_range = 30  # Medium exploration
            top_pct = 0.15  # Top 15%
        else:
            # Lots of practices: 80% focused, 20% exploration
            focused_pct = 0.8
            variation_range = 20  # Tight refinement
            top_pct = 0.1  # Top 10%
        
        # Find best practice setups (top performers)
        best_practice = practice_setups.nsmallest(max(1, int(len(practice_setups) * top_pct)), 'Lap Time')
        
        # Stage 1a: Focused search around best practice setups
        focused_samples = int(stage1_size * focused_pct)
        for _ in range(focused_samples):
            base_setup = best_practice[SETUP_PARAMS_PRACTICE].sample(1).iloc[0]
            candidate = []
            for param_name in SETUP_PARAMS:
                prac_name = 'Brake' if param_name == 'Brake Balance' else param_name
                base_value = base_setup[prac_name]
                variation = np.random.randint(-variation_range, variation_range + 1)
                value = np.clip(base_value + variation, 1, 500)
                candidate.append(value)
            stage1_candidates.append(candidate)
        
        # Stage 1b: Broader exploration
        for _ in range(stage1_size - focused_samples):
            candidate = []
            for param in SETUP_PARAMS:
                if np.random.random() < 0.7:
                    value = np.random.choice(setup_distributions[param])
                else:
                    value = np.random.randint(1, 501)
                candidate.append(value)
            stage1_candidates.append(candidate)
    else:
        # No practice data: pure exploration
        for _ in range(stage1_size):
            candidate = []
            for param in SETUP_PARAMS:
                if np.random.random() < 0.7:
                    value = np.random.choice(setup_distributions[param])
                else:
                    value = np.random.randint(1, 501)
                candidate.append(value)
            stage1_candidates.append(candidate)
    
    # Evaluate Stage 1 candidates
    stage1_candidates = np.array(stage1_candidates)
    stage1_preds, stage1_uncertainty = predict_with_uncertainty(stage1_candidates)
    
    # ==== STAGE 2: EXPLOITATION OF BEST CANDIDATES ====
    # Take top 10 candidates from Stage 1
    top_k = min(10, len(stage1_candidates))
    top_indices = np.argsort(stage1_preds)[:top_k]
    top_candidates = stage1_candidates[top_indices]
    
    # Generate variations around each top candidate
    stage2_candidates = []
    variations_per_candidate = (sample_size - stage1_size) // top_k
    
    for top_setup in top_candidates:
        # Create multiple variations with increasingly fine granularity
        for _ in range(variations_per_candidate):
            candidate = []
            for i, param_value in enumerate(top_setup):
                # Adaptive variation: smaller changes for high-confidence predictions
                base_range = 15
                variation = np.random.randint(-base_range, base_range + 1)
                value = np.clip(param_value + variation, 1, 500)
                candidate.append(value)
            stage2_candidates.append(candidate)
    
    # Evaluate Stage 2 candidates
    if stage2_candidates:
        stage2_candidates = np.array(stage2_candidates)
        stage2_preds, stage2_uncertainty = predict_with_uncertainty(stage2_candidates)
        
        # Combine all candidates
        all_candidates = np.vstack([stage1_candidates, stage2_candidates])
        all_preds = np.concatenate([stage1_preds, stage2_preds])
        all_uncertainty = np.concatenate([stage1_uncertainty, stage2_uncertainty])
    else:
        all_candidates = stage1_candidates
        all_preds = stage1_preds
        all_uncertainty = stage1_uncertainty
    
    # ==== STAGE 3: FINAL SELECTION WITH UNCERTAINTY ====
    # Penalize high uncertainty (exploration-exploitation balance)
    # Lower Confidence Bound: mean - 0.5 * std
    lcb_scores = all_preds - 0.5 * all_uncertainty
    
    # Find best by LCB (favors low predicted time + low uncertainty)
    best_idx = np.argmin(lcb_scores)
    best_setup = all_candidates[best_idx].tolist()
    best_lap_time = all_preds[best_idx]
    
    # ==== STAGE 4: GRADIENT-BASED REFINEMENT ====
    # Fine-tune the best setup using local search
    refined_setup = best_setup.copy()
    refined_time = best_lap_time
    
    # Try small adjustments (±1 to ±5) on each parameter
    improvement_found = True
    iterations = 0
    max_iterations = 3
    
    while improvement_found and iterations < max_iterations:
        improvement_found = False
        iterations += 1
        
        for param_idx in range(len(SETUP_PARAMS)):
            # Try small variations
            for delta in [-5, -3, -1, 1, 3, 5]:
                test_setup = refined_setup.copy()
                test_setup[param_idx] = np.clip(refined_setup[param_idx] + delta, 1, 500)
                
                test_pred, _ = predict_with_uncertainty(np.array([test_setup]))
                
                if test_pred[0] < refined_time:
                    refined_setup = test_setup
                    refined_time = test_pred[0]
                    improvement_found = True
                    break
    
    return refined_setup, refined_time

def generate_test_suggestions(current_setup, feature_importance, num_suggestions=3):
    """Generate experimental setup variations to test"""
    suggestions = []
    
    # Get top important parameters
    important_params = feature_importance.head(6)['Feature'].values
    important_params = [p for p in important_params if p in SETUP_PARAMS][:3]
    
    for i, param in enumerate(important_params[:num_suggestions]):
        param_idx = SETUP_PARAMS.index(param)
        current_value = current_setup[param_idx]
        
        # Generate variation
        if i == 0:
            # Increase most important parameter
            new_value = min(500, current_value + 20)
            direction = "increase"
        elif i == 1:
            # Decrease second most important
            new_value = max(1, current_value - 20)
            direction = "decrease"
        else:
            # Try significant change on third
            if current_value > 250:
                new_value = max(1, current_value - 50)
                direction = "significantly decrease"
            else:
                new_value = min(500, current_value + 50)
                direction = "significantly increase"
        
        test_setup = current_setup.copy()
        test_setup[param_idx] = new_value
        
        suggestions.append({
            'parameter': param,
            'current_value': current_value,
            'suggested_value': new_value,
            'direction': direction,
            'test_setup': test_setup
        })
    
    return suggestions

# Sidebar - Configuration
st.sidebar.markdown("## 🔧 Configuration")

# Track selection
available_tracks = track_data['Track'].unique()
selected_track = st.sidebar.selectbox(
    "Select Track",
    available_tracks,
    help="Choose the circuit for optimization"
)

# Get track conditions
track_conditions = track_data[track_data['Track'] == selected_track].iloc[0]

# Laps remaining input - ALWAYS 120 practice laps available
PRACTICE_LAPS_AVAILABLE = 120
laps_remaining = st.sidebar.number_input(
    "Practice Laps Remaining",
    min_value=1,
    max_value=PRACTICE_LAPS_AVAILABLE,
    value=PRACTICE_LAPS_AVAILABLE,
    help="Number of practice laps remaining (always 120 total)"
)

# File upload section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Import Data Files")

uploaded_track = st.sidebar.file_uploader("Upload track_data.csv (optional)", type=['csv'], key='track')
uploaded_practice = st.sidebar.file_uploader("Upload practice_data.csv (optional)", type=['csv'], key='practice')

if uploaded_track is not None:
    track_data = pd.read_csv(uploaded_track)
    st.sidebar.success("✓ Track data updated!")
    
if uploaded_practice is not None:
    practice_data = pd.read_csv(uploaded_practice)
    st.sidebar.success("✓ Practice data updated!")

# Display track info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Track Information")
total_laps = int(track_conditions['N. Laps'])
st.sidebar.markdown(f"""
- **Distance:** {track_conditions['Lap Distance']:.2f} km
- **Race Laps:** {total_laps}
- **Practice Laps:** 120
- **Cornering:** {track_conditions['Cornering']}
- **Grip:** {track_conditions['Grip']}
- **Temperature:** {track_conditions['Temperature']}°C
""")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f'<div class="sub-header">🏁 {selected_track} - Optimization Strategy</div>', 
                unsafe_allow_html=True)
    
    # Training models
    with st.spinner("Training AI models..."):
        # Check if we have practice data with Fuel/Tyre info
        track_practice = practice_data[practice_data['Track'] == selected_track]
        has_fuel_tyre = (len(track_practice) > 0 and 
                        'Fuel' in track_practice.columns and 
                        'Tyre Remaining' in track_practice.columns)
        
        if has_fuel_tyre:
            # Use BOTH simulator + practice data with fuel/tyre controls
            st.info("🎯 Using practice data with Fuel & Tyre Type controls for better accuracy")
            
            # Simulator data (no fuel/tyre)
            X_sim = simulator_data[TRACK_FEATURES + SETUP_PARAMS].copy()
            y_sim = simulator_data['Lap Time'].copy()
            
            # Add average fuel to simulator data (to standardize)
            avg_fuel = track_practice['Fuel'].mean()
            X_sim['Fuel'] = avg_fuel
            
            # Add most common tyre type (mode) to simulator data
            # One-hot encode tyre types
            most_common_tyre = track_practice[TYRE_FEATURE].mode()[0] if len(track_practice) > 0 else 'Soft'
            
            # Practice data (has fuel/tyre)
            # Note: Practice data uses 'Brake' instead of 'Brake Balance'
            prac_features = TRACK_FEATURES + SETUP_PARAMS_PRACTICE + CONTROL_FEATURES + [TYRE_FEATURE]
            X_prac = track_practice[prac_features].copy()
            # Rename to match simulator data
            X_prac = X_prac.rename(columns={'Brake': 'Brake Balance'})
            y_prac = track_practice['Lap Time'].copy()
            
            # One-hot encode tyre type for practice data
            X_prac_encoded = pd.get_dummies(X_prac, columns=[TYRE_FEATURE], prefix='Tyre')
            
            # Add same tyre encoding to simulator data (all same type)
            for col in X_prac_encoded.columns:
                if col.startswith('Tyre_'):
                    if col not in X_sim.columns:
                        tyre_type = col.replace('Tyre_', '')
                        X_sim[col] = 1 if tyre_type == most_common_tyre else 0
            
            # Ensure same column order
            common_cols = [col for col in X_prac_encoded.columns if col in X_sim.columns or not col.startswith('Tyre_')]
            for col in X_prac_encoded.columns:
                if col.startswith('Tyre_') and col not in X_sim.columns:
                    X_sim[col] = 0
            
            # Reorder columns to match
            X_sim = X_sim[X_prac_encoded.columns]
            
            # Combine datasets with DYNAMIC WEIGHTING
            # More practice data = higher weight (but KEEP all simulator data!)
            practice_laps = len(track_practice)
            
            # Dynamic weight calculation - BALANCED approach
            # Keep ALL simulator data, moderate practice weight to avoid overfitting
            sim_data_count = len(X_sim)  # ~10,000 samples
            
            if practice_laps < 20:
                # Few practices: light weight ~15% influence
                weight_multiplier = max(5, int(sim_data_count * 0.15 / practice_laps))
            elif practice_laps < 40:
                # Growing practices: ~25% influence
                weight_multiplier = int(sim_data_count * 0.30 / practice_laps)
            elif practice_laps < 60:
                # Medium practices: ~30% influence
                weight_multiplier = int(sim_data_count * 0.40 / practice_laps)
            elif practice_laps < 80:
                # Many practices: ~35% influence
                weight_multiplier = int(sim_data_count * 0.50 / practice_laps)
            elif practice_laps < 100:
                # Lots of practices: ~40% influence
                weight_multiplier = int(sim_data_count * 0.60 / practice_laps)
            else:
                # Full practices: ~45% influence (never more than 50%)
                weight_multiplier = int(sim_data_count * 0.80 / practice_laps)
            
            # Weight practice data by duplicating it
            X_prac_weighted = pd.concat([X_prac_encoded] * weight_multiplier, ignore_index=True)
            y_prac_weighted = pd.concat([y_prac] * weight_multiplier, ignore_index=True)
            
            # Combine: ALL simulator + WEIGHTED practice
            X_combined = pd.concat([X_sim, X_prac_weighted], ignore_index=True)
            y_combined = pd.concat([y_sim, y_prac_weighted], ignore_index=True)
            
            # Remove outliers
            q1 = y_combined.quantile(0.25)
            q3 = y_combined.quantile(0.75)
            iqr = q3 - q1
            extreme_outliers = (y_combined < (q1 - 3 * iqr)) | (y_combined > (q3 + 3 * iqr))
            X_combined = X_combined[~extreme_outliers]
            y_combined = y_combined[~extreme_outliers]
            
            # Train models WITH fuel/tyre controls
            models, scaler, use_controls = train_models(X_combined, y_combined, _use_fuel_tyre_controls=True)
            
            # Use median/mode for standardized predictions
            standard_fuel = track_practice['Fuel'].median()
            standard_tyre = most_common_tyre
            
            # Create tyre encoding for predictions
            tyre_cols = [col for col in X_combined.columns if col.startswith('Tyre_')]
            tyre_encoding = [1 if f'Tyre_{standard_tyre}' == col else 0 for col in tyre_cols]
            
            fuel_tyre_values = [standard_fuel] + tyre_encoding
            
            practice_count = len(track_practice)
            sim_count = len(X_sim)
            total_count = len(X_combined)
            practice_weight_pct = (practice_count * weight_multiplier / total_count) * 100
            
            st.success(f"✓ Model trained with {total_count} samples (controlling for Fuel & Tyre Type)")
            st.caption(f"📊 Standardized conditions: Fuel={standard_fuel:.1f}kg, Tyre={standard_tyre}")
            st.caption(f"💾 Data composition: ALL {sim_count} simulator samples + {practice_count} practice laps (×{weight_multiplier} weight)")
            st.caption(f"⚖️ Effective influence: Practice = **{practice_weight_pct:.1f}%**, Simulator = {100-practice_weight_pct:.1f}%")
            
        else:
            # Use only simulator data (original approach)
            st.info("📊 Using simulator data only (no practice data available)")
            
            X_sim = simulator_data[TRACK_FEATURES + SETUP_PARAMS].copy()
            y_sim = simulator_data['Lap Time'].copy()
            
            # Remove outliers
            q1 = y_sim.quantile(0.25)
            q3 = y_sim.quantile(0.75)
            iqr = q3 - q1
            extreme_outliers = (y_sim < (q1 - 3 * iqr)) | (y_sim > (q3 + 3 * iqr))
            X_sim = X_sim[~extreme_outliers]
            y_sim = y_sim[~extreme_outliers]
            
            # Train models WITHOUT fuel/tyre controls
            models, scaler, use_controls = train_models(X_sim, y_sim, _use_fuel_tyre_controls=False)
            fuel_tyre_values = None
            
            st.success(f"✓ Model trained with {len(X_sim)} samples")
        
        # Get feature importance
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X_sim.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    # Extract track features
    track_features_values = [
        track_conditions['Cornering'],
        track_conditions['Inclines'],
        track_conditions['Camber'],
        track_conditions['Grip'],
        track_conditions['Wind (Avg. Speed)'],
        track_conditions['Temperature'],
        track_conditions['Humidity'],
        track_conditions['Air Density'],
        track_conditions['Air Pressure'],
        track_conditions['Wind (Gusts)'],
        track_conditions['Altitude'],
        track_conditions['Roughness'],
        track_conditions['Width'],
        track_conditions['Lap Distance']
    ]
    
    # Show practice data context if available
    best_practice_lap = None
    if has_fuel_tyre and len(track_practice) > 0:
        best_practice_lap = track_practice['Lap Time'].min()
        avg_practice_lap = track_practice['Lap Time'].mean()
        st.info(f"🏁 Your practice data: Best lap = **{best_practice_lap:.3f}s**, Average = **{avg_practice_lap:.3f}s** ({len(track_practice)} laps)")
    
    # Optimize setup
    with st.spinner("Optimizing car setup... (analyzing 10000 configurations for maximum accuracy)"):
        # Pass practice data if available to focus search
        practice_for_optimization = track_practice if has_fuel_tyre else None
        
        optimal_setup, predicted_time = optimize_setup(
            models, scaler, track_features_values, 
            use_controls=use_controls, 
            fuel_tyre_values=fuel_tyre_values,
            practice_setups=practice_for_optimization
        )
        
        # Show improvement potential
        if best_practice_lap is not None:
            improvement = best_practice_lap - predicted_time
            if improvement > 0:
                st.success(f"🎯 Predicted improvement: **{improvement:.3f}s faster** than your best lap!")
            elif improvement < -0.5:
                st.warning(f"⚠️ Model suggests this setup may be {abs(improvement):.3f}s slower - consider testing carefully")
            else:
                st.info(f"📊 Predicted time very close to your best lap ({abs(improvement):.3f}s difference)")
    
    # Decision based on laps remaining
    if laps_remaining > 5:
        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🔬 Testing Phase ({laps_remaining} laps remaining)
        
        You have enough time to experiment and refine the setup. 
        Here's the **current optimal baseline** and **suggested tests**:
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display optimal baseline
        st.markdown("#### 📍 Current Optimal Baseline")
        setup_df = pd.DataFrame({
            'Parameter': SETUP_PARAMS,
            'Optimal Value': [int(v) for v in optimal_setup]
        })
        
        col_a, col_b, col_c = st.columns(3)
        for idx, row in setup_df.iterrows():
            if idx % 3 == 0:
                col_a.metric(row['Parameter'], row['Optimal Value'])
            elif idx % 3 == 1:
                col_b.metric(row['Parameter'], row['Optimal Value'])
            else:
                col_c.metric(row['Parameter'], row['Optimal Value'])
        
        st.info(f"⏱️ **Predicted Lap Time:** {predicted_time:.3f} seconds")
        
        # Generate test suggestions
        st.markdown("#### 🧪 Suggested Experimental Variations")
        st.markdown("""
        Test these variations to find improvements. After each test, add the results 
        to the practice data and re-run the optimizer.
        """)
        
        suggestions = generate_test_suggestions(optimal_setup, feature_importance)
        
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Test #{i}: {suggestion['direction']} {suggestion['parameter']}", expanded=True):
                col_test1, col_test2, col_test3 = st.columns([1, 1, 2])
                
                with col_test1:
                    st.markdown("**Current Value**")
                    st.markdown(f"### {suggestion['current_value']}")
                
                with col_test2:
                    st.markdown("**Suggested Value**")
                    st.markdown(f"### {suggestion['suggested_value']}")
                
                with col_test3:
                    st.markdown("**Full Test Setup:**")
                    test_setup_df = pd.DataFrame({
                        'Parameter': SETUP_PARAMS,
                        'Value': [int(v) for v in suggestion['test_setup']]
                    })
                    st.dataframe(test_setup_df, hide_index=True, use_container_width=True)
                
                # Predict test setup lap time
                test_lap_time = predict_lap_time(
                    models, scaler, track_features_values, suggestion['test_setup'],
                    use_controls=use_controls, fuel_tyre_values=fuel_tyre_values
                )
                
                delta = test_lap_time - predicted_time
                if delta < 0:
                    st.success(f"💡 Predicted improvement: **{abs(delta):.3f}s faster** (Est. lap time: {test_lap_time:.3f}s)")
                else:
                    st.warning(f"⚠️ Predicted: **{delta:.3f}s slower** (Est. lap time: {test_lap_time:.3f}s)")
        
        # Instructions for updating
        st.markdown("---")
        st.markdown("""
        ### 📝 After Testing
        
        1. **Test one or more suggested configurations** during practice
        2. **Record the actual lap times** achieved
        3. **Add the results** to the practice data CSV file
        4. **Re-run this optimizer** to get refined recommendations
        5. **Repeat** until you have ≤5 laps remaining or are satisfied with the setup
        """)
        
    else:
        # Final setup recommendation (≤5 laps)
        st.markdown('<div class="final-setup-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🏆 Final Race Setup ({laps_remaining} laps remaining)
        
        Time to commit! Here's your **optimized race setup** based on all available data:
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display final setup prominently
        st.markdown("#### 🎯 Optimal Configuration")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        for idx, (param, value) in enumerate(zip(SETUP_PARAMS, optimal_setup)):
            value_int = int(value)
            if idx % 3 == 0:
                col_f1.metric(
                    label=param,
                    value=value_int,
                    help=f"Optimal value for {param}"
                )
            elif idx % 3 == 1:
                col_f2.metric(
                    label=param,
                    value=value_int,
                    help=f"Optimal value for {param}"
                )
            else:
                col_f3.metric(
                    label=param,
                    value=value_int,
                    help=f"Optimal value for {param}"
                )
        
        st.success(f"🏁 **Expected Lap Time:** {predicted_time:.3f} seconds")
        
        # Strategy advice
        st.markdown("#### 💡 Race Strategy")
        
        # Analyze setup characteristics
        avg_setup = np.mean(optimal_setup)
        aggressive_params = sum(1 for v in optimal_setup if v > 300)
        conservative_params = sum(1 for v in optimal_setup if v < 200)
        
        if aggressive_params >= 3:
            strategy = "**Aggressive Setup** - High performance, requires smooth driving"
        elif conservative_params >= 3:
            strategy = "**Conservative Setup** - Balanced and consistent"
        else:
            strategy = "**Balanced Setup** - Good mix of speed and stability"
        
        st.info(strategy)
        
        # Export setup
        if st.button("📥 Export Setup Configuration"):
            export_df = pd.DataFrame({
                'Parameter': SETUP_PARAMS,
                'Value': [int(v) for v in optimal_setup]
            })
            export_df['Track'] = selected_track
            export_df['Predicted_Lap_Time'] = predicted_time
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"optimal_setup_{selected_track.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

with col2:
    st.markdown('<div class="sub-header">📈 Analytics</div>', unsafe_allow_html=True)
    
    # Feature importance for setup parameters
    st.markdown("#### 🔑 Most Important Setup Parameters")
    setup_importance = feature_importance[feature_importance['Feature'].isin(SETUP_PARAMS)].head(6)
    
    for _, row in setup_importance.iterrows():
        importance_pct = row['Importance'] * 100
        st.markdown(f"**{row['Feature']}**")
        st.progress(min(1.0, importance_pct / 10))
        st.caption(f"{importance_pct:.1f}% importance")
    
    # Practice data analysis
    st.markdown("#### 📊 Practice Session Data")
    track_practice = practice_data[practice_data['Track'] == selected_track]
    
    if len(track_practice) > 0:
        st.metric("Practice Laps Recorded", len(track_practice))
        st.metric("Best Practice Lap", f"{track_practice['Lap Time'].min():.3f}s")
        st.metric("Average Practice Lap", f"{track_practice['Lap Time'].mean():.3f}s")
        
        if st.checkbox("Show Practice Data"):
            st.dataframe(
                track_practice[['Round', 'Stint', 'Lap', 'Lap Time'] + SETUP_PARAMS].head(10),
                hide_index=True
            )
    else:
        st.warning(f"No practice data available for {selected_track}")
    
    # Model performance
    st.markdown("#### 🤖 Model Confidence")
    
    # Cross-validation scores (simplified display)
    from sklearn.model_selection import cross_val_score
    
    with st.spinner("Calculating..."):
        rf_model = models['Random Forest']
        cv_scores = cross_val_score(
            rf_model, 
            X_sim[TRACK_FEATURES + SETUP_PARAMS], 
            y_sim, 
            cv=3,
            scoring='r2'
        )
        
        avg_r2 = cv_scores.mean()
        
        st.metric("Model Accuracy (R²)", f"{avg_r2:.3f}")
        
        if avg_r2 > 0.8:
            st.success("✅ High confidence")
        elif avg_r2 > 0.6:
            st.info("✓ Good confidence")
        else:
            st.warning("⚠️ Moderate confidence")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏎️ F1 Race Optimizer | Built with Streamlit & Machine Learning</p>
    <p style='font-size: 0.8rem;'>Optimizes car setup based on simulator, practice, and track data</p>
</div>
""", unsafe_allow_html=True)
