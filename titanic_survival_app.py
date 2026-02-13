import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #013220;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #0b3b5c 0%, #1b5e7a 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-bottom: 5px solid #ffd700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0b3b5c;
        padding: 0.8rem;
        border-bottom: 3px solid #0b3b5c;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #0b3b5c 0%, #1b5e7a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .info-box {
        background-color: black;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: black;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0b3b5c 0%, #1b5e7a 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(11, 59, 92, 0.4);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #0b3b5c 0%, #1b5e7a 100%);
        color: white;
        border-radius: 15px;
        margin-top: 3rem;
    }
    .survived {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }
    .not-survived {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CACHING
# ============================================
@st.cache_data
def load_data():
    """Load Titanic dataset"""
    try:
        # Try to load from seaborn first
        df = sns.load_dataset('titanic')
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from seaborn: {e}. Using built-in sample data.")
        
        # Create sample Titanic data
        data = {
            'survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
                         0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3,
                       1, 2, 3, 1, 3, 2, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 2],
            'sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 
                    'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 
                    'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female',
                    'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female',
                    'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            'age': [22, 38, 26, 35, 35, 28, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 31, 45, 26, 18,
                    42, 25, 30, 50, 23, 33, 41, 29, 37, 44, 48, 21, 32, 36, 52, 47, 24, 34, 49, 19],
            'sibsp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                      1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            'parch': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07, 16.70, 26.55,
                     8.05, 31.28, 7.85, 16.00, 7.75, 21.00, 8.05, 12.35, 25.00, 13.50, 7.25, 52.00,
                     8.50, 27.50, 42.40, 9.75, 18.50, 7.75, 35.00, 8.05, 22.00, 15.50, 45.00, 11.25,
                     19.00, 13.00, 32.00, 8.50],
            'embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S', 'S', 'S', 'S',
                         'S', 'S', 'S', 'S', 'C', 'S', 'Q', 'S', 'S', 'C', 'S', 'Q', 'S', 'S', 'C', 'S',
                         'Q', 'S', 'C', 'S', 'Q', 'S', 'C', 'S']
        }
        df = pd.DataFrame(data)
        return df

@st.cache_data
def load_test_data():
    """Load test data from gender_submission file"""
    try:
        test_df = pd.read_csv('gender_submission (1).csv')
        return test_df
    except:
        return None

# ============================================
# DATA CLEANING FUNCTIONS
# ============================================
def clean_data(df):
    """Clean missing data and prepare features"""
    df_clean = df.copy()
    
    # Task 1: Clean missing data
    if 'age' in df_clean.columns:
        df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    
    if 'embarked' in df_clean.columns:
        df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0] if not df_clean['embarked'].mode().empty else 'S')
    
    if 'fare' in df_clean.columns:
        df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    return df_clean

def encode_features(df):
    """Encode categorical variables"""
    df_encoded = df.copy()
    
    # Task 2: Encode categorical variables
    if 'sex' in df_encoded.columns:
        df_encoded['sex_encoded'] = df_encoded['sex'].map({'male': 1, 'female': 0})
    
    if 'embarked' in df_encoded.columns:
        embarked_map = {val: idx for idx, val in enumerate(df_encoded['embarked'].unique())}
        df_encoded['embarked_encoded'] = df_encoded['embarked'].map(embarked_map)
    
    if 'sibsp' in df_encoded.columns and 'parch' in df_encoded.columns:
        df_encoded['family_size'] = df_encoded['sibsp'] + df_encoded['parch'] + 1
    
    return df_encoded

def prepare_features(df):
    """Prepare feature matrix for modeling"""
    # Select features
    feature_cols = []
    
    if 'pclass' in df.columns:
        feature_cols.append('pclass')
    
    if 'sex_encoded' in df.columns:
        feature_cols.append('sex_encoded')
    
    if 'age' in df.columns:
        feature_cols.append('age')
    
    if 'fare' in df.columns:
        feature_cols.append('fare')
    
    if 'family_size' in df.columns:
        feature_cols.append('family_size')
    
    if 'embarked_encoded' in df.columns:
        feature_cols.append('embarked_encoded')
    
    X = df[feature_cols].copy()
    y = df['survived'] if 'survived' in df.columns else None
    
    return X, y, feature_cols

# ============================================
# MODEL TRAINING FUNCTIONS
# ============================================
def train_models(X_train, y_train):
    """Train multiple classification models"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except:
            results[name] = {
                'model': model,
                'cv_mean': 0,
                'cv_std': 0
            }
    
    return trained_models, results

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test data"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'predictions': y_pred
        }
        
        # Calculate ROC-AUC if possible
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['y_pred_proba'] = y_pred_proba
            except:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5
        
        results[name] = metrics
    
    return results

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'coef_'):
        # For Logistic Regression
        importance = np.abs(model.coef_[0])
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return imp_df
    
    elif hasattr(model, 'feature_importances_'):
        # For Tree-based models
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return imp_df
    
    return None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/null/titanic.png", width=100)
    st.title("🚢 Titanic Survival")
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["📊 Data Overview", 
         "🔍 Data Cleaning", 
         "📈 Exploratory Analysis",
         "🤖 Model Training",
         "🎯 Prediction",
         "📋 Feature Importance"]
    )
    
    st.markdown("---")
    
    # Model parameters
    if page in ["🤖 Model Training", "🎯 Prediction", "📋 Feature Importance"]:
        st.subheader("⚙️ Model Parameters")
        
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        ) / 100
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=100,
            value=42,
            step=1
        )
    
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.info("""
    **🚢 Titanic Survival Prediction**
    
    This app predicts passenger survival using machine learning.
    
    **Features:**
    - 📊 Data cleaning & preprocessing
    - 🔍 Exploratory data analysis
    - 🤖 Multiple ML models
    - 🎯 Individual predictions
    - 📋 Feature importance analysis
    """)

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<div class="main-header">🚢 Titanic Survival Prediction</div>', unsafe_allow_html=True)

# Load data
df = load_data()
test_df = load_test_data()

# ============================================
# PAGE 1: DATA OVERVIEW
# ============================================
if page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Passengers", f"{df.shape[0]}")
    
    with col2:
        survived = df['survived'].sum() if 'survived' in df.columns else 0
        st.metric("Survived", f"{survived}")
    
    with col3:
        survival_rate = (survived / df.shape[0] * 100) if df.shape[0] > 0 else 0
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    
    with col4:
        missing_age = df['age'].isnull().sum() if 'age' in df.columns else 0
        st.metric("Missing Age", f"{missing_age}")
    
    with col5:
        num_features = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", f"{num_features}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Sample")
        # Convert to string to avoid Arrow issues
        st.dataframe(df.head(10).astype(str), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Info")
        
        # Create info dataframe with string conversion
        info_data = {
            'Column': df.columns.tolist(),
            'Type': [str(dtype) for dtype in df.dtypes.values],
            'Non-Null': df.count().values.tolist(),
            'Null': df.isnull().sum().values.tolist(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values.tolist()
        }
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 Basic Statistics")
    # Convert to string to avoid Arrow issues with mixed types
    stats_df = df.describe().round(2).astype(str)
    st.dataframe(stats_df, use_container_width=True)

# ============================================
# PAGE 2: DATA CLEANING
# ============================================
elif page == "🔍 Data Cleaning":
    st.markdown('<div class="sub-header">🔍 Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🧹 Data Cleaning Steps:</h4>
    <ol>
        <li><strong>Missing Age</strong> - Fill with median age</li>
        <li><strong>Missing Embarked</strong> - Fill with mode (most common port)</li>
        <li><strong>Encode Sex</strong> - Convert to numeric (male: 1, female: 0)</li>
        <li><strong>Encode Embarked</strong> - Convert to numeric</li>
        <li><strong>Create Family Size</strong> - sibsp + parch + 1</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Before Cleaning")
        
        # Show missing values before
        missing_before = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Missing Values': df.isnull().sum().values.tolist(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values.tolist()
        })
        missing_before = missing_before[missing_before['Missing Values'] > 0]
        if not missing_before.empty:
            st.dataframe(missing_before, use_container_width=True)
        else:
            st.write("No missing values found.")
        
        # Sample of original data
        st.write("**Sample of original data:**")
        sample_cols = ['age', 'embarked', 'sex']
        available_cols = [col for col in sample_cols if col in df.columns]
        if available_cols:
            st.dataframe(df[available_cols].head(10).astype(str), use_container_width=True)
    
    # Clean and encode data
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    
    with col2:
        st.subheader("✅ After Cleaning")
        
        # Show missing values after
        missing_after = pd.DataFrame({
            'Column': df_encoded.columns.tolist(),
            'Missing Values': df_encoded.isnull().sum().values.tolist(),
            'Missing %': (df_encoded.isnull().sum() / len(df_encoded) * 100).round(2).values.tolist()
        })
        missing_after = missing_after[missing_after['Missing Values'] > 0]
        if missing_after.empty:
            st.success("✅ No missing values remaining!")
        else:
            st.dataframe(missing_after, use_container_width=True)
        
        # Sample of cleaned data
        st.write("**Sample of cleaned data:**")
        clean_cols = ['age', 'embarked', 'sex', 'sex_encoded', 'embarked_encoded', 'family_size']
        available_cols = [col for col in clean_cols if col in df_encoded.columns]
        if available_cols:
            st.dataframe(df_encoded[available_cols].head(10).astype(str), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Encoding Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sex Encoding:**")
        sex_mapping = pd.DataFrame({
            'Sex': ['female', 'male'],
            'Encoded': [0, 1]
        })
        st.table(sex_mapping)
    
    with col2:
        if 'embarked' in df_encoded.columns:
            st.write("**Embarked Encoding:**")
            embarked_unique = df_encoded['embarked'].dropna().unique()
            embarked_mapping = pd.DataFrame({
                'Embarked': embarked_unique,
                'Encoded': range(len(embarked_unique))
            })
            st.table(embarked_mapping)
    
    # Download cleaned data
    csv = df_encoded.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="titanic_cleaned.csv",
        mime="text/csv"
    )

# ============================================
# PAGE 3: EXPLORATORY ANALYSIS
# ============================================
elif page == "📈 Exploratory Analysis":
    st.markdown('<div class="sub-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Clean data for analysis
    df_clean = clean_data(df)
    
    # Survival by Gender
    st.subheader("👥 Survival by Gender")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_survival = df_clean.groupby('sex')['survived'].mean() * 100
        colors = ['#ff6b6b', '#4ecdc4']
        bars = ax.bar(gender_survival.index, gender_survival.values, color=colors, alpha=0.8)
        ax.set_ylabel('Survival Rate (%)')
        ax.set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        class_survival = df_clean.groupby('pclass')['survived'].mean() * 100
        bars = ax.bar(['1st Class', '2nd Class', '3rd Class'], class_survival.values, 
                      color=['#ffd93d', '#6bcb77', '#9b59b6'], alpha=0.8)
        ax.set_ylabel('Survival Rate (%)')
        ax.set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Age Distribution
    st.subheader("📊 Age Distribution by Survival")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        survived_ages = df_clean[df_clean['survived']==1]['age'].dropna()
        not_survived_ages = df_clean[df_clean['survived']==0]['age'].dropna()
        
        ax.hist([not_survived_ages, survived_ages],
                bins=20, label=['Did Not Survive', 'Survived'],
                color=['#ff6b6b', '#4ecdc4'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        survived_fares = df_clean[df_clean['survived']==1]['fare'].dropna()
        not_survived_fares = df_clean[df_clean['survived']==0]['fare'].dropna()
        
        ax.hist([not_survived_fares, survived_fares],
                bins=20, label=['Did Not Survive', 'Survived'],
                color=['#ff6b6b', '#4ecdc4'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Fare')
        ax.set_ylabel('Frequency')
        ax.set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Embarked Analysis
    st.subheader("🚢 Survival by Embarkation Port")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        embarked_survival = df_clean.groupby('embarked')['survived'].mean() * 100
        bars = ax.bar(embarked_survival.index, embarked_survival.values, 
                     color=['#ff6b6b', '#4ecdc4', '#ffd93d'], alpha=0.8)
        ax.set_xlabel('Embarkation Port')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_title('Survival Rate by Embarkation Port', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        if 'sibsp' in df_clean.columns and 'parch' in df_clean.columns:
            df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
            family_survival = df_clean.groupby('family_size')['survived'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(family_survival.index, family_survival.values, marker='o', 
                   linestyle='-', color='#4ecdc4', linewidth=2, markersize=8)
            ax.set_xlabel('Family Size')
            ax.set_ylabel('Survival Rate (%)')
            ax.set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ============================================
# PAGE 4: MODEL TRAINING
# ============================================
elif page == "🤖 Model Training":
    st.markdown('<div class="sub-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    # Clean and prepare data
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    X, y, feature_cols = prepare_features(df_encoded)
    
    if y is None:
        st.error("❌ Target variable 'survived' not found in dataset!")
        st.stop()
    
    if X.shape[1] == 0:
        st.error("❌ No features available for training!")
        st.stop()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    st.markdown(f"""
    <div class="info-box">
    <h4>📊 Data Split</h4>
    <ul>
        <li><strong>Training samples:</strong> {X_train.shape[0]}</li>
        <li><strong>Testing samples:</strong> {X_test.shape[0]}</li>
        <li><strong>Features used:</strong> {', '.join(feature_cols)}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Train models
    with st.spinner("🤖 Training models..."):
        trained_models, cv_results = train_models(X_train, y_train)
        eval_results = evaluate_models(trained_models, X_test, y_test)
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for name in trained_models.keys():
        if name in eval_results:
            metrics = eval_results[name]
            cv_mean = cv_results[name]['cv_mean']
            cv_std = cv_results[name]['cv_std']
            
            comparison_data.append({
                'Model': name,
                'CV Accuracy': f"{cv_mean:.3f} ± {cv_std:.3f}",
                'Test Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0.5):.3f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    st.markdown("---")
    
    # Visual comparison
    st.subheader("📈 Performance Visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    models = list(eval_results.keys())
    accuracies = [eval_results[m]['accuracy'] for m in models]
    colors = ['#667eea', '#764ba2', '#4ecdc4']
    bars = ax.bar(models, accuracies, color=colors[:len(models)], alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Precision-Recall-F1 comparison
    ax = axes[0, 1]
    x = np.arange(len(models))
    width = 0.25
    
    precisions = [eval_results[m]['precision'] for m in models]
    recalls = [eval_results[m]['recall'] for m in models]
    f1_scores = [eval_results[m]['f1'] for m in models]
    
    ax.bar(x - width, precisions, width, label='Precision', color='#ff6b6b', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', color='#4ecdc4', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#ffd93d', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall & F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. ROC-AUC comparison
    ax = axes[1, 0]
    
    for name in models:
        if 'y_pred_proba' in eval_results[name]:
            y_pred_proba = eval_results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = eval_results[name]['roc_auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # 4. Confusion Matrix for best model
    ax = axes[1, 1]
    
    # Find best model by F1-score
    best_model_name = max(eval_results, key=lambda x: eval_results[x]['f1'])
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================
# PAGE 5: PREDICTION
# ============================================
elif page == "🎯 Prediction":
    st.markdown('<div class="sub-header">🎯 Predict Passenger Survival</div>', unsafe_allow_html=True)
    
    # Clean and prepare data
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    X, y, feature_cols = prepare_features(df_encoded)
    
    if y is None:
        st.error("❌ Target variable 'survived' not found in dataset!")
        st.stop()
    
    if X.shape[1] == 0:
        st.error("❌ No features available for prediction!")
        st.stop()
    
    # Train model
    with st.spinner("🤖 Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Use Random Forest for predictions
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Enter Passenger Details</h4>
    <p>Fill in the passenger information to predict survival probability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class"
        )
        
        sex = st.radio(
            "Sex",
            options=["male", "female"],
            horizontal=True
        )
        
        age = st.slider(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            step=1
        )
    
    with col2:
        fare = st.number_input(
            "Fare (£)",
            min_value=0.0,
            max_value=600.0,
            value=50.0,
            step=5.0
        )
        
        embarked = st.selectbox(
            "Embarkation Port",
            options=['S', 'C', 'Q'],
            format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x]
        )
    
    with col3:
        sibsp = st.number_input(
            "Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
        
        parch = st.number_input(
            "Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
    
    # Create feature vector
    family_size = sibsp + parch + 1
    
    # Encode inputs
    sex_encoded = 1 if sex == 'male' else 0
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map.get(embarked, 0)
    
    # Create feature array
    feature_values = []
    for col in feature_cols:
        if col == 'pclass':
            feature_values.append(pclass)
        elif col == 'sex_encoded':
            feature_values.append(sex_encoded)
        elif col == 'age':
            feature_values.append(age)
        elif col == 'fare':
            feature_values.append(fare)
        elif col == 'family_size':
            feature_values.append(family_size)
        elif col == 'embarked_encoded':
            feature_values.append(embarked_encoded)
    
    features = np.array(feature_values).reshape(1, -1)
    
    # Make prediction
    if st.button("🔮 Predict Survival", use_container_width=True):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="survived">😊 SURVIVED</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="not-survived">😞 DID NOT SURVIVE</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Survival Probability")
            
            # Create gauge chart
            fig, ax = plt.subplots(figsize=(8, 4))
            
            colors = ['#ff6b6b', '#4ecdc4']
            ax.barh(['Probability'], [probability[1]], color=colors[1], alpha=0.8, height=0.5)
            ax.barh(['Probability'], [probability[0]], left=[probability[1]], 
                   color=colors[0], alpha=0.8, height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Survival Probability Distribution', fontsize=14, fontweight='bold')
            ax.set_yticks([])
            
            ax.text(probability[1]/2, 0, f"Survive: {probability[1]:.1%}", 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            ax.text(probability[1] + probability[0]/2, 0, f"Perish: {probability[0]:.1%}", 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            
            st.pyplot(fig)
            plt.close()

# ============================================
# PAGE 6: FEATURE IMPORTANCE
# ============================================
else:
    st.markdown('<div class="sub-header">📋 Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    # Clean and prepare data
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    X, y, feature_cols = prepare_features(df_encoded)
    
    if y is None:
        st.error("❌ Target variable 'survived' not found in dataset!")
        st.stop()
    
    if X.shape[1] == 0:
        st.error("❌ No features available for analysis!")
        st.stop()
    
    # Train models
    with st.spinner("🤖 Training models for feature importance..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        for name in models:
            models[name].fit(X_train, y_train)
    
    st.markdown("""
    <div class="info-box">
    <h4>📊 Feature Importance Analysis</h4>
    <p>Understanding which features most influence survival predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for each model
    tabs = st.tabs(list(models.keys()))
    
    for i, (name, model) in enumerate(models.items()):
        with tabs[i]:
            importance_df = get_feature_importance(model, feature_cols)
            
            if importance_df is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📊 {name} - Feature Importance")
                    st.dataframe(importance_df, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    if hasattr(model, 'coef_'):
                        coef = model.coef_[0]
                        colors = ['#4ecdc4' if c > 0 else '#ff6b6b' for c in coef]
                        ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
                        ax.set_xlabel('Coefficient Value')
                        ax.set_title(f'{name} - Feature Coefficients', fontsize=14, fontweight='bold')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                    else:
                        colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
                        ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
                        ax.set_xlabel('Importance Score')
                        ax.set_title(f'{name} - Feature Importance', fontsize=14, fontweight='bold')
                    
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
    
    st.markdown("---")
    
    # Overall feature importance comparison
    st.subheader("📈 Feature Importance Comparison Across Models")
    
    # Collect importance from all models
    all_importance = []
    for name, model in models.items():
        imp_df = get_feature_importance(model, feature_cols)
        if imp_df is not None:
            imp_df['Model'] = name
            all_importance.append(imp_df)
    
    if all_importance:
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        # Pivot for comparison
        pivot_importance = combined_importance.pivot(index='feature', columns='Model', values='importance')
        
        # Normalize for better comparison
        pivot_importance = pivot_importance.div(pivot_importance.max(axis=0), axis=1)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_importance.plot(kind='barh', ax=ax, color=['#667eea', '#764ba2', '#4ecdc4'])
        ax.set_xlabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class="success-box">
        <h4>🔑 Key Insights:</h4>
        <ul>
            <li><strong>Sex (Gender)</strong> is consistently the most important feature across all models</li>
            <li><strong>Passenger Class (Pclass)</strong> is the second most important feature</li>
            <li><strong>Fare</strong> shows significant importance, correlating with class</li>
            <li><strong>Age</strong> has moderate importance, with children having higher survival rates</li>
            <li><strong>Family Size</strong> and <strong>Embarkation Port</strong> have lower impact</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">🚢 Titanic Survival Prediction</p>
    <p style="opacity: 0.9;">Built with Streamlit, Scikit-learn, and Matplotlib</p>
    <p style="opacity: 0.7; font-size: 0.9rem; margin-top: 0.5rem;">
        📊 Logistic Regression | 🌳 Decision Tree | 🌲 Random Forest
    </p>
</div>
""", unsafe_allow_html=True)