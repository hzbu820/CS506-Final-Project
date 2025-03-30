#!/usr/bin/env python
"""
AAPL LSTM Direction Classifier
Uses LSTM to predict AAPL stock price direction (up/down)
Uses optimized technical indicators from the price predictor model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import time
import joblib
from itertools import product

# Add the src directory to the path so Python can find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Append to log file
    with open('outputs/aapl_lstm_log.txt', 'a') as f:
        f.write(log_message + '\n')

def create_advanced_lstm_model():
    """Create an LSTM model for classification with bidirectional layers and attention mechanism"""
    class AttentionLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.hidden_size = hidden_size
            self.attention = torch.nn.Linear(hidden_size, 1)
            
        def forward(self, lstm_output):
            # lstm_output shape: (batch_size, seq_len, hidden_size)
            attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            return context_vector, attention_weights
    
    class EnhancedLSTMClassifier(torch.nn.Module):
        """Enhanced LSTM model for stock direction classification"""
        
        def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
            super(EnhancedLSTMClassifier, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Simple LSTM layer with reduced complexity
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention mechanism
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
            
            # Simplified fully connected layers
            self.fc1 = torch.nn.Linear(hidden_size * 2, 64)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.bn1 = torch.nn.BatchNorm1d(64)
            
            # Output layer for binary classification (up/down)
            self.fc_out = torch.nn.Linear(64, 1)
            
        def forward(self, x):
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            
            # Forward propagate LSTM
            lstm_out, _ = self.lstm(x, (h0, c0))
            
            # Apply attention mechanism
            attn_output, _ = self.attention(lstm_out)
            
            # Process through simplified fully connected layers
            x = self.fc1(attn_output)
            x = self.bn1(x)
            x = torch.nn.functional.leaky_relu(x)  # Use LeakyReLU for better gradients
            x = self.dropout1(x)
            
            # Output layer (no activation, as we're using BCEWithLogitsLoss)
            logits = self.fc_out(x)
            return logits
        
        def predict(self, x, threshold=0.5):
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                logits = self.forward(x)
                probabilities = torch.sigmoid(logits)
                # Apply threshold
                return (probabilities > threshold).float()
    
    return EnhancedLSTMClassifier

def load_preprocessed_data(timeframe='15m'):
    """Load preprocessed AAPL data from CS506-Final-Project"""
    log_message(f"Loading preprocessed AAPL {timeframe} data from CS506-Final-Project")
    
    # Define the path to the preprocessed data
    data_path = f"CS506-Final-Project-main/data_processed/yfinance/full/AAPL_{timeframe}_full.csv"
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Convert datetime to proper format and set as index
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        log_message(f"Loaded {len(data)} rows of preprocessed data")
        
        # Display available columns
        log_message(f"Columns available: {', '.join(data.columns.tolist())}")
        
        return data
        
    except Exception as e:
        log_message(f"Error loading preprocessed data: {e}")
        raise

def get_stock_data(ticker="AAPL", timeframe='15m'):
    """Get stock data from preprocessed file"""
    # Use the preprocessed data from CS506-Final-Project
    df = load_preprocessed_data(timeframe)
    
    # Handle column name differences - rename to match expected format
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Apply mapping only for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    return df

def create_sequences(features, target, seq_length):
    """
    Create sequences for LSTM input
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])  # Target is the next direction after the sequence
    return np.array(X), np.array(y)

# Add Focal Loss implementation for handling class imbalance better than BCE
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification
        Args:
            alpha: Weight for the rare class (between 0-1). Default is 0.25
            gamma: Focusing parameter. Higher gamma means more focus on hard examples. Default is 2.0
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Binary cross entropy loss
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # pt is the probability of the target class
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Add alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def add_technical_indicators(df):
    """Add all technical indicators used in training to a DataFrame"""
    # First, exclude non-numeric columns that would cause errors
    exclude_columns = ['symbol', 'timeframe']
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    features = df.copy()
    
    # Basic indicators
    # EMA and SMA
    features['ema_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    features['sma_14'] = df['Close'].rolling(window=14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd_line'] = ema12 - ema26
    features['macd_signal'] = features['macd_line'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd_line'] - features['macd_signal']
    
    # Bollinger Bands
    middle_band = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    features['bollinger_upper'] = middle_band + (std_dev * 2)
    features['bollinger_middle'] = middle_band
    features['bollinger_lower'] = middle_band - (std_dev * 2)
    
    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    features['atr_14'] = tr.rolling(window=14).mean()
    
    # Price change and return metrics
    features['pct_change'] = df['Close'].pct_change() * 100
    features['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Stochastic Oscillator
    window = 14  # Standard window
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    features['stoch_k'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, 0.001))
    features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
    
    # OBV
    obv = (df['Volume'] * ((df['Close'].diff() > 0).astype(int) - 
                         (df['Close'].diff() < 0).astype(int))).cumsum()
    features['obv'] = obv
    features['obv_ema'] = features['obv'].ewm(span=20).mean()
    
    # Williams %R
    features['williams_r'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low).replace(0, 0.001))
    
    # ROC
    features['roc_5'] = df['Close'].pct_change(periods=5) * 100
    features['roc_10'] = df['Close'].pct_change(periods=10) * 100
    
    # Add market regime features
    
    # ADX (Trend Strength)
    window = 14
    atr = tr.rolling(window=window).mean()
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr.replace(0, 0.001))
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr.replace(0, 0.001))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
    features['adx'] = dx.rolling(window=window).mean()
    features['di_diff'] = plus_di - minus_di
    
    # Volatility features
    features['volatility_5'] = df['Close'].pct_change().rolling(window=5).std() * 100
    features['volatility_15'] = df['Close'].pct_change().rolling(window=15).std() * 100
    features['volatility_ratio'] = features['volatility_5'] / features['volatility_15'].replace(0, 0.001)
    
    # RSI derived features
    features['rsi_divergence'] = abs(features['rsi_14'] - 50)
    features['rsi_trend'] = features['rsi_14'].diff(3)
    
    # Efficiency Ratio
    price_change = abs(df['Close'] - df['Close'].shift(10))
    path_length = pd.Series(0.0, index=df.index)
    for i in range(1, 10):
        path_length += abs(df['Close'].shift(i-1) - df['Close'].shift(i))
    features['efficiency_ratio'] = price_change / path_length.replace(0, 0.001)
    
    # Choppiness Index
    window = 14
    atr_sum = atr.rolling(window=window).sum()
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    features['choppiness'] = 100 * np.log10(atr_sum / (highest_high - lowest_low).replace(0, 0.001)) / np.log10(window)
    
    # BB Width
    features['bb_width'] = (features['bollinger_upper'] - features['bollinger_lower']) / features['bollinger_middle'].replace(0, 0.001)
    
    # Add NEW relative strength indicators
    
    # Simulate market benchmark (can be replaced with actual data if available)
    # For this example, we'll create a synthetic market benchmark based on AAPL price with some noise
    market_close = df['Close'] * (1 + np.random.normal(0, 0.001, len(df)))
    
    # 1. Relative strength to "market"
    # Calculate percentage changes
    features['market_pct_change'] = market_close.pct_change() * 100
    
    # Relative strength (ratio of stock return to market return)
    features['rel_strength_1d'] = features['pct_change'] / features['market_pct_change'].replace(0, 0.001)
    
    # 2. Normalized relative strength
    # Calculate rolling correlation between stock and market
    rolling_corr = df['Close'].pct_change().rolling(window=20).corr(market_close.pct_change())
    features['market_correlation'] = rolling_corr
    
    # Calculate beta (stock volatility relative to market)
    stock_returns = df['Close'].pct_change().dropna()
    market_returns = market_close.pct_change().dropna()
    if len(stock_returns) > 20:
        # Calculate rolling beta using covariance and variance
        rolling_cov = stock_returns.rolling(window=20).cov(market_returns)
        rolling_var = market_returns.rolling(window=20).var()
        features['beta'] = rolling_cov / rolling_var.replace(0, 0.001)
    else:
        features['beta'] = 1.0  # Default if not enough data
    
    # 3. Relative strength indicator (RSI-based)
    features['rel_strength_rsi'] = features['rsi_14'] - features['rsi_14'].rolling(window=10).mean()
    
    # 4. Alpha (Excess return over expected return based on market)
    # Simple alpha calculation
    features['alpha'] = features['pct_change'] - (features['beta'] * features['market_pct_change'])
    
    # 5. Mean reversion signals
    # Distance from historical mean (z-score)
    price_mean = df['Close'].rolling(window=20).mean()
    price_std = df['Close'].rolling(window=20).std()
    features['price_z_score'] = (df['Close'] - price_mean) / price_std.replace(0, 0.001)
    
    # 6. Multiple timeframe momentum
    # Fast vs slow momentum comparison
    features['mom_fast'] = df['Close'].pct_change(periods=5) * 100
    features['mom_slow'] = df['Close'].pct_change(periods=20) * 100
    features['mom_divergence'] = features['mom_fast'] - features['mom_slow']
    
    # 7. Reversal detection - FIX for ValueError: Series.replace cannot use dict-value
    # Ratio of current close to recent high/low - safer implementation
    highest_high_safe = highest_high.copy()
    lowest_low_safe = lowest_low.copy()
    
    # Replace 0 values with a small value to avoid division by zero
    highest_high_safe = highest_high_safe.replace(0, 0.001)
    lowest_low_safe = lowest_low_safe.replace(0, 0.001)
    
    features['close_to_high_ratio'] = df['Close'] / highest_high_safe
    features['close_to_low_ratio'] = df['Close'] / lowest_low_safe
    
    # 8. Return acceleration (momentum change)
    features['return_accel'] = features['pct_change'].diff()
    
    # Clean up NaN values
    features = features.ffill().bfill().fillna(0)
    
    return features

def train_direction_model(ticker="AAPL", sequence_length=60, hidden_size=64, num_layers=1, learning_rate=0.001, batch_size=32, epochs=50, train_split=0.8):
    """
    Train an LSTM model to predict stock price direction
    
    Args:
        ticker: Stock ticker symbol
        sequence_length: Number of time steps in each sequence
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        train_split: Proportion of data to use for training
        
    Returns:
        Trained model, selected feature columns, scaler, and threshold
    """
    log_message(f"Starting model training for {ticker}")
    
    # Get stock data
    df = get_stock_data(ticker, timeframe='15m')
    log_message(f"Loaded {len(df)} rows of data")
    
    # Remove non-numeric columns
    exclude_columns = ['symbol', 'timeframe']
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Create target variable
    df['Price_Change'] = df['Close'].diff()
    df['Direction'] = (df['Price_Change'] > 0).astype(int)
    df = df.dropna()
    
    # Display class distribution
    class_counts = df['Direction'].value_counts()
    up_count = class_counts.get(1, 0)
    down_count = class_counts.get(0, 0)
    log_message(f"Class distribution - Up: {up_count} ({up_count/(up_count+down_count)*100:.2f}%), Down: {down_count} ({down_count/(up_count+down_count)*100:.2f}%)")
    
    # Calculate class weights for imbalanced classes - increased weight for UP class
    pos_weight = down_count / up_count * 2.5 if up_count > 0 else 3.0  # Increased from 1.5 to 2.5
    log_message(f"Using positive class weight: {pos_weight:.4f}")
    
    # Prepare features
    log_message("Adding technical indicators...")
    features = add_technical_indicators(df)
    
    # Ensure target is aligned with features
    target = df['Direction']
    target = target.loc[features.index]
    
    # Feature selection
    log_message("Performing feature selection...")
    X = features.values
    y = target.values
    
    # Use RandomForest for feature selection with higher weight for minority class
    class_weight = {
        0: 1.0,
        1: pos_weight
    }
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight=class_weight
    )
    rf.fit(X, y)
    
    # Select important features
    selector = SelectFromModel(rf, threshold="median", prefit=True)
    selected_mask = selector.get_support()
    selected_features = features.columns[selected_mask]
    
    # Save all selected features
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    
    log_message(f"Selected {len(selected_features)} features")
    
    # Optional: Print feature importances
    feature_importances = list(zip(features.columns, rf.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    log_message("Top 10 features by importance:")
    for feature, importance in feature_importances[:10]:
        log_message(f"  {feature}: {importance:.4f}")
    
    # Exclude target columns from features used for model training
    cols_to_exclude = ['Direction', 'Price_Change']
    feature_cols_for_model = [col for col in selected_features if col not in cols_to_exclude]
    
    # Save the model-specific feature columns (excluding targets)
    with open('outputs/feature_columns_for_model.txt', 'w') as f:
        f.write('\n'.join(feature_cols_for_model))
        
    log_message(f"Using {len(feature_cols_for_model)} features for model (excluding target columns)")
    
    # Create DataFrame with selected features
    features_for_model = features[feature_cols_for_model].copy()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_model)
    
    # Save the scaler for prediction
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(scaler, 'outputs/models/feature_scaler.pkl')
    
    # Create sequences
    X, y = create_sequences(scaled_features, target.values, sequence_length)
    
    # Cross-validation for time series
    n_folds = 5
    fold_size = len(X) // n_folds
    
    # Store results
    cv_results = []
    best_cv_f1 = 0
    best_cv_threshold = 0.35  # Start with a lower threshold (0.5 -> 0.35)
    best_fold = 0
    
    log_message(f"Performing {n_folds}-fold time-series cross-validation")
    
    for fold in range(n_folds - 1):  # Use n_folds-1 to keep the last fold as test
        # Define fold boundaries
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = val_start + fold_size
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        
        log_message(f"Fold {fold+1}: Train size={len(X_train)}, Val size={len(X_val)}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Define model
        input_size = X_train.shape[2]
        EnhancedLSTMClassifier = create_advanced_lstm_model()
        model = EnhancedLSTMClassifier(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5
        )
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Define loss function with class weighting
        focal_loss = FocalLoss(alpha=0.85, gamma=2.5)  # Increased alpha slightly (0.75 -> 0.85)
        
        # Define optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = focal_loss(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = focal_loss(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                log_message(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= 10:
                log_message(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        
        # Evaluate with different thresholds to find the best one
        val_outputs_np = np.array(val_outputs).flatten()
        val_targets_np = np.array(val_targets).flatten()
        
        # Try different thresholds for prediction
        thresholds = np.linspace(0.2, 0.6, 9)  # Changed range to focus on lower thresholds
        best_f1 = 0
        best_threshold = 0.35  # Start with a lower default threshold
        
        for threshold in thresholds:
            preds = (val_outputs_np > threshold).astype(int)
            if len(np.unique(preds)) > 1:  # Ensure we have both classes
                f1 = f1_score(val_targets_np, preds)
                
                # Count predictions
                up_preds = np.sum(preds)
                up_pct = up_preds / len(preds) * 100
                
                # Only update if we have a reasonable number of UP predictions
                if up_preds >= 10 and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    log_message(f"Fold {fold+1}, Threshold {threshold:.2f}: F1={f1:.4f}, UP={up_preds} ({up_pct:.1f}%)")
        
        # Save results for this fold
        cv_results.append({
            'fold': fold + 1,
            'f1': best_f1,
            'threshold': best_threshold
        })
        
        # Update best overall results
        if best_f1 > best_cv_f1:
            best_cv_f1 = best_f1
            best_cv_threshold = best_threshold
            best_fold = fold + 1
    
    # Log cross-validation results
    log_message("\nCross-validation results:")
    for result in cv_results:
        log_message(f"Fold {result['fold']}: F1={result['f1']:.4f}, Threshold={result['threshold']:.4f}")
    
    log_message(f"Best F1: {best_cv_f1:.4f} in Fold {best_fold} with threshold {best_cv_threshold:.4f}")
    
    # Train final model on full dataset
    log_message("\nTraining final model on full dataset...")
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    log_message(f"Final train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Define final model
    input_size = X_train.shape[2]
    EnhancedLSTMClassifier = create_advanced_lstm_model()
    final_model = EnhancedLSTMClassifier(
        input_size=input_size, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.5
    )
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)
    
    # Define loss function with class weighting
    focal_loss = FocalLoss(alpha=0.85, gamma=2.5)  # Increased alpha slightly
    
    # Define optimizer with weight decay
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop for final model
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training
        final_model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = final_model(batch_X)
            loss = focal_loss(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        final_model.eval()
        val_loss = 0
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = final_model(batch_X)
                loss = focal_loss(outputs, batch_y)
                val_loss += loss.item()
                
                val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            log_message(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = final_model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= 10:
            log_message(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for this fold
    final_model.load_state_dict(best_model_state)
    
    # Evaluate with different thresholds to find the best one
    val_outputs_np = np.array(val_outputs).flatten()
    val_targets_np = np.array(val_targets).flatten()
    
    # Try different thresholds for prediction
    thresholds = np.linspace(0.2, 0.6, 9)  # Changed range to focus on lower thresholds
    best_f1 = 0
    best_threshold = 0.35  # Start with a lower default threshold
    
    for threshold in thresholds:
        preds = (val_outputs_np > threshold).astype(int)
        if len(np.unique(preds)) > 1:  # Ensure we have both classes
            f1 = f1_score(val_targets_np, preds)
            
            # Count predictions
            up_preds = np.sum(preds)
            up_pct = up_preds / len(preds) * 100
            
            # Only update if we have a reasonable number of UP predictions
            if up_preds >= 10 and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                log_message(f"Threshold {threshold:.2f}: F1={f1:.4f}, UP={up_preds} ({up_pct:.1f}%)")
    
    # Save results for this fold
    cv_results.append({
        'fold': 'Final',
        'f1': best_f1,
        'threshold': best_threshold
    })
    
    log_message("\nFinal model performance:")
    log_message(f"F1={best_f1:.4f}, Threshold={best_threshold:.4f}")
    
    return final_model, selected_features, scaler, best_threshold

class GRUClassifier(torch.nn.Module):
    """GRU-based model for stock direction classification"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = torch.nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc = torch.nn.Linear(hidden_size * 2, 64)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(64)
        
        # Output layer
        self.fc_out = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        gru_out, _ = self.gru(x, h0)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        # Process through fully connected layer
        x = self.fc(context)
        x = self.bn(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc_out(x)
        return x
    
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return (probabilities > threshold).float()

class CNNLSTMClassifier(torch.nn.Module):
    """CNN-LSTM hybrid model for stock direction classification"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN layers for feature extraction
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = torch.nn.Dropout(0.2)
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = torch.nn.Linear(hidden_size * 2, 64)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(64)
        
        # Output layer
        self.fc_out = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        
        # Permute for CNN
        x_permuted = x.permute(0, 2, 1)  # [batch_size, input_size, seq_len]
        
        # CNN feature extraction
        x_cnn = torch.nn.functional.relu(self.conv1(x_permuted))
        x_cnn = self.pool(x_cnn)
        x_cnn = self.dropout_cnn(x_cnn)
        x_cnn = torch.nn.functional.relu(self.conv2(x_cnn))
        x_cnn = self.dropout_cnn(x_cnn)
        
        # Permute back for LSTM
        x_cnn = x_cnn.permute(0, 2, 1)  # [batch_size, seq_len/2, 64]
        
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x_cnn, (h0, c0))
        
        # Get final hidden state
        final_state = lstm_out[:, -1, :]
        
        # Process through fully connected layer
        x = self.fc(final_state)
        x = self.bn(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc_out(x)
        return x
    
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return (probabilities > threshold).float()

def train_ensemble_model(ticker="AAPL", sequence_length=60, train_split=0.8, n_models=3):
    """Train an ensemble of models for stock direction prediction"""
    log_message(f"Starting ensemble model training for {ticker} with {n_models} different architectures")
    
    # Get stock data
    df = get_stock_data(ticker, timeframe='15m')
    
    # Remove non-numeric columns
    exclude_columns = ['symbol', 'timeframe']
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Create target variable
    df['Price_Change'] = df['Close'].diff()
    df['Direction'] = (df['Price_Change'] > 0).astype(int)
    df = df.dropna()
    
    # Display class distribution
    class_counts = df['Direction'].value_counts()
    up_count = class_counts.get(1, 0)
    down_count = class_counts.get(0, 0)
    log_message(f"Class distribution - Up: {up_count} ({up_count/(up_count+down_count)*100:.2f}%), Down: {down_count} ({down_count/(up_count+down_count)*100:.2f}%)")
    
    # Prepare features
    log_message("Adding technical indicators for ensemble...")
    features = add_technical_indicators(df)
    
    # Ensure target is aligned with features
    target = df['Direction']
    target = target.loc[features.index]
    
    # Feature selection
    log_message("Performing feature selection for ensemble...")
    X = features.values
    y = target.values
    
    # Use RandomForest for feature selection with higher weight for minority class
    class_weight = {
        0: 1.0,
        1: down_count / up_count * 2.5 if up_count > 0 else 3.0
    }
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight=class_weight
    )
    rf.fit(X, y)
    
    # Select important features
    selector = SelectFromModel(rf, threshold="median", prefit=True)
    selected_mask = selector.get_support()
    selected_features = features.columns[selected_mask]
    
    log_message(f"Selected {len(selected_features)} features for ensemble models")
    
    # Exclude target-related columns
    cols_to_exclude = ['Direction', 'Price_Change']
    feature_cols_for_model = [col for col in selected_features if col not in cols_to_exclude]
    
    # Create DataFrame with selected features
    features_for_model = features[feature_cols_for_model].copy()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_model)
    
    # Save the scaler
    os.makedirs('outputs/models/ensemble', exist_ok=True)
    joblib.dump(scaler, 'outputs/models/ensemble/feature_scaler.pkl')
    
    # Save the feature columns
    with open('outputs/models/ensemble/feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_cols_for_model))
    
    # Create sequences
    X, y = create_sequences(scaled_features, target.values, sequence_length)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    log_message(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Define different model architectures and hyperparameters
    model_configs = [
        # Standard LSTM (current architecture)
        {
            'type': 'LSTM',
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30,
            'threshold': 0.35
        },
        # GRU (often works better for financial data)
        {
            'type': 'GRU',
            'hidden_size': 96,
            'num_layers': 1,
            'dropout': 0.3,
            'learning_rate': 0.002,
            'batch_size': 64,
            'epochs': 30,
            'threshold': 0.4
        },
        # CNN-LSTM hybrid (captures local and global patterns)
        {
            'type': 'CNN-LSTM',
            'hidden_size': 48,
            'num_layers': 1,
            'dropout': 0.4,
            'learning_rate': 0.0015,
            'batch_size': 48,
            'epochs': 30,
            'threshold': 0.3
        },
        # Deep LSTM for complex patterns
        {
            'type': 'LSTM',
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.6,
            'learning_rate': 0.0008,
            'batch_size': 32,
            'epochs': 30,
            'threshold': 0.45
        },
    ]
    
    # Subset to the requested number of models
    model_configs = model_configs[:n_models]
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ensemble components
    ensemble_models = []
    ensemble_thresholds = []
    
    # Get input size
    input_size = X_train_tensor.shape[2]
    
    # Train each model in the ensemble
    for i, config in enumerate(model_configs):
        log_message(f"\n=== Training Ensemble Model {i+1}/{len(model_configs)} ({config['type']}) ===")
        
        # Create model based on type
        if config['type'] == 'LSTM':
            EnhancedLSTMClassifier = create_advanced_lstm_model()
            model = EnhancedLSTMClassifier(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(device)
        elif config['type'] == 'GRU':
            model = GRUClassifier(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(device)
        elif config['type'] == 'CNN-LSTM':
            model = CNNLSTMClassifier(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(device)
        
        # Create dataloader with the model's batch size
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False
        )
        
        # Loss function with class weighting
        focal_loss = FocalLoss(alpha=0.85, gamma=2.5)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_f1 = 0
        best_model_state = None
        best_threshold = config['threshold']
        
        log_message(f"Starting training with initial threshold {best_threshold}")
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = focal_loss(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = focal_loss(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(test_loader)
            scheduler.step(val_loss)
            
            # Find optimal threshold
            val_outputs_np = np.array(val_outputs).flatten()
            val_targets_np = np.array(val_targets).flatten()
            
            # Try different thresholds
            test_thresholds = np.linspace(0.2, 0.6, 9)
            current_threshold = best_threshold
            current_f1 = 0
            
            for test_threshold in test_thresholds:
                test_preds = (val_outputs_np > test_threshold).astype(int)
                if len(np.unique(test_preds)) > 1:
                    test_f1 = f1_score(val_targets_np, test_preds, zero_division=0)
                    
                    # Count predictions
                    up_preds = np.sum(test_preds)
                    down_preds = len(test_preds) - up_preds
                    
                    # Only update if we have both classes and improved F1
                    if up_preds > 0 and down_preds > 0 and test_f1 > current_f1:
                        current_f1 = test_f1
                        current_threshold = test_threshold
            
            # Calculate metrics with current threshold
            preds = (val_outputs_np > current_threshold).astype(int)
            f1 = f1_score(val_targets_np, preds, zero_division=0)
            
            # Log progress every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log_message(f"Fold {i+1}, Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Threshold: {current_threshold:.4f}")
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()
                best_threshold = current_threshold
        
        # Save the model - moved outside epoch loop
        model_save_path = f"outputs/models/ensemble/model_{i+1}_{config['type']}.pth"
        torch.save({
            'model_type': config['type'],
            'model_state_dict': best_model_state,
            'threshold': float(best_threshold),
            'f1_score': float(best_f1),
            'config': config
        }, model_save_path)
        
        log_message(f"Saved model {i+1} with F1: {best_f1:.4f} and threshold: {best_threshold:.4f}")
        
        # Add to ensemble
        ensemble_models.append((model, config['type']))
        ensemble_thresholds.append(best_threshold)
    
    # Evaluate ensemble on test set - moved outside the model loop
    log_message("\n=== Evaluating Ensemble Model ===")
    
    # Make predictions with each model
    all_model_outputs = []
    
    for (model, model_type), threshold in zip(ensemble_models, ensemble_thresholds):
        model.eval()
        model_outputs = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                model_outputs.extend(probs.cpu().numpy().flatten())
        
        all_model_outputs.append(np.array(model_outputs).flatten())
    
    # Combine predictions (average of probabilities)
    ensemble_probs = np.mean(all_model_outputs, axis=0)
    
    # Find optimal threshold for ensemble
    ensemble_thresholds = np.linspace(0.2, 0.6, 9)
    best_ensemble_f1 = 0
    best_ensemble_threshold = 0.35  # Default
    
    for threshold in ensemble_thresholds:
        ensemble_preds = (ensemble_probs > threshold).astype(int)
        if len(np.unique(ensemble_preds)) > 1:
            ens_f1 = f1_score(y_test, ensemble_preds, zero_division=0)
            
            # Count predictions
            up_count = np.sum(ensemble_preds)
            down_count = len(ensemble_preds) - up_count
            
            # Only update if we have both classes and improved F1
            if up_count > 0 and down_count > 0 and ens_f1 > best_ensemble_f1:
                best_ensemble_f1 = ens_f1
                best_ensemble_threshold = threshold
                
                # Log distribution
                up_pct = up_count / len(ensemble_preds) * 100
                log_message(f"Threshold {threshold:.2f}: F1={ens_f1:.4f}, UP={up_count} ({up_pct:.1f}%), DOWN={down_count}")
    
    # Make final ensemble predictions with best threshold
    final_preds = (ensemble_probs > best_ensemble_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds, zero_division=0)
    recall = recall_score(y_test, final_preds, zero_division=0)
    f1 = f1_score(y_test, final_preds, zero_division=0)
    
    log_message("\nEnsemble Model Performance:")
    log_message(f"Accuracy: {accuracy:.4f}")
    log_message(f"Precision: {precision:.4f}")
    log_message(f"Recall: {recall:.4f}")
    log_message(f"F1 Score: {f1:.4f}")
    log_message(f"Best Threshold: {best_ensemble_threshold:.4f}")
    
    # Display classification report
    report = classification_report(y_test, final_preds, target_names=['Down', 'Up'])
    log_message(f"\nClassification Report:\n{report}")
    
    # Save ensemble metadata
    ensemble_meta = {
        'model_types': [m[1] for m in ensemble_models],
        'ensemble_threshold': float(best_ensemble_threshold),
        'f1_score': float(f1),
        'model_count': len(ensemble_models)
    }
    
    with open('outputs/models/ensemble/ensemble_meta.json', 'w') as f:
        import json
        json.dump(ensemble_meta, f)
    
    log_message("Saved ensemble metadata")
    
    return ensemble_meta

def predict_with_ensemble(new_data=None, model_dir='outputs/models/ensemble', sequence_length=60):
    """Make predictions using the ensemble of models"""
    try:
        log_message("Making prediction with ensemble model...")
        
        # Load ensemble metadata
        try:
            import json
            with open(f"{model_dir}/ensemble_meta.json", 'r') as f:
                ensemble_meta = json.load(f)
            
            ensemble_threshold = ensemble_meta['ensemble_threshold']
            model_types = ensemble_meta['model_types']
            model_count = ensemble_meta['model_count']
            
            log_message(f"Loaded ensemble with {model_count} models and threshold {ensemble_threshold}")
        except Exception as e:
            log_message(f"Error loading ensemble metadata: {e}")
            ensemble_threshold = 0.35  # Default
            model_count = 3  # Default
        
        # If no new data is provided, load the most recent data
        if new_data is None:
            df = get_stock_data(ticker="AAPL", timeframe='15m')
            log_message(f"Loaded {len(df)} rows of current data")
        else:
            df = new_data
        
        # Prepare features
        log_message("Preparing features for ensemble prediction...")
        features_full = add_technical_indicators(df)
        
        # Load feature columns
        try:
            with open(f"{model_dir}/feature_columns.txt", 'r') as f:
                feature_cols = [line.strip() for line in f.readlines()]
            log_message(f"Loaded {len(feature_cols)} feature columns for ensemble")
        except Exception as e:
            log_message(f"Error loading feature columns: {e}, using all available features")
            # Exclude non-numeric and target columns
            exclude_cols = ['symbol', 'timeframe', 'Direction', 'Price_Change']
            feature_cols = [col for col in features_full.columns if col not in exclude_cols]
        
        # Extract only available features
        available_features = [f for f in feature_cols if f in features_full.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            log_message(f"Warning: {len(missing)} features are missing: {missing}")
        
        features = features_full[available_features].copy()
        
        # Fill NaN values
        features = features.ffill().bfill().fillna(0)
        
        # Scale features
        try:
            scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
            log_message("Using saved ensemble scaler")
        except Exception as e:
            log_message(f"Error loading scaler: {e}, using new scaler")
            scaler = StandardScaler()
        
        scaled_features = scaler.transform(features)
        
        # Create sequences
        X_sequences = []
        for i in range(len(scaled_features) - sequence_length + 1):
            X_sequences.append(scaled_features[i:i+sequence_length])
        
        if not X_sequences:
            log_message("Not enough data for sequence prediction")
            return None
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_sequences)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load each model and make predictions
        all_model_outputs = []
        input_size = X_tensor.shape[2]
        
        # Find all model files
        import glob
        model_files = glob.glob(f"{model_dir}/model_*.pth")
        
        if not model_files:
            log_message("No ensemble model files found")
            return None
        
        log_message(f"Found {len(model_files)} ensemble model files")
        
        for model_file in model_files:
            try:
                # Load model
                checkpoint = torch.load(model_file, map_location=device)
                model_type = checkpoint.get('model_type', 'LSTM')
                model_threshold = float(checkpoint.get('threshold', 0.35))
                
                # Create appropriate model
                if model_type == 'LSTM':
                    EnhancedLSTMClassifier = create_advanced_lstm_model()
                    model = EnhancedLSTMClassifier(
                        input_size=input_size,
                        hidden_size=64,
                        num_layers=1,
                        dropout=0.5
                    ).to(device)
                elif model_type == 'GRU':
                    model = GRUClassifier(
                        input_size=input_size,
                        hidden_size=96,
                        num_layers=1,
                        dropout=0.3
                    ).to(device)
                elif model_type == 'CNN-LSTM':
                    model = CNNLSTMClassifier(
                        input_size=input_size,
                        hidden_size=48,
                        num_layers=1,
                        dropout=0.4
                    ).to(device)
                else:
                    log_message(f"Unknown model type {model_type}, skipping")
                    continue
                
                # Load model weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Make predictions
                model_outputs = []
                with torch.no_grad():
                    X_tensor_device = X_tensor.to(device)
                    outputs = model(X_tensor_device)
                    probs = torch.sigmoid(outputs)
                    model_outputs = probs.cpu().numpy().flatten()
                
                all_model_outputs.append(model_outputs)
                log_message(f"Made predictions with {model_type} model, threshold: {model_threshold:.4f}")
                
            except Exception as e:
                log_message(f"Error with model {model_file}: {e}")
                import traceback
                log_message(traceback.format_exc())
        
        if not all_model_outputs:
            log_message("No models successfully made predictions")
            return None
        
        # Combine predictions (average of probabilities)
        ensemble_probs = np.mean(all_model_outputs, axis=0)
        
        # Apply ensemble threshold
        ensemble_preds = (ensemble_probs > ensemble_threshold).astype(int)
        
        # Create results DataFrame
        timestamps = df.index[sequence_length-1:]
        
        results = pd.DataFrame({
            'timestamp': timestamps,
            'probability': ensemble_probs,
            'prediction': ensemble_preds
        })
        
        # Count predictions of each class
        up_preds = np.sum(ensemble_preds)
        down_preds = len(ensemble_preds) - up_preds
        up_pct = up_preds / len(ensemble_preds) * 100 if len(ensemble_preds) > 0 else 0
        
        log_message(f"Ensemble prediction distribution: UP={up_preds} ({up_pct:.1f}%), DOWN={down_preds}")
        
        # Display the most recent prediction
        if len(results) > 0:
            latest_prediction = results.iloc[-1]
            direction = "UP" if latest_prediction['prediction'] == 1 else "DOWN"
            log_message(f"\nLatest Ensemble Prediction ({latest_prediction['timestamp']}):")
            log_message(f"Direction: {direction} with probability {latest_prediction['probability']:.4f}")
        
        return results
        
    except Exception as e:
        log_message(f"Error making ensemble prediction: {e}")
        import traceback
        log_message(traceback.format_exc())
        return None

def main():
    """Main function to run the AAPL direction prediction model"""
    log_message("Starting AAPL direction prediction with LSTM Classifier")
    
    try:
        # Define sequence_length here so it's available for both training and prediction
        sequence_length = 60
        
        # Ask user which model to train
        model_type = input("Which model to train? (1: Single model, 2: Ensemble, Default: 1): ").strip() or "1"
        
        if model_type == "2":
            # Train ensemble model
            log_message("Training ensemble model...")
            n_models = 3
            ensemble_meta = train_ensemble_model(
                ticker="AAPL",
                sequence_length=sequence_length,
                train_split=0.8,
                n_models=n_models
            )
            
            log_message("Ensemble training complete!")
            
            # Add a small delay to ensure files are fully written
            log_message("Waiting for model files to be completely written...")
            time.sleep(2)
            
            # Test ensemble prediction
            predict_with_ensemble(sequence_length=sequence_length)
            
        else:
            # Train single model
            model, feature_columns, scaler, threshold = train_direction_model(
                ticker="AAPL",
                sequence_length=sequence_length,
                hidden_size=64,
                num_layers=1,
                learning_rate=0.001,
                batch_size=32,
                epochs=50,
                train_split=0.8
            )
            
            log_message("Training complete!")
            
            # Add a small delay to ensure file is fully written
            log_message("Waiting for model file to be completely written...")
            time.sleep(2)
            
            # Test in-memory prediction
            # ... (rest of the code for in-memory prediction)
        
    except Exception as e:
        log_message(f"Error: {e}")
        log_message(f"Traceback: {sys.exc_info()}")
        raise

if __name__ == "__main__":
    main() 