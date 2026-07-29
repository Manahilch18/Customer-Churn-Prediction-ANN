import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src import config

SEED = 42

def set_seeds():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

def build_ann(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu',
                     kernel_initializer=initializers.HeNormal(seed=SEED),
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(32, activation='relu',
                     kernel_initializer=initializers.HeNormal(seed=SEED),
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.20),

        layers.Dense(16, activation='relu',
                     kernel_initializer=initializers.HeNormal(seed=SEED)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_baseline(X_train, y_train, X_test, y_test):
    print("\n--- Training Baseline Model ---")
    model = build_ann(input_dim=X_train.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    checkpoint = ModelCheckpoint(config.BASELINE_MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=0)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    return model

def train_class_weighted(X_train, y_train, X_test, y_test):
    print("\n--- Training Class-Weighted Model ---")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train['Exited']), y=y_train['Exited'])
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    model = build_ann(input_dim=X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return model

def run_cross_validation(X_train, y_train):
    print("\n--- Running Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_array = X_train.values if hasattr(X_train, 'values') else X_train
    y_array = np.array(y_train)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), 1):
        X_tr, X_val = X_array[train_idx], X_array[val_idx]
        y_tr, y_val = y_array[train_idx], y_array[val_idx]

        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr)
        X_val_scaled = fold_scaler.transform(X_val)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_tr.ravel()), y=y_tr.ravel())
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        model = build_ann(input_dim=X_tr_scaled.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)

        model.fit(
            X_tr_scaled, y_tr,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        val_probs = model.predict(X_val_scaled, verbose=0).ravel()
        val_preds = (val_probs >= 0.5).astype(int)

        fold_results.append({
            'Fold': fold,
            'Precision': precision_score(y_val, val_preds),
            'Recall': recall_score(y_val, val_preds),
            'F1': f1_score(y_val, val_preds),
            'ROC-AUC': roc_auc_score(y_val, val_probs)
        })
        
        print(f"Fold {fold} -> Recall: {fold_results[-1]['Recall']:.3f}, ROC-AUC: {fold_results[-1]['ROC-AUC']:.3f}")
        
    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(os.path.join(config.REPORTS_DIR, 'cross_validation_results.csv'), index=False)
    print("Saved CV results.")

def load_and_preprocess_data():
    df = pd.read_csv(config.RAW_DATA_PATH)
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True, dtype=int)
    
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Save feature columns
    feature_columns = list(X.columns)
    joblib.dump(feature_columns, config.FEATURE_COLUMNS_PATH)
    
    return X, y

def main():
    set_seeds()
    
    # Step 1: Load and preprocess full dataset (~10,000 rows)
    X, y = load_and_preprocess_data()
    
    # Carve out 20% held-out test split BEFORE fitting scaler/model
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Step 2: Fit ONE new StandardScaler on the raw train dataset (~8,000 rows)
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Fit and saved single StandardScaler on train dataset shape {X_train_raw.shape} to {config.SCALER_PATH}")
    
    # Delete final_scaler.pkl if it exists
    if os.path.exists(config.FINAL_SCALER_PATH):
        os.remove(config.FINAL_SCALER_PATH)
        print(f"Deleted {config.FINAL_SCALER_PATH}")
        
    # Scale data using the single fitted scaler
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Save processed splits for evaluation script consistency
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(config.X_TRAIN_PATH, index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(config.X_TEST_PATH, index=False)
    pd.DataFrame(y_train).to_csv(config.Y_TRAIN_PATH, index=False)
    pd.DataFrame(y_test).to_csv(config.Y_TEST_PATH, index=False)
    
    # Step 3: Retrain the production ANN from scratch on training set
    print("\n--- Training Production Model on Train Dataset ---")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    model = build_ann(input_dim=X_train_scaled.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    model.save(config.PRODUCTION_MODEL_PATH)
    joblib.dump(config.DEFAULT_CAPACITY, config.BUSINESS_CAPACITY_PATH)
    print(f"Saved retrained model to {config.PRODUCTION_MODEL_PATH}")

if __name__ == "__main__":
    main()
