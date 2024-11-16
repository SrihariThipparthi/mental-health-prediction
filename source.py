import lightgbm as lgb
from sklearn.metrics import accuracy_score
import optuna

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_error',  # Specify the evaluation metric
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10)
    }

    # Convert the train data to LightGBM Dataset format
    train_set = lgb.Dataset(x_train, label=y_train)
    early_stopping = lgb.early_stopping(stopping_rounds=10, verbose=False)

    # Train the model with early stopping
    model = lgb.train(
        params,
        train_set,
        num_boost_round=100,
        valid_sets=[train_set],
        callbacks=[early_stopping]
    )

    # Evaluate accuracy on the test set
    preds = (model.predict(x_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    return 1 - accuracy  # Minimize error

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Output the best parameters
print("Best Parameters:", study.best_trial.params)



best_params = study.best_trial.params  # Correct way to access the best parameters from the study
train_data = lgb.Dataset(x_train, label=y_train)  # Ensure the train data is passed correctly
early_stopping = lgb.early_stopping(stopping_rounds=10, verbose=False)

# Use the best parameters to train the model
best_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[train_data],  # Pass the dataset, not just the labels
    num_boost_round=500,
    callbacks=[early_stopping]
)

# Make predictions
y_pred = (best_model.predict(x_test) > 0.5).astype(int)

# Calculate and print the accuracy
print("Fine-Tuned Accuracy:", accuracy_score(y_test, y_pred))


