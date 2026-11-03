"""
One module per model. Each train_* function returns the fitted model with its train and
test predictions; XGBoost also returns its preprocessor and early-stopping info, and the
deep learning models also return a training info dict.
"""