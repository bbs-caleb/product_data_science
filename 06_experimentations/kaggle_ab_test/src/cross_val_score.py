if isinstance(cv, tuple):
    n_splits, n_repeats = cv
    kfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, ...)
    n_folds = n_splits * n_repeats  
else:
    kfold = KFold(n_splits=cv, ...)
    n_folds = cv
