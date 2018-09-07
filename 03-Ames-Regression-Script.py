train_dir = 'data/train.csv'
kaggl_dir  = 'data/test.csv'
# submission_path = 'data/test_submission.csv'
submission_path = None
brute = False
interaction_only = True
run_lin = True
run_ridge = True
run_las = True
run_elnet = True

# Standard Imports
import numpy as np
import pandas as pd
    
# Additional Imports:
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso, ElasticNetCV

np.random.seed(42)

train_data = pd.read_csv(train_dir, index_col = 'Id')
kaggl_data = pd.read_csv(kaggl_dir,  index_col = 'Id')

# Train/Test Split

X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print('Training data has {} rows.'.format(X_train.shape[0]))
print('Testing data has {} rows.'.format(X_test.shape[0]))
print('Kaggle data has {} rows.'.format(kaggl_data.shape[0]))

# Manual Feature Engineering
print('Manual Feature Engineering...')

# Create an 'EDA' dataframe we'll use to do some exploring
EDA = X_train.copy()
EDA['SalePrice'] = y_train

# There are 27 neighborhoods. Let's put them into groups of 9:
neighborhood_ranks = EDA.groupby('Neighborhood')['SalePrice'].mean().sort_values().index

low_neigh  = neighborhood_ranks[:9]
mid_neigh  = neighborhood_ranks[9:18]
high_neigh = neighborhood_ranks[18:]

def manual_feature_eng(data):
    '''Some basic manual feature engineering based on EDA of X_train'''
    eng_data = data.copy()
    # Years info:
    eng_data['Years_Old'] = 2018 - eng_data['Year Built']
    eng_data['Garage Age'] = 2018 - eng_data['Garage Yr Blt']
    eng_data['Years Since Sale'] = 2018 - eng_data['Yr Sold']
    eng_data['Years Since Remodel'] = 2018 - eng_data['Year Remod/Add']
    eng_data.drop(['Year Built','Garage Yr Blt','Yr Sold','Year Remod/Add'],
                 axis=1, inplace=True)
    # Neighborhood info:
    eng_data['High_Neigh'] = [1 if x in high_neigh else 0 for x in eng_data['Neighborhood']]
    eng_data['Mid_Neigh'] = [1 if x in mid_neigh else 0 for x in eng_data['Neighborhood']]
    eng_data['Low_Neigh'] = [1 if x in low_neigh else 0 for x in eng_data['Neighborhood']]
    eng_data.drop('Neighborhood', axis=1, inplace=True)
    
    # Is there miscellaneous furniture?
    eng_data['MiscFurn'] = eng_data['Misc Val'] > 0
    return eng_data

X_train = manual_feature_eng(X_train)
X_test = manual_feature_eng(X_test)
kaggl_data = manual_feature_eng(kaggl_data)

# Data Preprocessing: Categorical Data
print('Processing Categorical Data...')

# Before we begin, let's check to see if there are any columns in the Kaggle 
# set that aren't in the training set:

assert [col for col in kaggl_data.columns if col not in X_train.columns] == []

# And vice versa:

assert [col for col in X_train.columns if col not in kaggl_data.columns] == []

# All of our preprocessing will ultimately go here:
def preprocessing(data):
    try:
        cleaned_data = data.drop('PID', axis=1)
    except:
        cleaned_data = data
    fillna_dict = {
        'Pool QC':'No Pool',
        'Alley':'No Alley',
        # Let's let the get_dummies drop 'Misc Features' if NA
        'Fence':'No Fence',
        'Fireplace Qu':'No Fireplace',
        # Lot frontage can be mean imputed
        'Garaga Finish': 'No Garage',
        'Garage Qual': 'No Garage',
        'Garage Cond': 'No Garage',
        'Garage Type': 'No Garage',
        'Bsmt Exposure':'No Garage',
        'BsmtFin Type 2':'No Basement',
        'BsmtFin Type 2':'No Basement',
        'BsmtFin Type 1':'No Basement',
        'Bsmt Cond':'No Basement',
        'Bsmt Qual':'No Basement',
        'Mas Vnr Type':'No Mas Vnr'        
    }
    
    cleaned_data = cleaned_data.fillna(fillna_dict)
    
    return(cleaned_data)
    
X_train = preprocessing(X_train)
X_test  = preprocessing(X_test)
kaggl_data = preprocessing(kaggl_data)

# Grab the string columns:
string_cols = X_train.select_dtypes(exclude=[np.number]).columns

# Get some dummies:
X_train = pd.get_dummies(X_train, columns=string_cols)
X_test = pd.get_dummies(X_test, columns=string_cols)
kaggl_data = pd.get_dummies(kaggl_data, columns=string_cols)

# Addressing Column Mismatch After Dummifying
print('Addressing column mismatch...')

# Add columns of zeros to test and kaggle sets for columns that *do* appear in
# the training set.

model_cols = X_train.columns

def add_model_cols(data, model_cols):
    new_data = data.copy()
    for missing_col in [col for col in model_cols if col not in data.columns]:
        new_data[missing_col] = 0
    return new_data

X_test = add_model_cols(X_test, model_cols=model_cols)
kaggl_data = add_model_cols(kaggl_data, model_cols=model_cols)

# Now, let's only consider columns in X_test and kaggl_data that appear in
# the training set. We'll call these 'model columns':

kaggl_data = kaggl_data[model_cols]
X_test     = X_test[model_cols]

# Make sure we've done this correctly:
assert X_train.shape[1] == X_test.shape[1] == kaggl_data.shape[1]
assert X_train.columns.all() == X_test.columns.all()== kaggl_data.columns.all() 

# Imputing Numerical Missing Data: Handling Numerical Data
print('Imputing missing numerical data...')

imp = Imputer(strategy='mean')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test  = imp.transform(X_test)
kaggl_data = imp.transform(kaggl_data)

def array_null_check(array):
    '''Turns an array into a dataframe so that we can check for null values'''
    return pd.DataFrame(array).isnull().sum().sum()

assert array_null_check(X_train) == array_null_check(X_test)                                  == array_null_check(kaggl_data)

# Brute Force Feature Engineering

if brute:
    print('Brute force feature engineering...')
    pf = PolynomialFeatures(interaction_only=interaction_only)
    X_train = pf.fit_transform(X_train)
    X_test  = pf.transform(X_test)
    kaggl_data = pf.transform(kaggl_data)

# Maybe this is too many columns???
print('X_train has:\n---{} rows\n---{} columns'.format(X_train.shape[0], X_train.shape[1]))

# Scaling
print('Scaling all columns...')

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test  = ss.transform(X_test)
kaggl_data = ss.transform(kaggl_data)

# Feature Elimination

from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_regression

if brute:
    print('Performing automatic feature elimination')
    # Only do feature elimination if feature engineering happened by brute force
    feature_variances = np.apply_along_axis(np.var, axis=0, arr= X_train)

    # Define a percentile threshold. Do I want the top 1% of features by variance?
    perc_thresh = np.percentile(feature_variances, 99)
    perc_thresh

    vt = VarianceThreshold(threshold=perc_thresh)
    X_train_reduced = vt.fit_transform(X_train)
    X_test_reduced  = vt.transform(X_test)
    kaggl_reduced   = vt.transform(kaggl_data)
    print('X_train now has:\n---{} rows\n---{} columns'.format(X_train.shape[0], X_train.shape[1]))
else:
    X_train_reduced = X_train
    X_test_reduced  = X_test
    kaggl_reduced   = kaggl_data

# Or do I want to select the top 1% of features according 
# to the f_regression function?

# sp = SelectPercentile(score_func=f_regression, percentile = 1)
# X_train_reduced = sp.fit_transform(X_train, y_train)
# X_test_reduced  = sp.transform(X_test)
# kaggl_reduced   = sp.transform(kaggl_data)
# print(X_train.shape[1])

## Modeling

# Linear Regression

if run_lin:
    lin = LinearRegression()
    lin.fit(X_train_reduced, y_train)
    cv_scores = cross_val_score(lin, X_train_reduced, y_train, cv=3).mean()

    print('{} model has average performance of {}'
          .format(str(lin).split('(')[0], cv_scores.mean()))

# Ridge Regression

if run_ridge:
    rid = RidgeCV()
    rid.fit(X_train_reduced, y_train)
    cv_scores = cross_val_score(rid, X_train_reduced, y_train, cv=3).mean()

    print('{} model has average performance of {}'
          .format(str(rid).split('(')[0], cv_scores.mean()))

# Lasso Regression

if run_las:
    # Define a reasonable range of alphas based on previous LASSO fits:
    alphas = np.logspace(2,4,20)
    las = LassoCV(alphas=alphas, n_jobs=-1)
    las.fit(X_train_reduced, y_train)
    cv_scores = cross_val_score(las, X_train_reduced, y_train, cv=3).mean()
    best_alpha = las.alpha_
    print('{} model has average performance of {}'
          .format(str(las).split('(')[0], cv_scores.mean()))

las = Lasso(alpha=best_alpha, max_iter=2000)
cv_scores = cross_val_score(las, X_train_reduced, y_train, cv=3).mean()
las.fit(X_train_reduced, y_train)
print('{} model has average performance of {}'
      .format(str(las).split('(')[0], cv_scores.mean()))

# ElasticNet Regression

if run_elnet:
    elnet = ElasticNetCV(n_alphas=10)
    elnet.fit(X_train_reduced, y_train)
    cv_scores = cross_val_score(elnet, X_train_reduced, y_train, cv=3).mean()

    print('{} model has average performance of {}'
          .format(str(elnet).split('(')[0], cv_scores.mean()))

# Final Model Test

models = {}

try:
    lin_score = lin.score(X_test_reduced, y_test)
    models[lin_score] = lin
    print('Test set performance of {}: {}'.format(str(lin).split('(')[0],lin_score))
except:
    pass    

try:
    rid_score = rid.score(X_test_reduced, y_test)
    models[rid_score] = rid
    print('Test set performance of {}: {}'.format(str(rid).split('(')[0],rid_score))
except:
    pass    

try:
    las_score = las.score(X_test_reduced, y_test)
    models[las_score] = las
    print('Test set performance of {}: {}'.format(str(las).split('(')[0],las_score))
except:
    pass          

try:
    elnet_score = elnet.score(X_test_reduced, y_test)
    models[elnet_score] = elnet
    print('Test set performance of {}: {}'.format(str(elnet).split('(')[0],elnet_score))
except:
    pass   

high_score = max(models.keys())
print('Best performing model was {},\nwith test set performance of {}'.format(
    str(models[high_score]).split('(')[0], round(high_score,5)))

# Choosing a Model and Outputting Submission:

# Choose a model based on test set performance:
chosen_model = models[high_score]

if submission_path:

    kaggl_preds = chosen_model.predict(kaggl_reduced)

    kaggl_id = pd.read_csv('data/test.csv')['Id']

    sample_submission = pd.read_csv('data/sample_submission.csv')
    submission_columns= sample_submission.columns

    submission = pd.DataFrame({submission_columns[0]:kaggl_id,
                               submission_columns[1]:kaggl_preds})

    submission.to_csv(submission_path, index=False) 

