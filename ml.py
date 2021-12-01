import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge,Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("bengaluru_house_prices.csv")
# print(df.head())# print(df.shape)
# print(df.groupby('area_type')['area_type'].agg('count'))

df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
# print(df2.head()) # print(df2.isnull().sum())

df3 = df2.dropna()                       # print(df3.isnull().sum())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.drop('size', axis='columns', inplace=True)
# print(df3.head())
# #df3[df3.bhk = 20] # print(df3.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
# print(df3[df3['total_sqft'].apply(is_float)].head())
# print(df3[~df3['total_sqft'].apply(is_float)].head())

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

# print(convert_sqft_to_num('2166'))
# print(convert_sqft_to_num('2100-2850'))

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
# print(df4)
# print(df4.loc[30])
# print(df4.head(3))

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
# print(df5.head(3))

# print(len(df5.location.unique()))
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count')
#print(location_stats)#print(len(location_stats[location_stats<=10]))
location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x : 'other' if x in location_stats_less_than_10 else x)
# print(len(df5.location.unique()))


#we need to remove outliers in our code

df6 = df5[~(df5.total_sqft/df5.bhk < 300)]
#print(df6) #print(df6.price_per_sqft.describe())

def remove_pps_outliners(df):     #removing all the data beyong (sd+mean) and (mean-sd)
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-sd)) & (subdf.price_per_sqft <= (m+sd))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)               #remove/ignore all df_out
    return df_out

df7 = remove_pps_outliners(df6)
# print(df7.shape)

# if we want to know 3bedroom is more or 2 bedroom is more for same area

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
# plot_scatter_chart(df7, "Hebbal")
# plot_scatter_chart(df7, "Rajaji Nagar")

def remove_bhk_outliers(df):                #per bedroom stats
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
# print(df8.shape)
# plot_scatter_chart(df8,"Rajaji Nagar")
# plot_scatter_chart(df8,"Hebbal")

# import matplotlib
# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df8.price_per_sqft, rwidth = 0.8)
# plt.xlabel("price per sqft")
# plt.ylabel("count")
# plt.show()
#
# print(df8.bath.unique())
#
# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("Number of bathrooms")
# plt.ylabel("Count")
# plt.show()
# print(df8[df8.bath>10])
# df8[df8.bath>df8.bhk+2]

df9 = df8[df8.bath<df8.bhk+2]
# print(df9.shape)
df10 = df9.drop(['price_per_sqft'], axis='columns')
# print(df10.head(10))

dummies = pd.get_dummies(df10.location)
# print(dummies.head(3))

df11 = pd.concat([df10,dummies.drop('other', axis='columns')], axis='columns')
# print(df11.head(3))

df12 = df11.drop('location',axis='columns')
X = df12.drop('price', axis='columns')
Y = df12.price
# print(X.head(), Y.head())
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.2, random_state=10)


lr = LinearRegression()
lr.fit(xtrain, ytrain)
lr.score(xtest, ytest)


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)        #this will create random sets to test and train models
cross_val_score(LinearRegression(),X,Y, cv = cv)            #model,xtest,ytest, will give score


def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 10, 50, 200, 500],
                'selection': ['random', 'cyclic']
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [1, 10, 50, 200, 500],
                'fit_intercept': [True, False]
            }
        },
        'descision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'absolute_error'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score'])
best = find_best_model_using_gridsearchcv(X,Y)
print(best)
print(X.columns)
print(np.where(X.columns == '2nd Phase Judicial Layout'))
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    return lr.predict([x])[0]
predict_price('2nd Phase Judicial Layout',1000,2,2)

import pickle
with open('real_estate.pickle','wb') as f:
    pickle.dump(lr,f)

import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))