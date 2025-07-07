from __future__ import division



from datetime import datetime, timedelta,date
import pandas as pd
# %matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


import pandas as pd


adv = pd.read_excel("online_retail_II.xlsx") 

pyoff.init_notebook_mode()
adv.head()

adv['InvoiceDate'] = pd.to_datetime(adv['InvoiceDate'])

adv['InvoiceYearMonth'] = adv['InvoiceDate'].map(lambda date: 100*date.year + date.month)

adv.describe()

adv['Country'].value_counts()

adv_uk = adv.query("Country=='United Kingdom'").reset_index(drop=True)

adv_user = pd.DataFrame(adv['Customer ID'].unique())
adv_user.columns = ['Customer ID']
adv_user.head()

adv_uk.head()

adv_max_purchase = adv_uk.groupby('Customer ID').InvoiceDate.max().reset_index()
adv_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
adv_max_purchase.head()

adv_max_purchase['Recency'] = (adv_max_purchase['MaxPurchaseDate'].max() - adv_max_purchase['MaxPurchaseDate']).dt.days
adv_max_purchase.head()

adv_user.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
adv_user = pd.merge(adv_user, adv_max_purchase[['CustomerID','Recency']], on='CustomerID')
adv_user.head()

sse={}
adv_recency = adv_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(adv_recency)
    adv_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans = KMeans(n_clusters=4)
adv_user['RecencyCluster'] = kmeans.fit_predict(adv_user[['Recency']])

adv_user.head()

adv_user.groupby('RecencyCluster')['Recency'].describe()

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

adv_user = order_cluster('RecencyCluster', 'Recency',adv_user,False)

adv_user.head()

adv_user.groupby('RecencyCluster')['Recency'].describe()

adv_frequency = adv_uk.groupby('Customer ID').InvoiceDate.count().reset_index()
adv_frequency.columns = ['CustomerID','Frequency']

adv_frequency.head()

adv_user = pd.merge(adv_user, adv_frequency, on='CustomerID')

adv_user.head()

sse={}
adv_recency = adv_user[['Frequency']].copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(adv_recency)
    adv_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans=KMeans(n_clusters=4)
adv_user['FrequencyCluster']=kmeans.fit_predict(adv_user[['Frequency']])

adv_user = order_cluster('FrequencyCluster', 'Frequency', adv_user, True )
adv_user.groupby('FrequencyCluster')['Frequency'].describe()

adv_uk['Revenue'] = adv_uk['Price'] * adv_uk['Quantity']
adv_revenue = adv_uk.groupby('Customer ID').Revenue.sum().reset_index()

adv_revenue.head()

adv_revenue.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
adv_user = pd.merge(adv_user, adv_revenue, on='CustomerID')
adv_user.head()

sse={}
adv_recency = adv_user[['Revenue']].copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(adv_recency)
    adv_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans = KMeans(n_clusters=4)
adv_user['RevenueCluster'] = kmeans.fit_predict(adv_user[['Revenue']])

adv_user = order_cluster('RevenueCluster', 'Revenue',adv_user,True)
adv_user.groupby('RevenueCluster')['Revenue'].describe()

adv_user['OverallScore'] = adv_user['RecencyCluster'] + adv_user['FrequencyCluster'] + adv_user['RevenueCluster']
adv_user.groupby('OverallScore')[['Recency','Frequency','Revenue']].mean()

adv_user['Segment'] = 'Low-Value'
adv_user.loc[adv_user['OverallScore']>2,'Segment'] = 'Mid-Value'
adv_user.loc[adv_user['OverallScore']>4,'Segment'] = 'High-Value'

adv_user

adv_uk.head()

adv_uk['InvoiceDate'].describe()

adv_3m = adv_uk[(adv_uk.InvoiceDate.dt.date < date(2010,6,1)) & (adv_uk.InvoiceDate.dt.date >= date(2010,3,1))].reset_index(drop=True)
adv_6m = adv_uk[(adv_uk.InvoiceDate.dt.date >= date(2010,6,1)) & (adv_uk.InvoiceDate.dt.date < date(2010,12,1))].reset_index(drop=True)

adv_6m['Revenue'] = adv_6m['Price'] * adv_6m['Quantity']
adv_user_6m = adv_6m.groupby('Customer ID')['Revenue'].sum().reset_index()
adv_user_6m.columns = ['CustomerID','m6_Revenue']

adv_user_6m.head()

plt.figure(figsize=(10, 6))
plt.hist(adv_user_6m['m6_Revenue'], bins=50, color='skyblue', edgecolor='black')
plt.title('6-Month Revenue Distribution')
plt.xlabel('Revenue')
plt.ylabel('Number of Customers')
plt.grid(True)
plt.show()

adv_user.head()

adv_uk.head()

adv_merge = pd.merge(adv_user, adv_user_6m, on='CustomerID', how='left')

adv_merge = adv_merge.fillna(0)

adv_graph = adv_merge[adv_merge['m6_Revenue'] < 50000]

segment_settings = {'Low-Value': {'color': 'blue', 'size': 50, 'alpha': 0.8, 'label': 'Low'},
    'Mid-Value': {'color': 'green', 'size': 70, 'alpha': 0.5, 'label': 'Mid'},
    'High-Value': {'color': 'red', 'size': 90, 'alpha': 0.9, 'label': 'High'}}

plt.figure(figsize=(12, 8))
for segment, settings in segment_settings.items():
    subset = adv_graph[adv_graph['Segment'] == segment]
    plt.scatter(subset['OverallScore'], subset['m6_Revenue'],
        c=settings['color'], s=settings['size'],
        alpha=settings['alpha'], label=settings['label'],
        edgecolor='black', linewidth=0.5)

plt.title('6-Month Customer Lifetime Value by RFM Score', fontsize=16)
plt.xlabel('RFM Score (OverallScore)', fontsize=14)
plt.ylabel('6-Month Revenue (m6_Revenue)', fontsize=14)
plt.legend(title='Customer Segment')
plt.grid(True)
plt.tight_layout()
plt.show()

adv_merge = adv_merge[adv_merge['m6_Revenue']<adv_merge['m6_Revenue'].quantile(0.99)]

adv_merge.head()

kmeans = KMeans(n_clusters=3)
adv_merge['LTVCluster'] = kmeans.fit_predict(adv_merge[['m6_Revenue']])

adv_merge.head()

adv_merge = order_cluster('LTVCluster', 'm6_Revenue',adv_merge,True)

adv_cluster = adv_merge.copy()
adv_cluster.groupby('LTVCluster')['m6_Revenue'].describe()

adv_cluster.head()

adv_class = pd.get_dummies(adv_cluster)
adv_class.head()

corr_matrix = adv_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)

X = adv_class.drop(['LTVCluster','m6_Revenue'],axis=1)
y = adv_class['LTVCluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

y_pred = ltv_xgb_model.predict(X_test)

print(classification_report(y_test, y_pred))

adv_merge.to_excel("adv_merge_output.xlsx", index=False)

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(ltv_xgb_model, f)
