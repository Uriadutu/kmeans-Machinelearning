import numpy as np
import pandas as pd  
import os


for dirname, _, filenames in os.walk('./csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

date_format = '%d-%m-%Y %H:%M'
df = pd.read_csv('./csv/OnlineRetail.csv', encoding="ISO-8859-1", parse_dates=['InvoiceDate'], date_format=date_format)
df.head()

online = df.copy()

from datetime import datetime

def get_month(x):
    return datetime(x.year, x.month, 1)

online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month)
grouping = online.groupby('CustomerID')['InvoiceMonth']
online['CohortMonth'] = grouping.transform('min')
online.head()

def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

invoice_year, invoice_month, _ = get_date_int(online, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date_int(online, 'CohortMonth')
years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month
online['CohortIndex'] = years_diff * 12 + months_diff + 1
online.head()

grouping = online.groupby(['CohortMonth','CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()
cohort_counts = cohort_data.pivot(
    index='CohortMonth',
    columns='CohortIndex',
    values='CustomerID'
)
cohort_counts.head()

cohort_sizes = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_sizes, axis=0)
retention.round(3)*100

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
plt.title('Retention rates')
sns.heatmap(data=retention,
           annot= True,
           fmt='.0%',
           vmax=0.5,
           vmin=0.0,
           cmap='Blues')
plt.show()

snapshot_date = max(online['InvoiceDate']) + pd.Timedelta(days=1)
online['TotalSum'] = online['Quantity'] * online['UnitPrice']

datamart = online.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date-x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'
})

datamart.rename(columns = 
                {'InvoiceDate': 'Recency',
                 'InvoiceNo': 'Frequency',
                 'TotalSum': 'MonetaryValue'}, inplace=True)

datamart.head()

r_labels = range(4, 0, -1)
f_labels = range(1,5)
m_labels = range(1,5)

r_quartiles = pd.qcut(datamart['Recency'], 4, labels = r_labels)
f_quartiles = pd.qcut(datamart['Frequency'], 4, labels = f_labels)
m_quartiles = pd.qcut(datamart['MonetaryValue'], 4, labels = m_labels)

datamart1 = datamart.assign(
    R=r_quartiles.values, 
    F=f_quartiles.values, 
    M=m_quartiles.values
)

def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
    
datamart1['RFM_Segment'] = datamart1.apply(join_rfm, axis=1)
datamart1['RFM_Score'] = datamart1[['R','F','M']].sum(axis=1)

def segment_me(df):
    if df['RFM_Score']>=9:
        return 'Gold'
    elif (df['RFM_Score']>=5) and (df['RFM_Score']<9):
        return 'Silver'
    else:
        return 'Bronze'
    
datamart1['General_Segment'] = datamart1.apply(segment_me,axis=1)

datamart_agg = datamart1.groupby('General_Segment').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'MonetaryValue':['mean','count']
    }).round(1)

datamart_agg

datamart.describe()

datamart['MonetaryValue'] = np.maximum(datamart['MonetaryValue'], 0)
datamart['MonetaryValue'].min()


from sklearn.preprocessing import StandardScaler

small_constant = 1e-10
datamart_log = np.log(datamart+small_constant)

scaler = StandardScaler()
datamart_normalized = scaler.fit_transform(datamart_log)

print('mean: ', datamart_normalized.mean(axis=0).round(2))
print('std: ', datamart_normalized.std(axis=0).round(2))


from sklearn.cluster import KMeans

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(datamart_normalized)
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(datamart_normalized)
cluster_labels = kmeans.labels_

datamart2 = datamart.assign(Cluster = cluster_labels)

datamart_agg2 = datamart2.groupby('Cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'MonetaryValue':['mean','count']
    }).round(0)

datamart_agg2

datamart_normalized = pd.DataFrame(datamart_normalized,                                    
                                   index=datamart.index,
                                   columns=datamart.columns)

datamart_normalized['Cluster'] = datamart2['Cluster']

datamart_melt = pd.melt(datamart_normalized.reset_index(),
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                        var_name='Attribute',
                        value_name='Value')

sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=datamart_melt)\
.set_title('Snake plot of standardized variables')

plt.show()
