"""HELP International have been able to raise around $ 10 million. 
Now the CEO of the NGO needs to decide how to use this money strategically and effectively. 
So, CEO has to make decision to choose the countries that are in the direst need of aid. 
Hence, your Job as a Data scientist is to categorise the countries using 
some socio-economic and health factors that determine the overall development of the country. 
Then you need to suggest the countries which the CEO needs to focus on the most."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
money=1e7
data=pd.read_csv("C:/Users/Prekshita/Downloads/Country-data.csv")
dataset=data.drop(columns='country',axis=1,inplace=False)

kmeans=KMeans(n_clusters=3,n_init=10,max_iter=500,random_state=42)
data['cluster']=kmeans.fit_predict(dataset)

pca=PCA(n_components=0.95)
d=pca.fit_transform(dataset)
print("PCA Components:",pca.n_components_) 
print("Silhouette Score:",silhouette_score(dataset,data['cluster'])) #0.7
loadings = pd.DataFrame(pca.components_, columns=dataset.columns)
for i in loadings:
    num=loadings[i]
    mask = num.abs().between(0.3, 0.8)
    if mask.any():
        print(i) 

plt.scatter(data["income"],data["gdpp"],c=data['cluster'],cmap='viridis',s=50,alpha=0.7)
plt.xlabel('Income')
plt.ylabel('GDPP')
plt.title('CLuster Visualization')
plt.colorbar(label='cluster')
plt.close()

cluster_profiles = data.groupby('cluster').mean(numeric_only=True)
neediest_cluster = cluster_profiles['income'].idxmin()  # or based on other indicator
countries_in_need = data[data['cluster']==neediest_cluster]['country'].tolist()

poor_countries=data[data['country'].isin(countries_in_need)]

poor_countries = poor_countries.copy()
poor_countries['life_expec_range']=pd.cut(poor_countries['life_expec'],bins=5)
living_profile=poor_countries.groupby('life_expec_range',observed=True)

for i in poor_countries['life_expec_range'].unique():
    bins=i
    midpoint=(bins.left+bins.right)/2
    poor_countries.loc[poor_countries['life_expec_range']==bins,'life_expec_range_midpoint']=midpoint
    countries_list=poor_countries[poor_countries['life_expec_range_midpoint']==midpoint]['country'].tolist()
    
print(poor_countries['life_expec_range'].value_counts())
poor_countries_dataset=poor_countries.drop(columns=['country','life_expec_range'],axis=1,inplace=False)
poor_countries['segmentation']=kmeans.fit_predict(poor_countries_dataset)
poor_countries.drop(columns='cluster',axis=1,inplace=True)

cluster_profiles = poor_countries.groupby('segmentation').mean(numeric_only=True)
print(cluster_profiles)

plt.scatter(poor_countries["income"],poor_countries["gdpp"],c=poor_countries['segmentation'],cmap='viridis',s=50,alpha=0.7)
plt.xlabel('Income')
plt.ylabel('GDPP')
plt.title('CLuster Visualization')
plt.colorbar(label='cluster')
plt.close()

high_poverty_list=poor_countries[poor_countries['segmentation']==1]['country'].tolist()
high_poverty=poor_countries[poor_countries['country'].isin(high_poverty_list)].copy()
print("Number of poorest countries in urgent need:",high_poverty['country'].count())
moderate_poverty_list=poor_countries[poor_countries['segmentation']==2]['country'].tolist()
moderate_poverty=poor_countries[poor_countries['country'].isin(moderate_poverty_list)].copy()
print("Number of quite poor countries in somewhat urgent need:",moderate_poverty['country'].count())
lower_poverty_list=poor_countries[poor_countries['segmentation']==0]['country'].tolist()
lower_poverty=poor_countries[poor_countries['country'].isin(lower_poverty_list)].copy()
print("Number of relatively less poor countries in not-so urgent need:",lower_poverty['country'].count())
print()

high_poverty.loc[:,'economy']=high_poverty[['gdpp','income']].mean(axis=1)
high_poverty.loc[:,'Trade to GDPP']=(high_poverty[['imports','exports']].mean(axis=1))/high_poverty['gdpp']
moderate_poverty.loc[:,'economy']=moderate_poverty[['gdpp','income']].mean(axis=1)
moderate_poverty.loc[:,'Trade to GDPP']=(moderate_poverty[['imports','exports']].mean(axis=1))/moderate_poverty['gdpp']
lower_poverty.loc[:,'economy']=lower_poverty[['gdpp','income']].mean(axis=1)
lower_poverty.loc[:,'Trade to GDPP']=(lower_poverty[['imports','exports']].mean(axis=1))/lower_poverty['gdpp']

def plotting(df):
    plt.scatter(df["economy"],df["Trade to GDPP"],c=df['segmentation'],cmap='viridis',s=50,alpha=0.7)
    plt.xlabel('Economy')
    plt.ylabel('Trade to GDPP')
    plt.title('CLuster Visualization')
    plt.colorbar(label='cluster')
    plt.show()

plotting(lower_poverty)


def threshold(df,column):
    threshold=df[column].quantile(0.5)
    return threshold


inflation_threshold=threshold(poor_countries,'inflation')

def report_high(df,column,threshold):
    record=df[df[column]>threshold].iloc[:,:]
    return record
def report_low(df,column,threshold):
    record=df[df[column]<threshold].iloc[:,:]
    return record

groups = {
    'high_poverty': high_poverty,
    'moderate_poverty': moderate_poverty,
    'lower_poverty': lower_poverty
}

for name, i in groups.items():
    t=threshold(poor_countries,'child_mort')
    print(f"Number of countries with high child mortality in {name}: ",report_high(i,'child_mort',t)['country'].count())
    t=threshold(poor_countries,'inflation')
    print(f"Number of countries with high inflation in {name}: ",report_high(i,'inflation',t)['country'].count())
    print()

for name, i in groups.items():
    t=threshold(i,'total_fer')
    print(f"Number of countries with low fertility in {name}: ",report_low(i,'total_fer',t)['country'].count())
    t=threshold(i,'health')
    print(f"Number of countries with low federal health spending in {name}: ",report_low(i,'health',t)['country'].count())
    print()

