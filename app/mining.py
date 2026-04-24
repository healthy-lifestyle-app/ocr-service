import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# CSV oku
df = pd.read_csv('data/food_data.csv')

print("RAW DATA:")
print(df)

# Eksik verileri temizle
df = df.dropna()

print("\nCLEAN DATA:")
print(df)

# Feature seç (modelin göreceği veriler)
X = df[['calories', 'protein', 'carb', 'sugar', 'fat']]

# Normalize (çok önemli!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans modeli
kmeans = KMeans(n_clusters=3, random_state=42)

# Cluster ata
df['cluster'] = kmeans.fit_predict(X_scaled)

print("\nCLUSTER SONUCU:")
print(df)

# Grafik
plt.scatter(df['sugar'], df['calories'], c=df['cluster'])
plt.xlabel('Sugar')
plt.ylabel('Calories')
plt.title('Food Clustering')
plt.show()