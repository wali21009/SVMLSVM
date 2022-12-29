# Import library yang dibutuhkan
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Load data Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Buat objek LS-SVM dan lakukan pelatihan
lsvm = LinearSVC(C=1.0, random_state=42)
lsvm.fit(X_train, y_train)

# Prediksi kelas pada data uji
y_pred = lsvm.predict(X_test)

# Hitung akurasi prediksi
accuracy = (y_pred == y_test).mean()
print("Akurasi prediksi: {:.2f}".format(accuracy))