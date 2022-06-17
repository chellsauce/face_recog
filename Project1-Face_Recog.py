from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti = fetch_olivetti_faces()

# there are 400 images - 10x40 (40 people - 1 person @ 10 images) - 1 image = 64x64
features = olivetti.data # the data contains pixel intensity that has been normalized
# represent target variables (people) w/ integers (face ids)
targets = olivetti.target

### Understanding the dataset of images
# the size of variable 'features' is 400x4096, which is huge dimensions
# print(features.shape)

## to show the image of every user_id (line 24-32)
#fig, sub_plot = plt.subplots(nrows=5, ncols=8, figsize=(14, 8)) # empty plot
#sub_plot = sub_plot.flatten()

#for user_id in np.unique(targets):
#    index = user_id * 8
#    sub_plot[user_id].imshow(features[index].reshape(64, 64), cmap='gray')
#    sub_plot[user_id].set_xticks([])
#    sub_plot[user_id].set_yticks([])
#    sub_plot[user_id].set_title('Face id : %s' % user_id)

#plt.suptitle('the dataset')
#plt.show()

## to show the images of every user_id

#fig, sub_plot = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))

#for j in range(10):
#    sub_plot[j].imshow(features[j].reshape(64, 64), cmap='gray')
#    sub_plot[j].set_xticks([])
#    sub_plot[j].set_yticks([])
#    sub_plot[j].set_title('Face id = 0')

#plt.show()

### Understanding eigen vectors

## split the original dataset
X_train, X_test, y_train,  y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

### reduce the amount of dimensions w/ PCA

## find optimal number of eigenvectors (PCs)
#pca = PCA()
#pca_fit = pca.fit(features)

#plt.figure(1, figsize=(12, 8))
#plt.plot(pca_fit.explained_variance_, linewidth = 2)
#plt.xlabel('Components')
#plt.ylabel('Explained Variances')
#plt.show()

## it is shown that the number of optimal PCs is 100

pca = PCA(n_components=100, whiten=True) # whiten is to increase the accuracy of principal components
pca.fit(X_train)
X_pca = pca.fit_transform(features)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

### Check the eigenvectors ('eigenfaces') - 1 PC (eigenvector) has 4096 features

#n_eigenfaces = len(pca.components_)
#eigenfaces = pca.components_.reshape((n_eigenfaces, 64, 64))

#fig, sub_plot = plt.subplots(nrows=10, ncols=10, figsize=(18, 18))
#sub_plot = sub_plot.flatten()

#for i in range(n_eigenfaces):
#    sub_plot[i].imshow(eigenfaces[i], cmap='gray')
#    sub_plot[i].set_xticks([])
#    sub_plot[i].set_yticks([])

#plt.suptitle('Eigenfaces')
#plt.show()

## from here we can see the effect of dimensional reduction of PCs features

### Constructing Machine Learning Models and find out the acc score of each models

models = [('Logistic Regression', LogisticRegression()), ('Support Vector Machine', SVC()), ('Naive Bayes', GaussianNB())]

## find the accuracy score of the models
#for name, model in models:
#    classifier_models = model
#    classifier_models.fit(X_train_pca, y_train)
#    y_pred = classifier_models.predict(X_test_pca)
#    print('Results with %s' % name)
#    print('Accuracy Score %s' % (metrics.accuracy_score(y_test, y_pred)))

### Using Cross Validation
for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_score = cross_val_score(model, X_pca, targets, cv=kfold)
    print('Results with %s' % name)
    print('Mean of the cross validation scores: %s' %cv_score.mean())

