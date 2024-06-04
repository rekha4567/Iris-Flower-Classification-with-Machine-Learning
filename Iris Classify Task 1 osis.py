#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[8]:


from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[3]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the features of the training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


from sklearn.svm import SVC

# Initialize the SVM model with a linear kernel
svm_model = SVC(kernel='linear')

# Train the SVM model using the scaled training data
svm_model.fit(X_train_scaled, y_train)


# In[6]:


from sklearn.metrics import accuracy_score

# Make predictions on the scaled testing data
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[7]:


# Example of predicting the species for new data
new_data = [[5.1, 3.5, 1.4, 0.2],  # Example measurements of an iris flower
            [6.3, 3.3, 4.7, 1.6],  # Another example measurements of an iris flower
            [6.4, 2.8, 5.6, 2.2]]  # And one more example measurements of an iris flower

# Scale the new data using the same scaler used for training data
new_data_scaled = scaler.transform(new_data)

# Make predictions on the new data
predictions = svm_model.predict(new_data_scaled)

# Convert numeric predictions back to species names
species_names = iris.target_names
predicted_species = [species_names[prediction] for prediction in predictions]

print("Predicted species for new data:")
print(predicted_species)


# In[ ]:




