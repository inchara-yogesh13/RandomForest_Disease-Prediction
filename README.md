Disease Prediction and Medicine Recommendation System using Random Forest

Project Description

The Disease Prediction and Medicine Recommendation System is a machine learning project that predicts a patient's disease based on the symptoms they enter. The system is trained using a healthcare dataset containing various symptoms, corresponding diseases, and recommended medicines. It uses the Random Forest Classification algorithm, an ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

During the training phase, the model learns the relationship between symptom combinations and their associated diseases. The dataset is divided into training and testing sets to evaluate the model's performance, and metrics such as accuracy, classification report, and confusion matrix are used to assess its effectiveness. After training, the model is saved so it can be reused without retraining.

When a user enters one or more symptoms, the system converts the input into the same format used during training and predicts the most likely disease. After identifying the disease, it retrieves the corresponding medicine from the dataset and displays it along with the prediction. This provides users with a quick indication of a possible illness and its associated medication.

The project demonstrates the practical application of machine learning in healthcare, showing how classification algorithms can assist in early disease prediction and support decision-making. While the system is intended for educational purposes and should not replace professional medical advice, it highlights the potential of AI in improving healthcare accessibility and efficiency.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Machine Learning Algorithm
Random Forest Classifier
Project Workflow
Load the healthcare dataset.
Preprocess the data by separating symptoms (features) and diseases (target).
Split the dataset into training and testing sets.
Train the Random Forest classifier.
Evaluate the model using accuracy and other performance metrics.
Save the trained model for future use.
Accept user-entered symptoms as input.
Predict the most probable disease.
Retrieve and display the recommended medicine corresponding to the predicted disease.
Key Features
Predicts diseases based on multiple symptoms.
Uses Random Forest for robust and accurate classification.
Recommends medicine associated with the predicted disease.
Saves the trained model for efficient future predictions.
Simple command-line interface that can be extended into a web or desktop application.
