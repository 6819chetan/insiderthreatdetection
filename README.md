# insiderthreatdetection
<B>Insider Threat Risk:<B> <br>Insider threats are critical risks, originating from individuals within the organization who have legitimate access, potentially leading to data breaches, financial losses, and reputational damage.<br>

Objective: <br>The goal of this study is to classify user activities as either malicious or non-malicious using advanced machine learning models on an extracted master file (CSV format) from the CERT r4.2 dataset.
<br>
Advanced ML Models:<br> CNN, GAN, and DNN models were used to classify insider threats effectively by analyzing user behavior in the CSV file.
<br>
Dataset Characteristics: <br>The CERT r4.2 dataset, organized in the master CSV file, contains a significant class imbalance, with malicious instances far fewer than normal activities.
<br>
Handling Class Imbalance with SMOTE: <br>To address this imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to generate synthetic samples of the minority (malicious) class, enhancing the training data.
<br>
DNN for Classification:<br> The DNN model was structured to capture complex patterns in user activities, improving classification accuracy between malicious and non-malicious users.
<br>
GAN for Data Augmentation: <br>GANs were used to generate realistic synthetic data samples, diversifying the training set and further balancing the classes.
<br>
CNN for Sequential Pattern Recognition: <br>The CNN model extracted spatial and temporal features from user activity data in the CSV file, aiding in identifying subtle patterns associated with insider threats.
<br>
Improved Detection Accuracy: <br>By leveraging CNN, GAN, and DNN models together, the approach achieved enhanced accuracy in distinguishing between malicious and non-malicious activities within the imbalanced dataset.
