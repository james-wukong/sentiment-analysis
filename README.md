## Pipeline

Step 1: Description

Step 2: Exploratory Data Analysis and Preprocessing

Step 3: Load the Pretrained Model

Step 4: Prepare Tensor Dataset for Training, Validation and Testing

Step 5: Setting up BERT Pretrained Model

Step 6: Setting Up Optimizer, Loss, and Metrics

Step 7: Train the Extended NN Model

Step 8: Evaluate the New Model with Testing Dataset

Step 9: Fine-tuning hyperparameters and NN Model to improve performance

Step 10: Make Predictions

### 1. Description

This project is to analyze twitter comments and classify them into categories. The data source is from [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset/data), under license 'CC BY-NC-SA 4.0 DEED' which means we are free to share and adapt this dataset.

We are going to build the model with pre-trained model 'bert-base-uncased'.

According to the description of dataset, we have the following explanation for category column:

- 0 Indicating it is a Neutral Tweet/Comment
- 1 Indicating a Postive Sentiment
- -1 Indicating a Negative Tweet/Comment

However, we need to convert the category values into {0: 'Neutral', 1: 'Positive', 2: 'Negative'} to fit the model

### 2. Exploratory Data Analysis and Preprocessing

Explore dataset by doing:

1. checking null values and choose either imputing or dropping
2. add text_length feature and check the distribution of text length
3. check the distribution of target label

Pre-processing:

1. Clean the text content by removing hyperlinks, HTML tags, and stopwords, etc


### 3. Load the Pretrained Model

1. Load "bert-base-uncased" as the base model 
2. Load its tokenizer to tokenize the dataset

### 4. Prepare Tensor Dataset for Training, Validation and Testing

1. Split dataset into training, validation, and testing datasets
2. Tokenize text feature in each dataset
3. Load datasets into tensor dataset by using from_tensor_slice
4. Create batches for each dataset, and shuffle training dataset

### 5. Setting up BERT Pretrained Model

1. Setting pretrained model using TFBertForSequenceClassification
2. Get BertConfig from pretrained model
3. Freeze all layers and remove the last 2 layer from pretrained model (dropout layer and output layer)

### 6. Setting Up Optimizer, Loss, and Metrics

1. Compile the new model, with optimizer Adam, loss SparseCategoricalCrossEntropy, and metrics accuracy

### 7. Train the Extended NN Model

1. Add LSTM, BatchNormalization and Dropout layer blocks to pretrained model
2. Add output Dense layer with 3 units(category values) and activation function 'softmax'
3. Train the new model with training dataset, include callbacks: EarlyStoppingAtMinLoss and CustomCheckpoint

### 8. Evaluate the New Model with Testing Dataset

1. Evaluate the new model by checking the result returned by custom_model.evaluate(test_ds)
2. Plot Learning Curve of loss and accuracy during training
3. Compare the predictions and actual values
4. Evaluating with confusion matrix

### 9. Fine-tuning hyperparameters and NN Model to improve performance

1. Update hyperparameters, such as learning rate, batch_size, to improve model's performance
2. Add/Remove NN layers to improve the performance
3. Repeat until get better result

### 10. Make Predictions

1. Tokenize a random sentence so that we can use it in the custom model
2. Check the result provided by the new model
3. Repeat several times to check the performance of model
