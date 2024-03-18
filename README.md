# Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow
## Part I. The Fundamentals of Machine Learning
1.	The Machine Learning Landscape
2.	End-to-End Machine Learning Project
3.	Classification
4.	Training Models
5.	Support Vector Machines
6.	Decision Trees
7.	Ensemble Learning and Random Forests
8.	Dimensionality Reduction
9.	Unsupervised Learning Techniques
## Part II. Neural Networks and Deep Learning
10.	Introduction to Artificial Neural Networks with Keras
11.	Training Deep Neural Networks
12.	Custom Models and Training with TensorFlow
13.	Loading and Preprocessing Data with TensorFlow
14.	Deep Computer Vision Using Convolutional Neural Networks
15.	Processing Sequences Using RNNs and CNNs
16.	Natural Language Processing with RNNs and Attention
17.	Representation Learning and Generative Learning Using Autoencoders and GANs
18.	Reinforcement Learning
19.	Training and Deploying TensorFlow Models at Scale

## Part I. The Fundamentals of Machine Learning (1-275)
### 1.	The Machine Learning Landscape (1-33)
- 1.1 What Is Machine Learning?
- 1.2 Why Use Machine Learning?
- 1.3 Examples of Applications
- 1.4 Types of Machine Learning Systems
- 1.4.1 Supervised/Unsupervised Learning
- 1.4.2 Batch and Online Learning
- 1.4.3 Instance-Based Versus Model-Based Learning
- 1.5 Main Challenges of Machine Learning
- 1.5.1 Insufficient Quantity of Training Data
- 1.5.2 Nonrepresentative Training Data
- 1.5.3 Poor-Quality Data
- 1.5.4 Irrelevant Features
- 1.5.5 Overfitting the Training Data
- 1.5.6 Underfitting the Training Data
- 1.5.7 Stepping Back
- 1.6 Testing and Validating
- 1.6.1 Hyperparameter Tuning and Model Selection
- 1.6.2 Data Mismatch
- 1.7 Exercises

### 2.	End-to-End Machine Learning Project (35-84)
2.1 Working with Real Data
2.2 Look at the Big Picture
2.2.1 Frame the Problem
2.2.2 Select a Performance Measure
2.2.3 Check the Assumptions
2.3 Get the Data
2.3.1 Create the Workspace
2.3.2 Download the Data
2.3.3 Take a Quick Look at the Data Structure
2.3.4 Create a Test Set
2.4 Discover and Visualize the Data to Gain Insights
2.4.1 Visualizing Geographical Data
2.4.2 Looking for Correlations
2.4.3 Experimenting with Attribute Combinations
2.5 Prepare the Data for Machine Learning Algorithms
2.5.1 Data Cleaning
2.5.2 Handling Text and Categorical Attributes
2.5.3 Custom Transformers
2.5.4 Feature Scaling
2.5.5 Transformation Pipelines
2.6 Select and Train a Model
2.6.1 Training and Evaluating on the Training Set
2.6.2 Better Evaluation Using Cross-Validation
2.7 Fine-Tune Your Model
2.7.1 Grid Search
2.7.2 Randomized Search
2.7.3 Ensemble Methods
2.7.4 Analyze the Best Models and Their Errors
2.7.5 Evaluate Your System on the Test Set
2.8 Launch, Monitor, and Maintain Your System
2.9 Try It Out!
2.10 Exercises

### 3.	Classification (85-108)
3.1 MNIST
3.2 Training a Binary Classifier
3.3 Performance Measures
3.3.1 Measuring Accuracy Using Cross-Validation
3.3.2 Confusion Matrix
3.3.3 Precision and Recall
3.3.4 Precision/Recall Trade-off
3.3.5 The ROC Curve
3.4 Multiclass Classification
3.5 Error Analysis
3.6 Multilabel Classification
3.7 Multioutput Classification
3.8 Exercises

### 4.	Training Models (111-151)
4.1 Linear Regression
4.1.1 The Normal Equation
4.1.2 Computational Complexity
4.2 Gradient Descent
4.2.1 Batch Gradient Descent
4.2.2 Stochastic Gradient Descent
4.2.3 Mini-batch Gradient Descent
4.3 Polynomial Regression
4.4 Learning Curves
4.5 Regularized Linear Models
4.5.1 Ridge Regression
4.5.2 Lasso Regression
4.5.3 Elastic Net
4.5.4 Early Stopping
4.6 Logistic Regression
4.6.1 Estimating Probabilities
4.6.2 Training and Cost Function
4.6.3 Decision Boundaries
4.6.4 Softmax Regression
4.7 Exercises

### 5.	Support Vector Machines (153-174)
5.1 Linear SVM Classification
5.1.1 Soft Margin Classification
5.2 Nonlinear SVM Classification
5.2.1 Polynomial Kernel
5.2.2 Similarity Features
5.2.3 Gaussian RBF Kernel
5.2.4 Computational Complexity
5.3 SVM Regression
5.4 Under the Hood
5.4.1 Decision Function and Predictions
5.4.2 Training Objective
5.4.3 Quadratic Programming
5.4.4 The Dual Problem
5.4.5 Kernelized SVMs
5.4.6 Online SVMs
5.5 Exercises

### 6.	Decision Trees (175-186)
6.1 Training and Visualizing a Decision Tree
6.2 Making Predictions
6.3 Estimating Class Probabilities
6.4 The CART Training Algorithm
6.5 Computational Complexity
6.6 Gini Impurity or Entropy?
6.7 Regularization Hyperparameters
6.8 Regression
6.9 Instability
6.10 Exercises

### 7.	Ensemble Learning and Random Forests (189-211)
7.1 Voting Classifiers
7.2 Bagging and Pasting
7.2.1 Bagging and Pasting in Scikit-Learn
7.2.2 Out-of-Bag Evaluation
7.3 Random Patches and Random Subspaces
7.4 Random Forests
7.4.1 Extra-Trees
7.4.2 Feature Importance
7.5 Boosting
7.5.1 AdaBoost
7.5.2 Gradient Boosting
7.6 Stacking
7.7 Exercises

### 8.	Dimensionality Reduction (213-233)
8.1 The Curse of Dimensionality
8.2 Main Approaches for Dimensionality Reduction
8.2.1 Projection
8.2.2 Manifold Learning
8.3 PCA
8.3.1 Preserving the Variance
8.3.2 Principal Components
8.3.3 Projecting Down to d Dimensions
8.3.4 Using Scikit-Learn
8.3.5 Explained Variance Ratio
8.3.6 Choosing the Right Number of Dimensions
8.3.7 PCA for Compression
8.3.8 Randomized PCA
8.3.9 Incremental PCA
8.4 Kernel PCA
8.4.1 Selecting a Kernel and Tuning Hyperparameters
8.5 LLE
8.6 Other Dimensionality Reduction Techniques
8.7 Exercises

### 9.	Unsupervised Learning Techniques (235-275)
9.1 Clustering
9.1.1 K-Means
9.1.2 Limits of K-Means
9.1.3 Using Clustering for Image Segmentation
9.1.4 Using Clustering for Preprocessing
9.1.5 Using Clustering for Semi-Supervised Learning
9.1.6 DBSCAN
9.1.7 Other Clustering Algorithms
9.2 Gaussian Mixtures
9.2.1 Anomaly Detection Using Gaussian Mixtures
9.2.2 Selecting the Number of Clusters
9.2.3 Bayesian Gaussian Mixture Models
9.2.4 Other Algorithms for Anomaly and Novelty Detection
9.3 Exercises

## Part II. Neural Networks and Deep Learning (279-718)
### 10.	Introduction to Artificial Neural Networks with Keras (279-327)
10.1 From Biological to Artificial Neurons
10.1.1 Biological Neurons
10.1.2 Logical Computations with Neurons
10.1.3 The Perceptron
10.1.4 The Multilayer Perceptron and Backpropagation
10.1.5 Regression MLPs
10.1.6 Classification MLPs
10.2 Implementing MLPs with Keras
10.2.1 Installing TensorFlow 2
10.2.2 Building an Image Classifier Using the Sequential API
10.2.3 Building a Regression MLP Using the Sequential API
10.2.4 Building Complex Models Using the Functional API
10.2.5 Using the Subclassing API to Build Dynamic Models
10.2.6 Saving and Restoring a Model
10.2.7 Using Callbacks
10.2.8 Using TensorBoard for Visualization
10.3 Fine-Tuning Neural Network Hyperparameters
10.3.1 Number of Hidden Layers
10.3.2 Number of Neurons per Hidden Layer
10.3.3 Learning Rate, Batch Size, and Other Hyperparameters
10.4 Exercises

### 11.	Training Deep Neural Networks (331-373)
11.1 The Vanishing/Exploding Gradients Problems
11.1.1 Glorot and He Initialization
11.1.2 Nonsaturating Activation Functions
11.1.3 Batch Normalization
11.1.4 Gradient Clipping
11.2 Reusing Pretrained Layers
11.2.1 Transfer Learning with Keras
11.2.2 Unsupervised Pretraining
11.2.3 Pretraining on an Auxiliary Task
11.3 Faster Optimizers
11.3.1 Momentum Optimization
11.3.2 Nesterov Accelerated Gradient
11.3.3 AdaGrad
11.3.4 RMSProp
11.3.5 Adam and Nadam Optimization
11.3.6 Learning Rate Scheduling
11.4 Avoiding Overfitting Through Regularization
11.4.1 ℓ1 and ℓ2 Regularization
11.4.2 Dropout
11.4.3 Monte Carlo (MC) Dropout
11.4.4 Max-Norm Regularization
11.5 Summary and Practical Guidelines
11.6 Exercises

### 12.	Custom Models and Training with TensorFlow (375-410)
12.1 A Quick Tour of TensorFlow
12.2 Using TensorFlow like NumPy
12.2.1 Tensors and Operations
12.2.2 Tensors and NumPy
12.2.3 Type Conversions
12.2.4 Variables
12.2.5 Other Data Structures
12.3 Customizing Models and Training Algorithms
12.3.1 Custom Loss Functions
12.3.2 Saving and Loading Models That Contain Custom Components
12.3.3 Custom Activation Functions, Initializers, Regularizers, and Constraints
12.3.4 Custom Metrics
12.3.5 Custom Layers
12.3.6 Custom Models
12.3.7 Losses and Metrics Based on Model Internals
12.3.8 Computing Gradients Using Autodiff
12.3.9 Custom Training Loops
12.4 TensorFlow Functions and Graphs
12.4.1 AutoGraph and Tracing
12.4.2 TF Function Rules
12.5 Exercises

### 13.	Loading and Preprocessing Data with TensorFlow (413-442)
13.1 The Data API
13.1.1 Chaining Transformations
13.1.2 Shuffling the Data
13.1.3 Preprocessing the Data
13.1.4 Putting Everything Together
13.1.5 Prefetching
13.1.6 Using the Dataset with tf.keras
13.2 The TFRecord Format
13.2.1 Compressed TFRecord Files
13.2.2 A Brief Introduction to Protocol Buffers
13.2.3 TensorFlow Protobufs
13.2.4 Loading and Parsing Examples
13.2.5 Handling Lists of Lists Using the SequenceExample Protobuf
13.3 Preprocessing the Input Features
13.3.1 Encoding Categorical Features Using One-Hot Vectors
13.3.2 Encoding Categorical Features Using Embeddings
13.3.3 Keras Preprocessing Layers
13.4 TF Transform
13.5 The TensorFlow Datasets (TFDS) Project
13.6 Exercises

### 14.	Deep Computer Vision Using Convolutional Neural Networks (445-496)
14.1 The Architecture of the Visual Cortex
14.2 Convolutional Layers
14.2.1 Filters
14.2.2 Stacking Multiple Feature Maps
14.2.3 TensorFlow Implementation
14.2.4 Memory Requirements
14.3 Pooling Layers
14.3.1 TensorFlow Implementation
14.4 CNN Architectures
14.4.1 LeNet-5
14.4.2 AlexNet
14.4.3 GoogLeNet
14.4.4 VGGNet
14.4.5 ResNet
14.4.6 Xception
14.4.7 SENet
14.5 Implementing a ResNet-34 CNN Using Keras
14.6 Using Pretrained Models from Keras
14.7 Pretrained Models for Transfer Learning
14.8 Classification and Localization
14.9 Object Detection
14.9.1 Fully Convolutional Networks
14.9.2 You Only Look Once (YOLO)
14.10 Semantic Segmentation
14.11 Exercises

### 15.	Processing Sequences Using RNNs and CNNs (497-523)
15.1 Recurrent Neurons and Layers
15.1.1 Memory Cells
15.1.2 Input and Output Sequences
15.2 Training RNNs
15.3 Forecasting a Time Series
15.3.1 Baseline Metrics
15.3.2 Implementing a Simple RNN
15.3.3 Deep RNNs
15.3.4 Forecasting Several Time Steps Ahead
15.4 Handling Long Sequences
15.4.1 Fighting the Unstable Gradients Problem
15.4.2 Tackling the Short-Term Memory Problem
15.5 Exercises

### 16.	Natural Language Processing with RNNs and Attention (525-565)
16.1 Generating Shakespearean Text Using a Character RNN
16.1.1 Creating the Training Dataset
16.1.2 How to Split a Sequential Dataset
16.1.3 Chopping the Sequential Dataset into Multiple Windows
16.1.4 Building and Training the Char-RNN Model
16.1.5 Using the Char-RNN Model
16.1.6 Generating Fake Shakespearean Text
16.1.7 Stateful RNN
16.2 Sentiment Analysis
16.2.1 Masking
16.2.2 Reusing Pretrained Embeddings
16.3 An Encoder–Decoder Network for Neural Machine Translation
16.3.1 Bidirectional RNNs
16.3.2 Beam Search
16.4 Attention Mechanisms
16.4.1 Visual Attention
16.4.2 Attention Is All You Need: The Transformer Architecture
16.5 Recent Innovations in Language Models
16.6 Exercises

### 17.	Representation Learning and Generative Learning Using Autoencoders and GANs (567-607)
17.1 Efficient Data Representations
17.2 Performing PCA with an Undercomplete Linear Autoencoder
17.3 Stacked Autoencoders
17.3.1 Implementing a Stacked Autoencoder Using Keras
17.3.2 Visualizing the Reconstructions
17.3.3 Visualizing the Fashion MNIST Dataset
17.3.4 Unsupervised Pretraining Using Stacked Autoencoders
17.3.5 Tying Weights
17.3.6 Training One Autoencoder at a Time
17.4 Convolutional Autoencoders
17.5 Recurrent Autoencoders
17.6 Denoising Autoencoders
17.7 Sparse Autoencoders
17.8 Variational Autoencoders
17.8.1 Generating Fashion MNIST Images
17.9 Generative Adversarial Networks
17.9.1 The Difficulties of Training GANs
17.9.2 Deep Convolutional GANs
17.9.3 Progressive Growing of GANs
17.9.4 StyleGANs
17.10 Exercises

### 18.	Reinforcement Learning (609-664)
18.1 Learning to Optimize Rewards
18.2 Policy Search
18.3 Introduction to OpenAI Gym
18.4 Neural Network Policies
18.5 Evaluating Actions: The Credit Assignment Problem
18.6 Policy Gradients
18.7 Markov Decision Processes
18.8 Temporal Difference Learning
18.9 Q-Learning
18.9.1 Exploration Policies
18.9.2 Approximate Q-Learning and Deep Q-Learning
18.10 Implementing Deep Q-Learning
18.11 Deep Q-Learning Variants
18.11.1 Fixed Q-Value Targets
18.11.2 Double DQN
18.11.3 Prioritized Experience Replay
18.11.4 Dueling DQN
18.12 The TF-Agents Library
18.12.1 Installing TF-Agents
18.12.2 TF-Agents Environments
18.12.3 Environment Specifications
18.12.4 Environment Wrappers and Atari Preprocessing
18.12.5 Training Architecture
18.12.6 Creating the Deep Q-Network
18.12.7 Creating the DQN Agent
18.12.8 Creating the Replay Buffer and the Corresponding Observer
18.12.9 Creating Training Metrics
18.12.10 Creating the Collect Driver
18.12.11 Creating the Dataset
18.12.12 Creating the Training Loop
18.13 Overview of Some Popular RL Algorithms
18.14 Exercises

### 19.	Training and Deploying TensorFlow Models at Scale (667-718)
19.1 Serving a TensorFlow Model
19.1.1 Using TensorFlow Serving
19.1.2 Creating a Prediction Service on GCP AI Platform
19.1.3 Using the Prediction Service
19.2 Deploying a Model to a Mobile or Embedded Device
19.3 Using GPUs to Speed Up Computations
19.3.1 Getting Your Own GPU
19.3.2 Using a GPU-Equipped Virtual Machine
19.3.3 Colaboratory
19.3.4 Managing the GPU RAM
19.3.5 Placing Operations and Variables on Devices
19.3.6 Parallel Execution Across Multiple Devices
19.4 Training Models Across Multiple Devices
19.4.1 Model Parallelism
19.4.2 Data Parallelism
19.4.3 Training at Scale Using the Distribution Strategies API
19.4.4 Training a Model on a TensorFlow Cluster
19.4.5 Running Large Training Jobs on Google Cloud AI Platform
19.4.6 Black Box Hyperparameter Tuning on AI Platform
19.5 Exercises
19.6 Thank You!

