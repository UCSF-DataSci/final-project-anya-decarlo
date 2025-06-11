Section 1: Describing the Problem and the Dataset (*all figures and tables at the end, 3 pages of text)
Context: Metastatic breast cancer (MBC) is a diverse group of illnesses with numerous systemic therapy
options, including chemotherapy, hormonal therapies, targeted therapies, and immunotherapies [1]. While these
treatments aim to improve survival and quality of life, anti-cancer treatments are associated with adverse events
impacting multiple systems (eg. cardiovascular, respiratory, gastrointestinal, and immune systems), which can
result in significant morbidity or death. [2]. Additionally, both cancer subtype and burden of disease influence
survival patterns in this population, ranging from slow progression of disease over several years, to explosive
end-organ failure [3]. Predicting patients at risk of imminent death is essential both to facilitate end of life
planning, and to aid in transitioning to palliative care. This project will examine whether gated recurrent units
(GRU) can be used to predict acute death events among an unselected sample of individuals with metastatic
breast cancer followed at UCSF. While neural networks have the best predictive performance in large, complex
datasets [4], a GRU model was selected for this project given the longitudinal nature of data, and to facilitate
learning/trouble shooting prior to applying this model in large administrative databases.
Dataset: This project utilized electronic medical record (EMR) data from the UCSF deidentified clinical data
warehouse. Patients with metastatic breast cancer were identified using UCSF diagnostic grouper codes (“ICD
Cancer Breast UCSF Group”), and ICD-10 codes (C79.*). The cohort was restricted to individuals who were
deceased, with vital registry linkages used to identify patients with a documented date of death. The dataset
included 480 individuals from 7708 unique encounters, with up to 16 clinically relevant features associated with
each visit. These features included age, vital signs (body mass index, blood pressure, pulse, respiratory rate,
temperature), survival variables (survival from 1st encounter to death, duration of time between last encounter
and death), visit variables (time since last visit, duration of visit, visit department) and whether medications
were reviewed at the visit. The outcome of interest was defined as death within 2 weeks of the last follow up
visit. Patient data ranging from January 2016 to present was considered.
Patient-level variables stratified by the outcome are presented in Table 1 (see final pages). Eighty-three
individuals (17%) experienced the event. A total of 52 visit types were represented from 16 departments, with
the most common visits including chemotherapy assessment visits (53.6%) and general follow up visits (30.3%)
Section 2: Describing Gated Recurrent Units
Recurrent neural networks (RNNs) are a subtype of feedforward neural networks which can process input of
varying lengths [5]. Given this property, they are useful models for analyzing time series data, including data
from the EMR [6]. RNN’s use a forward pass mechanism to feed input through the network, integrating data
from the previous hidden states to update the hidden state at the current time step [7]. An overall loss is
computed by calculating the difference between predicted and true labels at each step, followed by
backpropogation of the error gradients through the network via the chain rule to update the network weights
[8]. One limitation of early “vanilla” RNNs however, is the “vanishing gradient problem” [9]. This phenomena
occurs in the setting of long term dependencies, when non linear activation functions (e.g. sigmoid functions)
become “saturated”, resulting in very small gradients over time [5]. Gated neural networks, including long short
term memory (LSTM) and gated recurrent units (GRUs) were developed to address these limitations.
Like LSTM’s, GRUs modulate the flow of information through the unit, however, unlike LSTM’s, instead of a
memory cell, they possess a candidate activation vector which is updated by two distinct gates. The reset gate
modulates how much information to forget from the last hidden state, and the update gate controls how much
information to transfer from the previous hidden state, and how much new information to integrate from time
“t” [5]. GRUs have simpler architecture and fewer parameters than LSTM’s, making them less computationally
expensive [5].
The reset gate can be defined by the expression 𝑟𝑡 = 𝜎(𝑊 𝑟𝑥𝑡 + 𝑈𝑟ℎ𝑡−1) where xt represents input at time “t”,
ℎ𝑡−1
𝑗 represents the hidden state at time t-1, and Wr and Ur represent weights. After computation of a weighted
sum, a sigmoid activation function is applied to compress rt values between 0 and 1. The candidate memory
vector “ℎ𝑡
̃
” is computed by applying element wise multiplication between the rt and ht-1 vectors, with
multiplication by weight “U”. After applying the weighted sum of the input, a tahn activation function is
applied to compress values between -1 and 1 as defined by the expression:
ℎ𝑡
̃
= tanh(𝑊𝑥𝑡 + 𝑈(𝑟𝑡 ⊙ ℎ𝑡−1)). Similar to the reset gate, the update gate (𝑧𝑡) can be defined by the
expression 𝑧𝑡 = 𝜎(𝑊 𝑧𝑥𝑡 + 𝑈𝑧ℎ𝑡−1), where zt represents the weighted sum of the input and hidden state vectors.
Lastly, the final memory at the current state “ht” can be defined by the expression: ℎ𝑡
𝑗 = (1 − 𝑧𝑡
𝑗) ⊙ ℎ𝑡−1
𝑗 +
𝑧𝑡
𝑗 ⊙ ℎ𝑡
𝑗̃
where element wise multiplication is applied between 1) the reset vector (rt ) and the candidate vector
(ℎ𝑡
̃
), and 2) the complement of the update vector (1 − 𝑧𝑡
𝑗) and the previous hidden state ℎ𝑡−1
𝑗
, with summation
of the two resulting vectors. The final memory at the current step is then fed into the next step of the network.
GRUs make several assumptions: Firstly, it is assumed that the observations from the current input are
temporally related to observations from previous time points [6]. Secondly, it is assumed that the same weights
and biases can be applied across all time points for a particular sequence [5]. As GRUs utilize sigmoid and tahn
activation functions, it is assumed that non-linear modelling of the data is appropriate. Lastly, when training
neural networks, some authors suggest sample sizes at least 10- 50 times larger than the number of adjustable
weights to adequately train the model [4], therefore, large datasets are optimal when using these networks.
Section 3: Architecture/Parameter Selection and Discussion of Results
The data was pre-processed into nested lists representing 1) patient ID, 2) the label of interest (0 or 1) and time
to event, and 3) unique visit times with associated features. Up to 17% of encounters were missing values for
vital signs (BMI, BP, temperature, pulse, RR). While specific GRU extensions (eg. GRU-D) are robust in
accounting for informative missing data [10], imputation based on the patient-specific averages were
performed for simplicity. All features were continuous excluding VisitType, Diagnosis, and Department (string
variables), and “Medication review” (binary). To address variable length sequencing, the
“pack_padded_sequence” function was applied prior to model training.
A GRU model of class “GRUModelWIthEmbeddings” (pytorch version 2.3.0) was applied to the dataset.
String/categorical variables were mapped to numeric indices. The dataset was split into a test set (10%, n=48), a
validation set (10%, n=48), and a training set (80%, n=384) to facilitate hyperparameter tuning. Given sample
size limitations, stratification by the outcome was applied during splitting to ensure representation of both
classes in all datasets.
The model was initialized to accept self, vocabulary size (the length of the unique string variables), the
embedding dimensions, the number of features in the hidden state, number of layers, and the output size as
arguments. Given that the string variables contained a relatively small number of unique vocabulary (n=7627),
the embedding dimension was set to 500, and a single GRU layer was utilized given the low dimensionality
dataset. Non-padded input was fed through the forward pass, with output from the final time step fed through a
fully connected layer. During the backward pass, the gradient of the loss was computed, using the
“BCEWithLogitsLoss” criterion given the binary classification problem. An Adam optimizer was selected to
enable adaptation of the learning rate for each parameter, with the initialized learning rate specified as below,
and the remaining parameters left at default settings ( momentum coefficients (0.9, 0.999), eps= 1e-8, and
weight decay= 0).
Hyper-parameter tuning included the number of features in the hidden layer (10, 20,30, 40, 50, 100), the
number of epochs (10, 50, 100) and the learning rate (0.1, 0.01, and 0.001, 0.0001). Tuning was performed on
the training dataset, with testing of the AUC in the validation set. As the dataset was small, dimensions of
hidden size were restricted to prevent overfitting. The optimal hyperparameters were selected based on the best
validation AUC. The final model was trained using these parameters, and data from both the validation and the
training dataset was combined for final model training, with testing in a holdout dataset. Additional model
metrics including accuracy, F1, recall, and precision scores were also computed.
The best hyperparameters identified were a hidden size of 10 and a learning rate of 0.01 over 10 epochs. Plots
of the Epoch vs loss, and Epoch vs AUC in the training data using the best hyperparameters are presented in
Figure 1. The AUC, sensitivity, and specificity for the combined training data, and hold-out test data are shown
in Table 2, with other model metrics stratified by class, including weighted and macro averages, presented in
Figure 2.
Discussion: In a small longitudinal dataset containing clinical variables from 480 individuals with metastatic
breast cancer, a single layer GRU model had poor predictive performance. This can be explained by several
factors. While GRUs have been successfully applied to EHR data [6], the number of unique patients in the
referenced study contained nearly 250 000 unique individuals, equating to a 500 fold larger sample size than
this analysis. Additionally, while features such as survival and vital signs are important predictive factors for
morbidity and mortality , the dataset lacked key factors influencing survival in this population, including prior
and present systemic therapies, recent hospitalizations, and biochemical predictors, to name a few. Imputation
of missing values may also have impacted model performance, as absent data may be informative (if patient had
a virtual visit, for example, vital signs would likely be absent).
While the model performed well in correctly identifying non-cases, it correctly identified 1/75 cases in the
training data, and 0/8 in the testing data, effectively failing to achieve its prediction task. When we examine our
loss and AUC plots by Epoch (Fig. 1), we note that while there is some improvement in the loss by Epoch, the
AUC does not improve over time, suggesting that the patterns in the training data are of low complexity, with
maximum performance reached early in the training. Moreover, the slightly higher AUC in the test data vs the
training data suggests a component of underfitting, suggesting that key factors are missing to sufficiently
explain the variability in the data. As mentioned, the dataset does not include a comprehensive list of known
predictors influencing survival in this population, likely contributing to this finding. This analysis moreover
highlights the importance of presenting diverse metrics when evaluating a model to prevent misinterpretation of
performance and biased presentation of results. Despite failure in predicting the event of interest, the accuracy
of the model was 83% due to its appropriate classification of the non-cases. Moreover, precision for the positive
label in the training data was high solely because the model predicted only 1 positive class observation which
corresponded to a true event. Overall, the model would likely perform better in a much larger dataset with a
larger number of diverse features.