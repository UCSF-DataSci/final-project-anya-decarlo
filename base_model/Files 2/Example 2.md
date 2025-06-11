Title: Using Quadratic Discriminant Analysis to Classify Breast Cancer
Introduction
Breast cancer is the a common cancer that affects women worldwide. In 2022, WHO
announced that it was the most common cancer in women in 157 out of 185 countries1
. In
the United states, it is the second most common cancer among women and accounts for
around 41,000 deaths per year2
. Breast cancer is characterized by uncontrolled growth of
cells that result in a malignant tumor, which can metastasize and spread to other organs.
This is in contrast to benign tumors which are also abnormal growths but not invade
surrounding tissues.
To better understand the genetic characteristics of breast cancer, a dataset was selected
from the Curated Microarray Database3, a repository of microarray datasets curated from
Gene Expression Omnibus studies. The selected dataset contained 289 samples, 143
breast adenocarcinoma samples and 146 normal samples. The input featured 35982
genes. The dataset was projected into two dimensional space using principle component
analysis (PCA) for data visualization. Dimensional reduction revealed two clusters,
suggesting the potential of using a discriminant analysis technique for classification of the
two classes.
Data was divided into 75:25 training and testing sets. Principal components (PC1 and PC2)
were identified using the training set and used to project both the training and testing set
onto 2-D space. Table 1 presents the baseline characteristics of the training and testing
datasets after dimension reduction along with the mean and standard deviation (SD)
values of the first two principal components obtained through PCA. The p-values from t-
test indicate the significance of differences between groups.
Training Set
Overall Breast Adenocarcinoma Normal P-Value
n 216 106 110
PC1,
mean (SD)
-0.0 (70.2) 44.4 (60.0) -42.8 (49.8) <0.001
PC2,
mean (SD)
-0.0 (47.4) 11.1 (56.2) -10.7 (34.0) 0.001
Testing Set
Overall Breast Adenocarcinoma Normal P-Value
n 73 37 36
PC1,
mean (SD)
4.1 (70.6) 50.9 (61.0) -44.0 (41.8) <0.001
PC2,
mean (SD)
1.8 (44.8) 14.4 (47.5) -11.1 (38.4) 0.014
Table 1. Baseline Characteristics and PCA Results of Training and Testing Sets
DATASCI 224 Final Project
2
Quadratic Discriminant Analysis
Quadratic Discriminant Analysis (QDA) is a classification technique that identifies a
quadratic decision boundary between classes. It shares similar assumptions with linear
discriminant analysis (LDA). Both methods assume that observations are independent
within and across classes and that data in each class follow multivariate Gaussian
distributed but have different means. While LDA takes a stronger assumption and assumes
equal variances across classes, QDA allows different variances between groups.
In QDA, posterior probability 𝑃(𝑌=𝑦|X), probability of class 𝑦 given 𝑥 as inputs can be
computed using Bayes' theorem:
𝑃(Y=𝑦|X=𝑥)=
𝑝(X=𝑥|Y=𝑦)𝑃(Y=𝑦)
𝑃(X)
The multivariate Gaussian probability density function for class 𝑦 is given by:
𝑝(X=x|Y=y)=
1
√(2𝜋)𝑘|𝚺𝑦|
exp(−
1
2(𝒙−𝝁𝒚)⊤
𝚺𝑦
−1(𝒙−𝝁𝒚))
, where 𝑘 is the number of features, 𝝁𝒚 ∈ℝ𝒌 and 𝚺𝑦 ∈ℝ𝒌×𝒌 are the mean vector and
covariance matrix of class 𝑦. Assume that 𝑃(Y=𝑦)=𝜋𝑦 where 𝜋𝑦 is the prior probability
of each class obtained from finding class frequencies using training data, the solution of
QDA is given by taking the partial derivatives with respect to the 𝝁𝑦, 𝚺𝑦 through maximum
likelihood estimation:
max
𝛍𝐲, 𝚺𝐲,𝜋𝑦∑log(
1
√(2𝜋)𝑘|𝚺𝑦|
exp(−
1
2(𝒙𝒊−𝝁𝑦)⊤
𝚺𝑦
−1(𝒙𝒊−𝝁𝑦)))
𝑛
𝑖=1
+∑log(𝜋𝑦)
𝑛
𝑖=1
⟺ max
𝛍𝐲, 𝚺𝐲,𝜋𝑦 ∑−
𝑘
2log(2𝜋)−
1
2log|𝚺𝑦|−
1
2(𝒙𝒊−𝝁𝑦)⊤
𝚺𝑦
−1(𝒙𝒊−𝝁𝑦)
𝑛
𝑖=1
+∑log(𝜋𝑦)
𝑛
𝑖=1
Mean vector (𝝁
̂), covariance matrix(𝚺
̂
), and prior probability(𝜋
̂) for each class is obtained
using input values from the training data. The parameters can then be used in discriminant
function:
δy(𝑥)=−
1
2log|Σy|−
1
2(𝑥−μy)T
Σy
−1(𝑥−μy)+log(𝜋𝑦)
DATASCI 224 Final Project
3
Given observation with input 𝑥, the observation is classified in to class 𝑦
̂ for which the
discriminant function is maximized ( y
̂
=argmax
𝑦
δy(𝑥)). The decision boundary of the
binary classification problem from this project can be found when discriminant functions
𝛿𝑘(𝑥) are equal and chances of being in the two classes are 50:50 : 𝛿0(𝑥)=𝛿1(𝑥).
Results and Discussion
Classification results are presented in Table 2. QDA algorithm demonstrates slightly better
performance than LDA in distinguishing adenocarcinoma cases, with testing precision of
0.91 and recall of 0.78. The F1 Score of 0.84 further confirms the algorithm's robustness.
However, the marginal difference in evaluation metrics between the two models suggests
that relaxing the assumption of equal variance to unequal variance doesn't significantly
enhance predictive performance in our dataset. Furthermore, although QDA provides a
more flexible classification boundary by allowing different variances for each class, this is
done at the cost of potential overfitting, especially with smaller datasets such as ours.
Careful consideration of trade-offs between model complexity and predictive gains is
crucial in algorithm selection.
Training
Algorithms Class Precision Recall F1 Score Accuracy
QDA Adenocarcinoma 0.84 0.80 0.82 0.83
Normal 0.82 0.85 0.84
LDA Adenocarcinoma 0.82 0.75 0.78 0.80
Normal 0.78 0.84 0.81
Testing
Algorithms Class Precision Recall F1 Score Accuracy
QDA Adenocarcinoma 0.91 0.78 0.84 0.85
Normal 0.80 0.92 0.86
LDA Adenocarcinoma 0.90 0.76 0.82 0.84
Normal 0.79 0.92 0.85
Table 2. Classification report of LDA and QDA models.
Further exploration of alternative methodologies or feature engineering techniques could
potentially offer more improvements in classification accuracy. To account for the possible
violation of normality assumption among principle components, future models can
consider relaxing the normality assumption and exploring the use of more flexible methods
such as support vector machines or non-parametric methods such as decision trees and
random forests.
DATASCI 224 Final Project
4
Resources
1. Breast Cancer. World Health Organizaiton. Updated March 2024. Accessed June 7, 2024.
https://www.who.int/news-room/fact-sheets/detail/breast-cancer.
2. Breast Cancer Statistics. Centers for Disease Control and Prevention. Updated
November 2023. Accessed June 7, 2024. https://gis.cdc.gov/Cancer/USCS.
3. Feltes BC, Chandelier EB, Grisci BI, Dorn M. CuMiDa: An Extensively Curated Microarray
Database for Benchmarking and Testing of Machine Learning Approaches in Cancer
Research. J Comput Biol.
4. Friedman J, Hastie T, Tibshirani R. The Elements of Statistical Learning: Data Mining,
Inference, and Prediction. 2nd ed. New York, NY: Springer; 2009. Section 4.3, pp. 106-119.