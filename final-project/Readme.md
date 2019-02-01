## Preparing data


    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)    
First of all we need to remove the Data that have missing values (16 row).

داده های این پروژه مربوط به تشخیص دیابت در حدود 700 نفر است و هدف نهایی کلاس بندی داده ها به دو دسته ی سالم و بیمار می باشد.در این پروژه ما مسئله را با 8 روش logistic regression ، LDA ، QDA ، gnb ، SVM ، random forest  ، bagging و decision tree و در 4 حالت پیش پردازش خودرمزنگار با دولایه پنهان ، پیش پردازش خودرمزنگار با سه لایه پنهان ، پیش پردازش PCA و بدون پیش پردازش برای هرکدام بررسی نموده ایم و نتایج را با صورت های متنوع مقایسه کرده ایم که 

# Analysis
## Logistic regression

### Auto encoder pre-processing results

| neurons | AUC    |
|---------|--------|
| 3 , 2   | 0.6315 |
| 4 , 2   | 0.5319 |
| 4 , 3   | 0.5132 |
| 5 , 2   | 0.5916 |
| 5 , 3   | 0.5545 |
| 5 , 4   | 0.6315 |
| 6 , 3   | 0.6818 |
| 6 , 4   | 0.5192 |
| 6 , 5   | 6154   |
| 7 , 2   | 0.5319 |
| 7 , 3   | `0.7078` |
| 7 , 4   | 0.5397 |
| 7 , 5   | 0.6813 |
| 7 , 6   | 0.5949 |



| neurons   | AUC    |
|-----------|--------|
| 4 , 3 , 2 | 0.5    |
| 5 , 3 , 2 | 0.5192 |
| 5 , 4 , 2 | 0.4953 |
| 5 , 4 , 3 | 0.5975 |
| 6 , 3 , 2 | 0.5086 |
| 6 , 4 , 2 | 0.5272 |
| 6 , 5 , 2 | 0.5    |
| 6 , 5 , 3 | 0.4953 |
| 6 , 5 , 4 | 0.6069 |
| 7 , 4 , 2 | 0.5272 |
| 7 , 4 , 3 | 0.5358 |
| 7 , 5 , 2 | 0.5119 |
| 7 , 5 , 3 | 0.6639 |
| 7 , 5 ,4  | 0.6175 |
| 7 , 6 , 2 | 0.5179 |
| 7 , 6 , 3 | 0.5949 |
| 7 , 6 , 4 | 0.6473 |
| 7 , 6 , 5 | `0.6873` |


### PCA pre-processing results



با توجه به مقادیر بدست آمده ،میتوان مشاهده کرد که بهترین حالت الگوریتم logistic regression در پیش پرداز با خودرمزنگار با دولایه پنهان در حالت 7و3 نورون با مقدارAUCبرابر با 0.7078 است.
درحالت پیش پردازش با خودرمزنگار با سه لایه پنهان بهترین حالت با تعداد 7و6و5 نورون با مقدار 0.6873 است.
درحالت پیش پردازش با PCAبهترین حالت با 3 مولفه اصلی بدست می آید که مقدار آن برابر 0.7158 است.
همچنین مقدار AUCدر حالت بدون پیش پردازش برابر با 0.7665 میشود.
باتوجه به مقادیر ومقایسه آن ها میتوان نتیجه گرفت که logistic regressionدر حالت بدون پیش پردازش دارای بهترین مقدار    AUC و در حالت پیش پردازش با خودرمزنگار با سه لایه پنهان با 6و5و3 نورون دارای کمترین مقدار AUC است که برابر با 0.4953 می باشد. بنابراین الگوریتم logistic regression بهتر است بدون پردازش اجرا شود.


| Components | AUC    |
|------------|--------|
| 1          | 0.5272 |
| 2          | 0.7099 |
| 3          | `0.7158` |
| 4          | 0.7112 |
| 5          | 0.7052 |
| 6          | 0.7112 |
| 7          | 0.7065 |

# Conclusion
# References
* [autoencoder-keras-tutorial](https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial)

* [https://keras.io/getting-started/sequential-model-guide/#stacked-lstm-for-sequence-classification](https://keras.io/getting-started/sequential-model-guide/#stacked-lstm-for-sequence-classification)

* [Applied Deep Learning - Part 3: Autoencoders](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798)