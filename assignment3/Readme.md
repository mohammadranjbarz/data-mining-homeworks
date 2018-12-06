## Preparing data
In [Breast Cancer Wisconsin (Prognostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Prognostic%29)
 Each record represents follow-up data for one breast cancer case. These are consecutive
patients seen by Dr. Wolberg since 1984, and include only those cases exhibiting
invasive breast cancer and no evidence of distant metastases at the time of diagnosis. 

These are attributes that sample-code-number is just an ID and we don't count it a feature,
and the class attribute is our output for  regression. 

       #  Attribute                     Domain
       -- -----------------------------------------
       1. Sample code number            id number
       2. Clump Thickness               1 - 10
       3. Uniformity of Cell Size       1 - 10
       4. Uniformity of Cell Shape      1 - 10
       5. Marginal Adhesion             1 - 10
       6. Single Epithelial Cell Size   1 - 10
       7. Bare Nuclei                   1 - 10
       8. Bland Chromatin               1 - 10
       9. Normal Nucleoli               1 - 10
       10. Mitoses                      1 - 10
       11. Class:                       (2 for benign, 4 for malignant)
       
First of all we need to remove the Data that have missing values (16 row).


# Analysis
## Regression Tree

    MSE : 0.17329908146827155
    R^2 score : $r2_score(y_test, y_pred)
![Decision tree](./results/decision_tree.jpg)

## Decision Tree

    MSE : 0.20641685544583133
    R^2 score : $r2_score(y_test, y_pred)


![Regression decision tree](./results/regression_tree.jpg)

# Bagging 

        
    Confusion matrix : [[85  3]
     [ 2 47]]
    MSE : 0.145985401459854
    Accuracy : 0.9635036496350365
    R^2 score : 0.8425287356321839
    
# SVM
        
    Confusion matrix : [[83  2]
     [ 4 48]]
    MSE : 0.17518248175182483
    Accuracy : 0.9562043795620438
    R^2 score : 0.8110344827586207
    
    
# Random forrest

    Confusion matrix : [[84  4]
     [ 3 46]]
    MSE : 0.20437956204379562
    Accuracy : 0.948905109489051
    R^2 score : 0.7795402298850574

## References


