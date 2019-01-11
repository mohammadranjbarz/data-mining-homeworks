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
    
classification_report

                  precision    recall  f1-score   support
    
               2       0.97      0.97      0.97        87
               4       0.94      0.94      0.94        50
    
       micro avg       0.96      0.96      0.96       137
       macro avg       0.95      0.95      0.95       137
    weighted avg       0.96      0.96      0.96       137


 
auc_score 
    
    0.9527586206896552
    
We compare result of neural network with result of [Assignment2](https://gitlab.com/mranjbar.z2993/sbu-data-mining/tree/master/assignment2)
to check accuracy of neural network results

* Compare based on precision
In precision analysis if this parameter is greater, the classification is better. according to the explanation the neural network that has highest precision value is better.
`5-Fold < LOOCV < Neural Network`

* Compare based on recall
In recall analysis if this parameter is greater, the classification is better. according to the explanation the neural network that has highest recall value is better.
`5-Fold < LOOCV < Neural Network`

* Compare based on f1-score
The value of the f1-score in the neural network is larger so we can say that the neural network is better.
`5-Fold < LOOCV < Neural Network`

# Conclusion
By comparing the values obtained in the neural network and the second project, we conclude that the neural network is a good model for data.


