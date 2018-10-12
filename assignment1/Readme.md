## Preparing data
First of all we need to remove the Data that have missing values.

## Analysis

### Linear regression 
We have 9 Features, we calculated the regression for this features, and save the result of them in 
[Results Data](./results)

Now we should analyze every feature data 
1. [bareNuclei](./results/bareNuclei.txt)
p-value is  0.000  so this feature is significant

2. [clump thickness](./results/clumpThickness.txt)
p-value is  0.000  so this feature is significant

3. [uniformity of cell size](./results/uniformityOfCellSize.txt)
p-value is  0.000  so this feature is significant

4. [uniformity of cell shape](./results/uniformityOfCellShape.txt)
p-value is  0.000  so this feature is significant

5. [marginal adhesion](./results/marginalAdhesion.txt)
p-value is  0.000  so this feature is significant

6. [single epithelial cell size](./results/singleEpithelialCellSize.txt)
p-value is  0.000  so this feature is significant

7. [bland chromatin](./results/blandChromatin.txt)
p-value is  0.000  so this feature is significant

8. [normal nucleoli](./results/normalNucleoli.txt)
p-value is  0.000  so this feature is significant

9. [mitoses](./results/mitoses.txt)
p-value is  0.000  so this feature is significant


### Multiple Linear Regression

As we see in Linear Regression all features were significant,
but if we want to know that if the features are truly significant 
and the effects are not from other features, we calculate the 
multiple regression with these features and we face with this 

[All features multiple regression result](./results/allFeatures.txt)

    Dep. Variable:                  class   R-squared:                       0.843
    Model:                            OLS   Adj. R-squared:                  0.841
    Method:                 Least Squares   F-statistic:                     402.5
    Date:                Wed, 10 Oct 2018   Prob (F-statistic):          4.46e-264
    Time:                        20:24:42   Log-Likelihood:                -303.90
    No. Observations:                 683   AIC:                             627.8
    Df Residuals:                     673   BIC:                             673.1
    Df Model:                           9                                         
    Covariance Type:            nonrobust        

                                 coef         std err      t        P>|t|      [0.025       0.975]
    const                        1.5047       0.033     45.807      0.000       1.440       1.569
    clumpThickness               0.0634      0.007      8.898      0.000       0.049       0.077
    uniformityOfCellSize         0.0437      0.013      3.428      0.001       0.019       0.069
    uniformityOfCellShape        0.0313      0.012      2.508      0.012       0.007       0.056
    marginalAdhesion             0.0165      0.008      2.065      0.039       0.001       0.032
    singleEpithelialCellSize     0.0202      0.010      1.924      0.055      -0.000       0.041
    bareNuclei                   0.0908      0.006     14.091      0.000       0.078       0.103
    blandChromatin               0.0384      0.010      3.802      0.000       0.019       0.058
    normalNucleoli               0.0371      0.007      4.981      0.000       0.022       0.052
    mitoses                      0.0020      0.010      0.197      0.844      -0.018       0.021
So the mitoses and singleEpithelialCellSize have p-value > 0.05 then these are 
insignificant features and we can remove them and again calculate the multiple
linear regression with other 7 features and get these.
and prob(F-statistic) is significant, this means at least one feature has relationship with response. 

[All significant features multiple regression result](./results/allSignificantFeatures.txt)

    Dep. Variable:                  class   R-squared:                       0.842
    Model:                            OLS   Adj. R-squared:                  0.841
    Method:                 Least Squares   F-statistic:                     515.4
    Date:                Thu, 11 Oct 2018   Prob (F-statistic):          6.47e-266
    Time:                        21:22:05   Log-Likelihood:                -305.95
    No. Observations:                 683   AIC:                             627.9
    Df Residuals:                     675   BIC:                             664.1
    Df Model:                           7                                         
    Covariance Type:            nonrobust   

                              coef        std err     t          P>|t|      [0.025      0.975]
    const                     1.5318      0.030     51.224      0.000       1.473       1.591
    clumpThickness            0.0638      0.007      8.960      0.000       0.050       0.078
    uniformityOfCellSize      0.0504      0.012      4.096      0.000       0.026       0.075
    bareNuclei                0.0913      0.006     14.187      0.000       0.079       0.104
    blandChromatin            0.0386      0.010      3.833      0.000       0.019       0.058
    normalNucleoli            0.0393      0.007      5.359      0.000       0.025       0.054
    uniformityOfCellShape     0.0331      0.012      2.654      0.008       0.009       0.058
    marginalAdhesion          0.0177      0.008      2.237      0.026       0.002       0.033
    
As we can see in second multiple regression, R-squared  decreases but this is natural
because when number of features decrease then R-squared decreases too, so the
best way is to compare Adjusted R-squared in two models that
 in these tho models are equal (0.841), so it tells us removing 2 feature doesnt decrease 
 R-squared too much so they are not important features.
 In other hand we can compare Prob(F-statistics) too, that when removing 2 Features the
 prob(F-statistics) decreases so we can say that this deleting features give us
 better result.
 
### Regularization
When Features or samples are too much the over fitting problem may happen
so we must regularization the features and remove some features that dont't 
have significant p-values, we use 3 ways to regularize our data in this assignment

#### Ridge
[Ridge regression Result](./results/ridgeRegression.txt)

                        Feature  Coefficients  t values  Standard Errors  Probabilites
    0                 constants        1.5047    45.809            0.033         0.000
    1            clumpThickness        0.0634     8.897            0.007         0.000
    2      uniformityOfCellSize        0.0437     3.427            0.013         0.001
    3     uniformityOfCellShape        0.0313     2.509            0.012         0.012
    4          marginalAdhesion        0.0165     2.066            0.008         0.039
    5  singleEpithelialCellSize        0.0202     1.924            0.010         0.055
    6                bareNuclei        0.0908    14.089            0.006         0.000
    7            blandChromatin        0.0383     3.801            0.010         0.000
    8            normalNucleoli        0.0371     4.981            0.007         0.000
    9                   mitoses        0.0020     0.198            0.010         0.843


#### Lasso
[Lasso regression Result](./results/lassoRegression.txt)

                        Feature  Coefficients  t values  Standard Errors  Probabilites
    0                 constants        1.6064    48.423            0.033         0.000
    1            clumpThickness        0.0558     7.757            0.007         0.000
    2      uniformityOfCellSize        0.0553     4.293            0.013         0.000
    3     uniformityOfCellShape        0.0315     2.502            0.013         0.013
    4          marginalAdhesion        0.0116     1.436            0.008         0.152
    5  singleEpithelialCellSize        0.0017     0.156            0.011         0.876
    6                bareNuclei        0.0952    14.640            0.007         0.000
    7            blandChromatin        0.0256     2.514            0.010         0.012
    8            normalNucleoli        0.0370     4.918            0.008         0.000
    9                   mitoses        0.0000     0.000            0.010         1.000


#### Elastic net
[Elastic net regression Result](./results/elasticNetRegression.txt)
    
                        Feature  Coefficients  t values  Standard Errors  Probabilites
    0                 constants        1.8835    49.730            0.038         0.000
    1            clumpThickness        0.0257     3.127            0.008         0.002
    2      uniformityOfCellSize        0.0577     3.926            0.015         0.000
    3     uniformityOfCellShape        0.0316     2.197            0.014         0.028
    4          marginalAdhesion        0.0000     0.000            0.009         1.000
    5  singleEpithelialCellSize        0.0000     0.000            0.012         1.000
    6                bareNuclei        0.0981    13.208            0.007         0.000
    7            blandChromatin        0.0000     0.000            0.012         1.000
    8            normalNucleoli        0.0246     2.868            0.009         0.004
    9                   mitoses        0.0000     0.000            0.011         1.000


#### Conclusion

## References
* https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

* http://statisticsbyjim.com/regression/interpret-f-test-overall-significance-regression



