Lab3 – Machine Learning Workshop R codes
================

## ML codes

Now are trying to practice the machine learning for iris data. First we
call the packages:

## iris data set

## Table 1

<div class="Rtable1"><table class="Rtable1">
<thead>
<tr>
<th class='rowlabel firstrow lastrow'></th>
<th class='firstrow lastrow'><span class='stratlabel'>No<br/><span class='stratn'>(N=100)</span></span></th>
<th class='firstrow lastrow'><span class='stratlabel'>Yes<br/><span class='stratn'>(N=50)</span></span></th>
<th class='firstrow lastrow'><span class='stratlabel'>Overall<br/><span class='stratn'>(N=150)</span></span></th>
</tr>
</thead>
<tbody>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Sepal.Length</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>5.80 (0.945)</td>
<td>5.94 (0.516)</td>
<td>5.84 (0.828)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>5.70 [4.30, 7.90]</td>
<td class='lastrow'>5.90 [4.90, 7.00]</td>
<td class='lastrow'>5.80 [4.30, 7.90]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Sepal.Width</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>3.20 (0.418)</td>
<td>2.77 (0.314)</td>
<td>3.06 (0.436)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>3.20 [2.20, 4.40]</td>
<td class='lastrow'>2.80 [2.00, 3.40]</td>
<td class='lastrow'>3.00 [2.00, 4.40]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Petal.Length</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>3.51 (2.10)</td>
<td>4.26 (0.470)</td>
<td>3.76 (1.77)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>3.20 [1.00, 6.90]</td>
<td class='lastrow'>4.35 [3.00, 5.10]</td>
<td class='lastrow'>4.35 [1.00, 6.90]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Petal.Width</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>1.14 (0.918)</td>
<td>1.33 (0.198)</td>
<td>1.20 (0.762)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>1.00 [0.100, 2.50]</td>
<td class='lastrow'>1.30 [1.00, 1.80]</td>
<td class='lastrow'>1.30 [0.100, 2.50]</td>
</tr>
</tbody>
</table>
</div>
## Table 2

<div class="Rtable1"><table class="Rtable1">
<thead>
<tr>
<th class='rowlabel firstrow lastrow'></th>
<th class='firstrow lastrow'><span class='stratlabel'>No<br/><span class='stratn'>(N=100)</span></span></th>
<th class='firstrow lastrow'><span class='stratlabel'>Yes<br/><span class='stratn'>(N=50)</span></span></th>
<th class='firstrow lastrow'><span class='stratlabel'>Overall<br/><span class='stratn'>(N=150)</span></span></th>
</tr>
</thead>
<tbody>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Sepal.Length</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>5.80 (0.945)</td>
<td>5.94 (0.516)</td>
<td>5.84 (0.828)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>5.70 [4.30, 7.90]</td>
<td class='lastrow'>5.90 [4.90, 7.00]</td>
<td class='lastrow'>5.80 [4.30, 7.90]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Sepal.Width</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>3.20 (0.418)</td>
<td>2.77 (0.314)</td>
<td>3.06 (0.436)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>3.20 [2.20, 4.40]</td>
<td class='lastrow'>2.80 [2.00, 3.40]</td>
<td class='lastrow'>3.00 [2.00, 4.40]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Petal.Length</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>3.51 (2.10)</td>
<td>4.26 (0.470)</td>
<td>3.76 (1.77)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>3.20 [1.00, 6.90]</td>
<td class='lastrow'>4.35 [3.00, 5.10]</td>
<td class='lastrow'>4.35 [1.00, 6.90]</td>
</tr>
<tr>
<td class='rowlabel firstrow'><span class='varlabel'>Petal.Width</span></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
<td class='firstrow'></td>
</tr>
<tr>
<td class='rowlabel'>Mean (SD)</td>
<td>1.14 (0.918)</td>
<td>1.33 (0.198)</td>
<td>1.20 (0.762)</td>
</tr>
<tr>
<td class='rowlabel lastrow'>Median [Min, Max]</td>
<td class='lastrow'>1.00 [0.100, 2.50]</td>
<td class='lastrow'>1.30 [1.00, 1.80]</td>
<td class='lastrow'>1.30 [0.100, 2.50]</td>
</tr>
</tbody>
</table>
</div>
## tarin and test

    ## [1] 105   5

    ## Sepal.Length  Sepal.Width Petal.Length  Petal.Width      Species 
    ##    "numeric"    "numeric"    "numeric"    "numeric"     "factor"

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
    ## 1          5.1         3.5          1.4         0.2      No
    ## 3          4.7         3.2          1.3         0.2      No
    ## 5          5.0         3.6          1.4         0.2      No
    ## 6          5.4         3.9          1.7         0.4      No
    ## 7          4.6         3.4          1.4         0.3      No
    ## 8          5.0         3.4          1.5         0.2      No

    ## [1] "No"  "Yes"

    ##     freq percentage
    ## No    70   66.66667
    ## Yes   35   33.33333

## split input and output

## some plots

![](Readme_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->![](Readme_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->
![](Readme_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## set the cross-validation and metric

## Logistics regression

## LASSO

## DT

![](Readme_files/figure-gfm/dt%20fit-1.png)<!-- --> \## RF

## Varible importance

![](Readme_files/figure-gfm/var%20imp%20plot-1.png)<!-- -->

## SVM

    ## Warning: package 'kernlab' was built under R version 4.3.3

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     cross

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

![](Readme_files/figure-gfm/plot%20for%20ann1-1.png)<!-- -->

![](Readme_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## ANN

We can use the **train()** function from the caret package to tune our
hyperparameters. Here, we will use the **nnet** package (method =
“nnet”). We can tune the size and decay hyperparameters.

- size: number of nodes in the hidden layer Note: There can only be one
  hidden layer using nnet
- decay: weight decay. regularization parameter to avoid overfitting,
  which adds a penalty for complexity.

First, we set up the grid using the expand.grid() function. We will
consider hidden node sizes (size) of 1, 3, 5 and 7 and decay values
ranging from 0 to 0.1 in 0.01 increments.

![](Readme_files/figure-gfm/plot%20for%20ann-1.png)<!-- -->

![](Readme_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

## summary of results in training data

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: logistic_regression, Elastic_net, knn, DT, RF, XGboost, SVM_linear1, SVM_nonlinear, ANN 
    ## Number of resamples: 10 
    ## 
    ## ROC 
    ##                          Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
    ## logistic_regression 0.7619048 0.8125000 0.8571429 0.8642857 0.8839286    1    0
    ## Elastic_net         0.4642857 0.7113095 0.8928571 0.8309524 1.0000000    1    0
    ## knn                 0.9523810 1.0000000 1.0000000 0.9934524 1.0000000    1    0
    ## DT                  0.7619048 0.9151786 0.9404762 0.9339286 1.0000000    1    0
    ## RF                  1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0
    ## XGboost             1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0
    ## SVM_linear1         0.6428571 0.7142857 0.7500000 0.8095238 0.9404762    1    0
    ## SVM_nonlinear       0.9523810 0.9732143 1.0000000 0.9880952 1.0000000    1    0
    ## ANN                 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0
    ## 
    ## Sens 
    ##                          Min.   1st Qu.    Median      Mean 3rd Qu. Max. NA's
    ## logistic_regression 0.5714286 0.7500000 0.8571429 0.8571429       1    1    0
    ## Elastic_net         0.4285714 0.7500000 0.9285714 0.8571429       1    1    0
    ## knn                 0.8571429 1.0000000 1.0000000 0.9857143       1    1    0
    ## DT                  0.7142857 0.8571429 1.0000000 0.9285714       1    1    0
    ## RF                  0.8571429 1.0000000 1.0000000 0.9714286       1    1    0
    ## XGboost             0.8571429 1.0000000 1.0000000 0.9714286       1    1    0
    ## SVM_linear1         0.7142857 0.8571429 0.8571429 0.8857143       1    1    0
    ## SVM_nonlinear       0.8571429 0.8571429 1.0000000 0.9428571       1    1    0
    ## ANN                 0.8571429 0.8928571 1.0000000 0.9571429       1    1    0
    ## 
    ## Spec 
    ##                          Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
    ## logistic_regression 0.0000000 0.2708333 0.5833333 0.5250000 0.7500000    1    0
    ## Elastic_net         0.0000000 0.3125000 0.5833333 0.5250000 0.6666667    1    0
    ## knn                 0.7500000 1.0000000 1.0000000 0.9750000 1.0000000    1    0
    ## DT                  0.6666667 0.8125000 1.0000000 0.9083333 1.0000000    1    0
    ## RF                  1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0
    ## XGboost             1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0
    ## SVM_linear1         0.2500000 0.3333333 0.4166667 0.5083333 0.5000000    1    0
    ## SVM_nonlinear       0.6666667 0.8125000 1.0000000 0.9083333 1.0000000    1    0
    ## ANN                 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    1    0

![](Readme_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

## Prediction

## ROC curve

    ##  [1] No  No  No  No  No  No  No  No  No  No  No  No  Yes Yes Yes Yes Yes Yes Yes
    ## [20] Yes Yes Yes Yes Yes Yes Yes Yes No  No  No  No  No  No  No  No  No  No  No 
    ## [39] No  No  No  No  No  No  No 
    ## Levels: No Yes

    ##                
    ## predictions_svm No Yes
    ##             No  30   3
    ##             Yes  0  12

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction No Yes
    ##        No  30   3
    ##        Yes  0  12
    ##                                          
    ##                Accuracy : 0.9333         
    ##                  95% CI : (0.8173, 0.986)
    ##     No Information Rate : 0.6667         
    ##     P-Value [Acc > NIR] : 2.435e-05      
    ##                                          
    ##                   Kappa : 0.8421         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.2482         
    ##                                          
    ##             Sensitivity : 1.0000         
    ##             Specificity : 0.8000         
    ##          Pos Pred Value : 0.9091         
    ##          Neg Pred Value : 1.0000         
    ##              Prevalence : 0.6667         
    ##          Detection Rate : 0.6667         
    ##    Detection Prevalence : 0.7333         
    ##       Balanced Accuracy : 0.9000         
    ##                                          
    ##        'Positive' Class : No             
    ## 

    ##  [1] No No No No No No No No No No
    ## Levels: No Yes

## Random Over sampling

Functions to deal with binary classification problems in the presence of
imbalanced classes. Synthetic balanced samples are generated according
to ROSE (Menardi and Torelli, 2014). Functions that implement more
traditional remedies to the class imbalance are also provided, as well
as different metrics to evaluate a learner accuracy. These are estimated
by holdout, bootrstrap or cross-validation methods.

### Example

    ## 
    ##   0   1 
    ## 980  20

    ## 
    ##   0   1 
    ## 507 493

## compare balanced and imbalanced

    ## Area under the curve (AUC): 0.803

![](Readme_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

    ## Area under the curve (AUC): 0.915

## Roc Eval

    ## Iteration: 
    ## 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

    ## 
    ## Call: 
    ## ROSE.eval(formula = cls ~ ., data = hacide.train, learner = glm, 
    ##     method.assess = "BOOT", B = 10, control.learner = list(family = binomial), 
    ##     trace = TRUE)
    ## 
    ## Summary of bootstrap distribution of auc: 
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.9274  0.9277  0.9278  0.9280  0.9282  0.9291
