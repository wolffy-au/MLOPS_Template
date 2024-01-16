This is an experiment comparing how the trained models fair with simple functions like multiplication and addition.

Continuous Variable Experiment 1
A simple dataset of multiplication tables (x * y) from 0 - 12 is created, a total of 169 rows. Algorithms RandomForestRegressor (R2 0.98892) and GradientBoostingRegressor (R2 0.99497) fair well during evaluation, while DecisionTreeRegressor (R2 0.97213) less so. What's interesting is that GBR and RFR's R2 score are better than DTR, but the results show almost perfect recall during inference for DTR.

Continuous Variable Experiment 2
Same dataset duplicated using the same dataset to create a total of 676 rows. Algorithms RandomForestRegressor (R2 0.99984), GradientBoostingRegressor (R2 0.99869) and DecisionTreeRegressor (R2 1.0). Again DTR has perfect recall during inference.

Continuous Variable Experiment 3
The same concept of dataset (x * y) but with an additional level of resolution, with increments of 0.1, a total of 16,900 rows. This time RandomForestRegressor (R2 0.99997) and KNeighborsRegressor (0.99993) well.

Continuous Variable Experiment 4
This dataset adds a level of complexity by adding a third value to the function with a simple addition (w + x * y), a total of 219,700 rows. The first point to note is the feature evaluation and selection doesn't really pick up the nuance of the third variable. As a result, the feature reduction is commented out. Again RandomForestRegressor (R2 0.99997) and KNeighborsRegressor (0.99994) well.