## Question 1
a. UnitPrice and flag1 are correlated with ListPrice
b. At each level except for the base level there is low variance in Cost. At each level there is near uniformity in flag1

## Question 2
a. One of the primary variables for explaining discount behavior is ListPrice
b. It appears they gravitate towards discount percentages that are easy to calculate

## Question 3
a. Line seems to be important for determining ListPriceMarkup based on correlation

## Question 4
a. Used Line, Revenue and Volume to create model for ExpectedListPriceMarkup
b. ProductNumber `11982`	"30 Inch Canfield Fan" has a ListPriceMarkup that is 1/3 of the ExpectedListPriceMarkup
c. 40% of items fall within 10% of ExpectedListPriceMarkup
d. Both variables appear to take far fewer values than ListPrice. X appears to have 10 unique values and Color has 3. They could be numerical features thaat have been binned. They 2 most predictive feature based on correlation coefficient are Cost and Line, those would be my guesses for X and Color
e. It appears to be the same 2 variables as they take the same number of unique values

## ListPrice Model
a. 50% of items fall within 10% of ExpectedListPrice
b. The model has an $R^2$ score of .51
c. Items that differ in actual price from predicted price
    * `15335	70 Inch Monarch Fan`
    * `11984	30 Inch Canfield Fan`
    * These could differ because not enough features were used in my model, or due to the one hot encoding