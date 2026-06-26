# Network Graph
![Great Expectations validation results overview](image/Result_overview.png)
![Great Expectations validation age](image/age.png)
![Great Expectations validation results country](image/country.png)
![Great Expectations validation results date](image/date.png)
![Great Expectations validation results email](image/email.png)
![Great Expectations validation results id not null](image/id_NOTNULL.png)
![Great Expectations validation results id unique](image/id_unique.png)
![Great Expectations validation results rows](image/rows.png)
![Great Expectations validation results salary](image/salary.png)

# Data Set Issues
Rows: there is 5015 rows of dataset, so it more than 1000 rows
Ages: 384 unpected age values has been found, which is about 7.888% of total data. example unexpected value is 999, -25, 200
Country: 301 unexpected value found, 6.051% of total data set. such as Mexico, Brazil, Germany, Spain, India,china 
Customer_id: 568 unexpected value found, 11.68 % of total data set. ex: C00009, C00019, C00100
Customer_id_not_null: 150 null value. 2.991% of total data set
Email: 346 unexpected value found. 7.56% of total data set. ex: missingatsign.com, user@, user@@domain.com, @domain.com
Salary: 425 Null value found. 8.475% of total data set. 
signup_date: Day value out of range ex: 2023-02-30 
Problem has been solved through data preprocess
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

# Pytest execution
![Test Result](image/pytest.png)

# A brief reflection on what you found challenging about resolving merge conflicts
The out-of-range numeric values in age and salary such as -5000 salary or -19 age would do the most damage to a model. The reason is that, unlike the missing values, these errors are still valid numbers, so they pass straight through type checks and into training without raising any flag. Standardization will get wrecked because the mean and standard deviation are dragged by ±999, gradient-based models become unstable, distance-based methods such as kNN are dominated by the bogus magnitudes, and linear-model coefficients and tree split points get pulled toward noise. One age = 999 row can degrades the scaling of the entire column. 