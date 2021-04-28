# Case study in order to predict sales prices 

# Objective

This project belongs to [kaggle's competitions](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In order to buy a house there are many different parameters that influences price negotiations. Therefore, the idea is to create a model that predicts the sales prices given a dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. 

# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, seaborn, matplotlib

# Data description

For each house the following information is given:

- **SalePrice:**  the property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass:** The building class.
- **MSZoning:** The general zoning classification.
- **LotFrontage:** Linear feet of street connected to property.
- **LotArea:** Lot size in square feet.
- **Street:** Type of road access.
- **Alley:** Type of alley access.
- **LotShape:** General shape of property.
- **LandContour:** Flatness of the property.
- **Utilities:** Type of utilities available.
- **LotConfig:** Lot configuration.
- **LandSlope:** Slope of property.
- **Neighborhood:** Physical locations within Ames city limits.
- **Condition1:** Proximity to main road or railroad.
- **Condition2:** Proximity to main road or railroad (if a second is present).
- **BldgType:** Type of dwelling.
- **HouseStyle:** Style of dwelling.
- **OverallQual:** Overall material and finish quality.
- **OverallCond:** Overall condition rating.
- **YearBuilt:** Original construction date.
- **YearRemodAdd:** Remodel date.
- **RoofStyle:** Type of roof
- **RoofMatl:** Roof material
- **Exterior1st:** Exterior covering on house
- **Exterior2nd:** Exterior covering on house (if more than one material)
- **MasVnrType:** Masonry veneer type
- **MasVnrArea:** Masonry veneer area in square feet
- **ExterQual:** Exterior material quality
- **ExterCond:** Present condition of the material on the exterior
- **Foundation:** Type of foundation
- **BsmtQual:** Height of the basement
- **BsmtCond:** General condition of the basement
- **BsmtExposure:** Walkout or garden level basement walls
- **BsmtFinType1:** Quality of basement finished area
- **BsmtFinSF1:** Type 1 finished square feet
- **BsmtFinType2:** Quality of second finished area (if present)
- **BsmtFinSF2:** Type 2 finished square feet
- **BsmtUnfSF:** Unfinished square feet of basement area
- **TotalBsmtSF:** Total square feet of basement area
- **Heating:** Type of heating
- **HeatingQC:** Heating quality and condition
- **CentralAir:** Central air conditioning
- **Electrical:** Electrical system
- **1stFlrSF:** First Floor square feet
- **2ndFlrSF:** Second floor square feet
- **LowQualFinSF:** Low quality finished square feet (all floors)
- **GrLivArea:** Above grade (ground) living area square feet
- **BsmtFullBath:** Basement full bathrooms
- **BsmtHalfBath:** Basement half bathrooms
- **FullBath:** Full bathrooms above grade
- **HalfBath:** Half baths above grade
- **Bedroom:** Number of bedrooms above basement level
- **Kitchen:** Number of kitchens
- **KitchenQual:** Kitchen quality
- **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)
- **Functional:** Home functionality rating
- **Fireplaces:** Number of fireplaces
- **FireplaceQu:** Fireplace quality
- **GarageType:** Garage location
- **GarageYrBlt:** Year garage was built
- **GarageFinish:** Interior finish of the garage
- **GarageCars:** Size of garage in car capacity
- **GarageArea:** Size of garage in square feet
- **GarageQual:** Garage quality
- **GarageCond:** Garage condition
- **PavedDrive:** Paved driveway
- **WoodDeckSF:** Wood deck area in square feet
- **OpenPorchSF:** Open porch area in square feet
- **EnclosedPorch:** Enclosed porch area in square feet
- **3SsnPorch:** Three season porch area in square feet
- **ScreenPorch:** Screen porch area in square feet
- **PoolArea:** Pool area in square feet
- **PoolQC:** Pool quality
- **Fence:** Fence quality
- **MiscFeature:** Miscellaneous feature not covered in other categories
- **MiscVal:** $Value of miscellaneous feature
- **MoSold:** Month Sold
- **YrSold:** Year Sold
- **SaleType:** Type of sale
- **SaleCondition:** Condition of sale

# Cleaning the data

### Missing values

- MSZoning has  4  missing values 
- LotFrontage has  486  missing values 
- Alley has  2721  missing values 
- Utilities has  2  missing values 
- Exterior1st has  1  missing values 
- Exterior2nd has  1  missing values 
- MasVnrType has  24  missing values 
- MasVnrArea has  23  missing values 
- BsmtQual has  81  missing values 
- BsmtCond has  82  missing values 
- BsmtExposure has  82  missing values 
- BsmtFinType1 has  79  missing values 
- BsmtFinSF1 has  1  missing values 
- BsmtFinType2 has  80  missing values 
- BsmtFinSF2 has  1  missing values 
- BsmtUnfSF has  1  missing values 
- TotalBsmtSF has  1  missing values 
- Electrical has  1  missing values 
- BsmtFullBath has  2  missing values 
- BsmtHalfBath has  2  missing values 
- KitchenQual has  1  missing values 
- Functional has  2  missing values 
- FireplaceQu has  1420  missing values 
- GarageType has  157  missing values 
- GarageYrBlt has  159  missing values 
- GarageFinish has  159  missing values 
- GarageCars has  1  missing values 
- GarageArea has  1  missing values 
- GarageQual has  159  missing values 
- GarageCond has  159  missing values 
- PoolQC has  2909  missing values 
- Fence has  2348  missing values 
- MiscFeature has  2814  missing values 
- SaleType has  1  missing values 
- SalePrice has  1459  missing values 
 
If we compare the total number of the whole data frame with the amount of null values, we can see that there are four parameters with most of the values missing. This are Alley, PoolQC, Fence and MiscFeature. These parameters have more that 2000 missing values, which is a huge amount of data missing, normally it is good to erase these data, but let's see first is there is any reason why there is this huge amount.

- Alley: The parameter alley only tell us the type of alley access, thus NA only means that the house does not have alley, so we will replace the missing values in this by No Alley.
- PoolQC: The parameter PoolQC is the pool quality. The missing values at PoolQC means that there is not a Pool. We corrroborate this by checking the parameter PoolArea which tell us the Pool area in square feet, if this is zero, means that there is not a pool. Thus we found that the number of houses that does not have a Pool matches with the number of missing values in PoolQc (2909).
- Fence: Fence means Fence quality, so here as well the null values means that this house does not have a fence, so we will change the missing values by No Fence.
- Misc Feature: This parameter MiscFeature means Miscellaneous feature not covered in other categories, here the missing values means that these houses did not have any other special feature. So the missing values were replaced by No SpFeature. 

Other feature that has huge amount of missing values is FireplaceQu and LotFrontage

- FireplaceQu: The parameter FireplaceQu indicates the Fireplace quality, therefore NaN is this case means that there is not a Fireplace. In order to corroborate it, we checked the paramete Fireplaces, which indicates the number of fireplaces. We found that the number of houses that does not have fireplaces correspond to the number of missing values in FireplaceQu (1420). Therefore we replace the missing values in this feature with No Fireplaces. 

- LotFrontage: We will fill the missing values with the median.

Next parameters that has missing values are the ones related to the Garage, i.e. GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond. We found that there is a relation between the number of missing values for GarageType and the Garage Area. The latest give us the size of garage in square feet. The number of values that correspond to zero square feet are eauql to the missing values in GarageType. 

- GarageType, GarageFinish, GarageQual and GarageCond: We change the missing values for No Garage.
- GarageYrBlt: We change the missing value with 0 because never was built a garage.

The following missing values are found in the parameters that are relate to the bassement. Again we can check if these missing values are not related to the lack of bassement in the house.

- BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2: Althougth we only have 78 houses without basement, if we change the NaN values by No Basement in all the Basement parameters that has missing values, we will only have maximum 4 houses that could be missplaced which will not affect much the model.

The other missing values correspond to the feature MasVnrType and MasVnrArea. For this features Seems there is a lot of outliers, therefore instead of filling the missing values with the media it is better to take them as the people who provide the information ommits to fill these because they were not masonry veneer.

- MasVnrArea: the missing values were replace with 0.
- MasVnrType: The missing values were replaced with None.

For the rest of the features with missing values, as these features has maximum 4 missing values out of 2919 we will fill the data with the corresponding media value in each case for the cases where the feature is float64. For the categorical variables we fill it with the mode.  

# Feature engineering

### We create new columns:

- TotalSF: New column that contains the total square feet that consider the basement, 1st floor and 2nd floor
- Total_Bathrooms: New column that contains the total square feet of all the bathrooms in the house
- Total_porch_sf: New column that contains the total square feet that of the porch 

### Simplified information of another features:

- HasPool:  1 if PoolArea > 0 else 0
- Has2ndFloor:  1 if 2ndFlrSF > 0 else 0
- HasGarage:  1 if GarageArea > 0 else 0
- HasBsmt:  1 if TotalBsmtSF > 0 else 0
- HasFireplace:  1 if Fireplaces > 0 else 0

### Numerical variables

- Year features: We convert all the years in year features to number of years.
- soandre: New column that gives the amount of years that was remove the add and the year that actially was sold.
- timenosold: New column that gives the amount of years that pass away between when the house was built and sold

 Drop columns: BsmtHalfBath', 'BsmtFullBath','MoSold

### Numerical variables

Drop columns: LotArea','BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF', 'EnclosedPorch', 'MiscVal', '3SsnPorch', 'ScreenPorch', 'Total_porch_sf
