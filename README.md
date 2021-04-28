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
