# =============================================================================
# Employee Turnover Dataset: Data Cleaning Script
# Author: Jerald Reinshagen
# Date: 05/15/26
# Description: This script profiles and cleans the Employee Turnover Dataset
#              to prepare it for future analysis and model building.
#
# References:
#   - pandas Development Team. (2024). pandas documentation (v2.x).
#     https://pandas.pydata.org/docs/
#   - NumPy Developers. (2024). NumPy documentation.
#     https://numpy.org/doc/
# =============================================================================

import pandas as pd   # pandas: data manipulation and analysis (pandas Development Team, 2024)
import numpy as np    # numpy: numerical operations (NumPy Developers, 2024)

# =============================================================================
# SECTION 1: LOAD THE DATASET
# =============================================================================

# Load the raw CSV file into a DataFrame
df = pd.read_csv('Employee_Turnover_Dataset.csv')
print("=" * 60)
print("SECTION 1: RAW DATASET OVERVIEW")
print("=" * 60)
print(f"Number of records (rows): {df.shape[0]}")
print(f"Number of variables (columns): {df.shape[1]}")
print(f"Column names: {df.columns.tolist()}")
print()


# =============================================================================
# SECTION 2: INSPECT FOR DATA QUALITY ISSUES
# =============================================================================

print("=" * 60)
print("SECTION 2: DATA QUALITY INSPECTION")
print("=" * 60)

# --- 2A. DUPLICATE ENTRIES ---
# df.duplicated() returns True for any row that is an exact copy of a prior row.
# Duplicate employee records can skew analysis and must be removed.
duplicate_count = df.duplicated().sum()
print(f"[Duplicates] Number of fully duplicated rows: {duplicate_count}")

# --- 2B. MISSING VALUES ---
# df.isnull().sum() counts NaN (missing) values in each column.
# Missing values can cause errors in calculations and models.
print()
print("[Missing Values] Count per column:")
print(df.isnull().sum())

# --- 2C. INCONSISTENT ENTRIES ---
# Categorical columns may have the same value spelled or formatted differently.
# For example, 'Direct Deposit', 'Direct_Deposit', and 'DirectDeposit' all
# represent the same paycheck method but would be treated as separate categories.
print()
print("[Inconsistent Entries] Unique values in PaycheckMethod:")
print(df["PaycheckMethod"].unique())
print()
print("[Inconsistent Entries] Unique values in JobRoleArea:")
print(df["JobRoleArea"].unique())

# --- 2D. FORMATTING ERRORS ---
# HourlyRate is stored as a string with a leading '$' and trailing space,
# which prevents numeric operations. The column name itself also has a
# trailing space ('HourlyRate '), which must be corrected.
print()
print("[Formatting] HourlyRate sample (raw):", df["HourlyRate "].head(3).tolist())
print("[Formatting] HourlyRate dtype:", df["HourlyRate "].dtype)

# --- 2E. OUTLIERS ---
# Values that fall outside a realistic or logical range are considered outliers.
# Negative values for distance or salary are physically impossible and indicate
# data entry errors. Extreme high values may also be erroneous.
# The IQR (interquartile range) method is used to detect statistical outliers:
#   Lower fence = Q1 - 1.5 * IQR
#   Upper fence = Q3 + 1.5 * IQR
print()
print("[Outliers] Negative AnnualSalary count:",
      (df["AnnualSalary"] < 0).sum())
print("[Outliers] Negative DrivingCommuterDistance count:",
      (df["DrivingCommuterDistance"] < 0).sum())

# IQR outlier check for DrivingCommuterDistance
Q1 = df["DrivingCommuterDistance"].quantile(0.25)
Q3 = df["DrivingCommuterDistance"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outlier_distance_count = ((df["DrivingCommuterDistance"] < lower_fence) |
                           (df["DrivingCommuterDistance"] > upper_fence)).sum()
print(f"[Outliers] DrivingCommuterDistance IQR fences: [{lower_fence:.2f}, {upper_fence:.2f}]")
print(f"[Outliers] DrivingCommuterDistance values outside IQR fences: {outlier_distance_count}")

# IQR outlier check for AnnualSalary
Q1_sal = df["AnnualSalary"].quantile(0.25)
Q3_sal = df["AnnualSalary"].quantile(0.75)
IQR_sal = Q3_sal - Q1_sal
lower_sal = Q1_sal - 1.5 * IQR_sal
upper_sal = Q3_sal + 1.5 * IQR_sal
outlier_salary_count = ((df["AnnualSalary"] < lower_sal) |
                         (df["AnnualSalary"] > upper_sal)).sum()
print(f"[Outliers] AnnualSalary IQR fences: [{lower_sal:.2f}, {upper_sal:.2f}]")
print(f"[Outliers] AnnualSalary values outside IQR fences: {outlier_salary_count}")
print()


# =============================================================================
# SECTION 3: CLEAN THE DATASET
# =============================================================================

print("=" * 60)
print("SECTION 3: DATA CLEANING")
print("=" * 60)

# Work on a copy so the raw DataFrame is preserved for comparison
df_clean = df.copy()

# --- 3A. FIX FORMATTING ERRORS FIRST ---
# Renaming the column with a trailing space before other operations.
# Technique: df.rename() — corrects column names so they can be referenced
# reliably without hidden whitespace characters causing KeyErrors.
df_clean.rename(columns={"HourlyRate ": "HourlyRate"}, inplace=True)
print("[Formatting Fix] Renamed 'HourlyRate ' -> 'HourlyRate'")

# Remove the '$' symbol and trailing/leading whitespace from HourlyRate,
# then convert to a float so numeric operations can be performed.
# Technique: str.replace() + astype(float) — transforms a string-formatted
# currency field into a usable numeric type. (pandas Development Team, 2024)
df_clean["HourlyRate"] = (
    df_clean["HourlyRate"]
    .str.replace("$", "", regex=False)
    .str.strip()
    .astype(float)
)
print("[Formatting Fix] HourlyRate converted from string ('$XX.XX') to float")

# --- 3B. REMOVE DUPLICATE ENTRIES ---
# Technique: df.drop_duplicates() — removes rows that are exact copies of a
# prior row, keeping the first occurrence. This ensures each employee record
# appears only once, preventing inflated counts and biased statistics.
# (pandas Development Team, 2024)
rows_before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index(drop=True, inplace=True)
rows_after = len(df_clean)
print(f"\n[Duplicates Removed] Rows before: {rows_before} | Rows after: {rows_after} "
      f"| Removed: {rows_before - rows_after}")

# --- 3C. FIX INCONSISTENT ENTRIES ---
# Technique: str.strip() + str.replace() + map() — standardizes categorical
# values so that all variations of the same category are unified into one
# consistent label. This prevents the same group from being split across
# multiple categories in analysis. (pandas Development Team, 2024)

# Standardize PaycheckMethod
# All variations of 'Mail Check' and 'Direct Deposit' are unified below.
paycheck_map = {
    "Mail Check":     "Mail Check",
    "Mail_Check":     "Mail Check",
    "MailedCheck":    "Mail Check",
    "Mailed Check":   "Mail Check",
    "Direct Deposit": "Direct Deposit",
    "Direct_Deposit": "Direct Deposit",
    "DirectDeposit":  "Direct Deposit",
}
df_clean["PaycheckMethod"] = df_clean["PaycheckMethod"].str.strip().map(paycheck_map)
print("\n[Inconsistency Fix] PaycheckMethod standardized.")
print("  Unique values now:", df_clean["PaycheckMethod"].unique())

# Standardize JobRoleArea
# Variations like 'Information_Technology', 'InformationTechnology', and
# 'Information Technology' are all mapped to the same clean label.
jobrole_map = {
    "Research":               "Research",
    "Information_Technology": "Information Technology",
    "InformationTechnology":  "Information Technology",
    "Information Technology": "Information Technology",
    "Sales":                  "Sales",
    "Human_Resources":        "Human Resources",
    "HumanResources":         "Human Resources",
    "Human Resources":        "Human Resources",
    "Laboratory":             "Laboratory",
    "Manufacturing":          "Manufacturing",
    "Healthcare":             "Healthcare",
    "Marketing":              "Marketing",
}
df_clean["JobRoleArea"] = df_clean["JobRoleArea"].str.strip().map(jobrole_map)
print("[Inconsistency Fix] JobRoleArea standardized.")
print("  Unique values now:", df_clean["JobRoleArea"].unique())

# --- 3D. HANDLE MISSING VALUES ---
# Technique: median imputation for numeric columns, mode imputation for
# categorical columns.
#
# Median is preferred over mean for numeric imputation because it is
# resistant to the influence of outliers — important given that AnnualSalary
# and DrivingCommuterDistance have extreme values. (pandas Development Team, 2024)
#
# Mode (most frequent value) is used for categorical columns because it
# preserves the existing distribution of categories without introducing new
# values that did not exist in the dataset.

# Numeric columns with missing values
numeric_missing = ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"]
for col in numeric_missing:
    median_val = df_clean[col].median()
    missing_count = df_clean[col].isnull().sum()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"\n[Missing Value Fix] '{col}': {missing_count} NaN(s) filled with median ({median_val})")

# Categorical column with missing values
cat_missing = ["TextMessageOptIn"]
for col in cat_missing:
    mode_val = df_clean[col].mode()[0]
    missing_count = df_clean[col].isnull().sum()
    df_clean[col] = df_clean[col].fillna(mode_val)
    print(f"\n[Missing Value Fix] '{col}': {missing_count} NaN(s) filled with mode ('{mode_val}')")

# --- 3E. HANDLE OUTLIERS ---
# Technique: IQR-based capping (Winsorization) — values below the lower fence
# or above the upper fence are clipped to the fence value rather than removed.
#
# Capping is chosen over deletion because deleting outlier rows would reduce
# the dataset size significantly and potentially remove otherwise valid employee
# records. Capping retains all records while constraining extreme values to a
# realistic range. (pandas Development Team, 2024)
#
# Negative values for AnnualSalary and DrivingCommuterDistance are physically
# impossible (a salary or distance cannot be negative), so they are treated as
# data entry errors and corrected via capping to the lower IQR fence (which
# will be at minimum 0 after capping).

numeric_cols_outlier = ["AnnualSalary", "DrivingCommuterDistance"]

for col in numeric_cols_outlier:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Ensure the lower fence is not negative for distance/salary
    lower = max(lower, 0)

    before_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    print(f"\n[Outlier Fix] '{col}': {before_outliers} value(s) capped to [{lower:.2f}, {upper:.2f}]")


# =============================================================================
# SECTION 4: POST-CLEANING SUMMARY
# =============================================================================

print()
print("=" * 60)
print("SECTION 4: POST-CLEANING SUMMARY")
print("=" * 60)
print(f"Final number of records: {df_clean.shape[0]}")
print(f"Final number of variables: {df_clean.shape[1]}")
print()
print("Remaining missing values per column:")
print(df_clean.isnull().sum())
print()
print("Sample of cleaned data (first 5 rows):")
print(df_clean.head())


# =============================================================================
# SECTION 5: EXPORT CLEANED DATASET
# =============================================================================

# Export the cleaned DataFrame to a CSV file.
# index=False prevents pandas from writing the row index as an extra column.
output_path = "Employee_Turnover_Cleaned.csv"
df_clean.to_csv(output_path, index=False)
print()
print(f"[Export] Cleaned dataset saved to: '{output_path}'")
print("Script complete.")
