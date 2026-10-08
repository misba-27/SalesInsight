import pandas as pd
import numpy as np

df = pd.read_csv("data/sales.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Remove duplicate records
df = df.drop_duplicates()

# Create Sales column
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# Create useful date columns
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")
df["Year"] = df["Date"].dt.year

print("==========================================")
print("       SALESINSIGHT DATA CLEANING")
print("==========================================")

print("\nData after cleaning and feature creation:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nInvalid Quantity values:", (df["Quantity"] <= 0).sum())
print("Invalid Unit_Price values:", (df["Unit_Price"] <= 0).sum())
# ==========================================
# NUMPY ANALYSIS
# ==========================================

sales_values = np.array(df["Sales"])

print("\n==========================================")
print("          NUMPY ANALYSIS")
print("==========================================")

print("Total Sales:", np.sum(sales_values))
print("Average Sales:", np.mean(sales_values))
print("Maximum Sale:", np.max(sales_values))
print("Minimum Sale:", np.min(sales_values))
print("Standard Deviation:", np.std(sales_values))
# ==========================================
# KEY ANALYSIS & RESULTS
# ==========================================

total_revenue = df["Sales"].sum()
average_sale = df["Sales"].mean()
highest_transaction = df["Sales"].max()
lowest_transaction = df["Sales"].min()

product_sales = df.groupby("Product")["Sales"].sum()
best_product = product_sales.idxmax()

category_sales = df.groupby("Category")["Sales"].sum()
best_category = category_sales.idxmax()

region_sales = df.groupby("Region")["Sales"].sum()
best_region = region_sales.idxmax()

daily_sales = df.groupby("Date")["Sales"].sum()
highest_sales_date = daily_sales.idxmax()

print("\n==========================================")
print("       KEY ANALYSIS & RESULTS")
print("==========================================")

print(f"Total Revenue       : ₹{total_revenue:,.2f}")
print(f"Average Sale        : ₹{average_sale:,.2f}")
print(f"Highest Transaction : ₹{highest_transaction:,.2f}")
print(f"Lowest Transaction  : ₹{lowest_transaction:,.2f}")
print(f"Best Product        : {best_product}")
print(f"Best Category       : {best_category}")
print(f"Best Region         : {best_region}")
print(f"Highest Sales Date  : {highest_sales_date.strftime('%d-%b-%Y')}")
# ==========================================
# SALES VISUALIZATIONS
# ==========================================

import matplotlib.pyplot as plt

# 1. Sales by Product
plt.figure(figsize=(10, 6))
product_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/product_sales.png")
plt.show()

# 2. Sales by Category
plt.figure(figsize=(8, 6))
category_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/category_sales.png")
plt.show()

# 3. Sales by Region
plt.figure(figsize=(8, 6))
region_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/region_sales.png")
plt.show()

# 4. Monthly Sales Trend
monthly_sales = (
    df.groupby(["Month", "Month_Name"])["Sales"]
    .sum()
    .reset_index()
    .sort_values("Month")
)

plt.figure(figsize=(10, 6))
plt.plot(
    monthly_sales["Month_Name"],
    monthly_sales["Sales"],
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/monthly_sales.png")
plt.show()

# 5. Sales Distribution by Category
plt.figure(figsize=(8, 8))
category_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Sales Distribution by Category")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/category_distribution.png")
plt.show()

print("\nAll visualizations created successfully!")
# ==========================================
# SAVE PROJECT OUTPUTS
# ==========================================

# Save cleaned dataset
df.to_csv("output/cleaned_sales.csv", index=False)

# Save key analysis results
results = {
    "Total Revenue": total_revenue,
    "Average Sale": average_sale,
    "Highest Transaction": highest_transaction,
    "Lowest Transaction": lowest_transaction,
    "Best Product": best_product,
    "Best Category": best_category,
    "Best Region": best_region,
    "Highest Sales Date": highest_sales_date.strftime("%Y-%m-%d")
}

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Metric", "Value"]
)

results_df.to_csv("output/analysis_results.csv", index=False)

print("\n==========================================")
print("       OUTPUT FILES SAVED SUCCESSFULLY")
print("==========================================")
print("Cleaned dataset  : output/cleaned_sales.csv")
print("Analysis results : output/analysis_results.csv")