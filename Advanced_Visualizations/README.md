# Advanced Visualizations for HR Employee Attrition Analysis

This project focuses on performing Exploratory Data Analysis (EDA) and creating advanced, interactive visualizations using the IBM HR Employee Attrition dataset sourced from Kaggle. The goal is to uncover patterns, relationships, and insights within the employee data.

## Dataset

*   **Source:** [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/patelprashant/employee-attrition)
*   **File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
*   **Description:** Contains fictional data on various employee attributes, performance metrics, and attrition status.

## Libraries Used

*   **Data Manipulation:** `pandas`, `numpy`
*   **Visualization:** `plotly` (`plotly.graph_objects`, `plotly.express`, `plotly.figure_factory`), `wordcloud`, `matplotlib`

## Analysis Performed

The analysis covers several stages:

1.  **Data Loading and Preprocessing:**
    *   Loading the dataset using `pandas`.
    *   Initial data exploration (`head`, `describe`, `info`, `nunique`).
    *   Removing columns with single unique values (`Over18`, `StandardHours`, `EmployeeCount`).

2.  **Univariate Analysis:**
    *   Visualizing distributions of numerical features (e.g., Age) using histograms (`plotly.express.histogram`) and distribution plots (`plotly.figure_factory.create_distplot`).
    *   Visualizing counts of categorical features (e.g., BusinessTravel, Education) using histograms and enhanced bar charts with tables (`plotly.figure_factory.create_table`, `plotly.graph_objects.Bar`).

3.  **Bivariate Analysis:**
    *   Comparing categorical and numerical variables (e.g., Average Monthly Income by Department) using bar charts (`plotly.express.bar`), ensuring aggregation (like mean) is applied.
    *   Exploring relationships between two numerical variables (e.g., Age vs. Monthly Income) using scatter plots (`plotly.express.scatter`).

4.  **Multivariate Analysis:**
    *   Visualizing correlations between numerical features using heatmaps (`plotly.express.imshow`).
    *   Creating complex scatter plots incorporating categorical variables (e.g., Age vs. Monthly Income colored by Attrition) with marginal box/violin plots (`plotly.express.scatter`).
    *   Generating grouped and stacked bar charts to compare metrics across multiple categories (e.g., Average Monthly Income by Department and Attrition) (`plotly.express.bar`).
    *   Using facet plots (`facet_col`, `facet_row`) to break down visualizations across different categories (e.g., Income by Department, split by Business Travel and Gender).

5.  **Statistical Analysis:**
    *   Employing box plots (`plotly.express.box`) and violin plots (`plotly.express.violin`) to understand distributions and potential differences across categories (e.g., Monthly Income by Education Field and Attrition).

6.  **Other Visualizations:**
    *   Generating a Word Cloud (`wordcloud.WordCloud`) to visualize the frequency of terms in the 'EducationField' column.
    *   Creating Treemaps (`plotly.express.treemap`) to visualize hierarchical data (e.g., Monthly Income distribution across Department, Business Travel, and Gender).

## Key Insights Example

*   The analysis revealed trends such as a general correlation between lower age and lower monthly income.
*   Treemaps helped identify that the R&D department accounts for the highest total monthly income, with males who travel rarely representing the largest portion within that group.
