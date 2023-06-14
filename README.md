# World-Bank-Data-ML
# Problem Statement
We will again look at exploring public data from the World Bank, and specifically
country-by-country indicators related to climate change: https://data.worldbank.org/
topic/climate-change. You may find additional relevant indicators (e.g. GDP per
capita) using the complete list https://data.worldbank.org/indicator. Note that
not all countries have entries for the most recent year(s).
Your goal is to:
• Find interesting clusters of data. Note that for meaningful clusters it is often a good
idea to look at normalised values like GDP per capita, CO2 production per head,
CO2 per $ of GDP or fraction of a sector. You might look at most recent values or
compare recent values with, say, values 30 or 40 years ago or use total historic values.
Hint: Applying clustering to one time series will not give you valuable insights.
Use at least one of the clustering methods from the lecture. Clustering works best
when the data are normalised (see Practical 8). Note that you usually want to show
the original (not normalised values) to display the clustering results. One way to
achieve this is to add the classifications as a new column to the dataframes and use
logical slicing. Produce a plot showing cluster membership and cluster centres using
pyplot.
• Create simple model(s) fitting data sets with curve_fit. This could be fits of time
series, but also, say, one attribute as a function of another. Keep the model simple
(e.g., exponential growth, logistic function, low order polynomials). Use the model
for predictions, e.g. values in ten or twenty years time including confidence ranges.
Use the attached function err_ranges to estimate lower and upper limits of the

confidence range and produce a plot showing the best fitting function and the con-
fidence range.

• You do not need to use the same data sets for clustering and fitting, but one approach
could be: find clusters of countries, pick one country from each cluster and compare
countries from one cluster and find similarities and differences, compare countries
from different clusters or pick a few countries from one cluster and compare with
other regions. Investigate trends. Do you find similar or different trends in different
clusters? Do you find similar or different trends in countries from the same cluster?
• You do not need to focus on CO2 or climate change. The choice of topic is yours.
