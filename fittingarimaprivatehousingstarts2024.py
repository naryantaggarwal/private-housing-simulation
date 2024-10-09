from pandas import DataFrame, read_excel, concat
from matplotlib import pyplot as plt
from statsmodels.tsa.api import acf, pacf, ARIMA, arma_order_select_ic

def correlogramAsDataFrame(correlogram, partial=False):
    # Find the correlogram with confidence intervals for each lag
    if partial:
        label="PACF"
    else:
        label="ACF"
    vals, confints = correlogram
    # Separate the lower bounds and upper bounds of the confidence intervals
    lower = confints.take(indices=0, axis=1)
    upper = confints.take(indices=1, axis=1)

    # Print the correlogram in text form for preciser reading
    return DataFrame({label: vals, "Lower": lower, "Upper": upper})

def plotCorrelogram(correlogram):
    # Plot the correlogram with the confidence intervals for each lag
    plt.plot(correlogram.iloc[:,0], color="gray", label=correlogram.columns[0])
    plt.plot(correlogram[["Lower"]], color="black", linestyle="dashed", label="Lower")
    plt.plot(correlogram[["Upper"]], color="black", linestyle="dotted", label="Upper")
    plt.legend()
    plt.show()

def diffSeries(series):
    return [series[i+1]-series[i] for i in range(len(series) - 1)]

# Access the data set; `sheet_name=None` means import all sheets from the workbook.
data=read_excel("https://www.census.gov/construction/nrc/xls/starts_cust.xlsx", sheet_name=None, header=[5, 5, 5, 5])

# Extract rows 0 to 779 and columns 0 to 1 from the "Seasonally Adjusted" sheet.
df = data["Seasonally Adjusted"].iloc[0:780, 0:2]
# Update the column headers
df.columns = ["Month", "PHS"]
# Get the phs `Series` by itself
phs = df["PHS"]
# Label the rows of `phs` by the month
phs.index = df["Month"]
# print(phs)

# Do an initial exploratory plot
plt.plot(phs)
plt.show()

# Find and display the autocorrelation function.
correlAcf = acf(phs, nlags=10, alpha=0.05)
acfDF = correlogramAsDataFrame(correlAcf)
plotCorrelogram(acfDF)
print(acfDF)

# Find and display the partial autocorrelation function.
correlPacf = pacf(phs, nlags=10, alpha=0.05)
pacfDF = correlogramAsDataFrame(correlPacf, partial=True)
plotCorrelogram(pacfDF)
print(pacfDF)

# Order selection
p = 1
d = 1
q = 2
model = ARIMA(phs, order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Use AIC/BIC to determine the model order
print(arma_order_select_ic(diffSeries(phs), max_ar=3, max_ma=5, ic=["aic", "bic"]))

# Use the last `holdout` months as the out-of-sample holdout/validation data set
holdout = 24
train_phs = phs[:-holdout]
test_phs = phs[-holdout:]

# Use the `train_phs` data to fit an ARIMA model.
p = 1
d = 1
q = 1
pre_model_train = ARIMA(train_phs, order=(p, d, q), freq="MS")
model_train = pre_model_train.fit()

# Forecast the values for the times in `test_phs` using the `train_phs` data
forecast = model_train.forecast(steps=holdout)
forecast.name = "Forecast"

# Define a convenience function to find the MAPE error metric
def mape(forecast, actual):
    return (abs(actual - forecast)/actual).mean()

# Find the MAPE values
mape_train = mape(model_train.fittedvalues, train_phs)
mape_test = mape(forecast, test_phs)

# Create a `DataFrame` with the actual values and the forecasted values for the holdout
validate_phs = concat([test_phs, forecast], axis="columns")

# Print the holdout `DataFrame` and the MAPE for both the training and testin data
print("Actual and forecasted PHS:")
print(validate_phs, '\n')
print("MAPE for the training data:", mape_train)
print("MAPE for the testing data:", mape_test, '\n')

# Plot the actual data, fitted values (estimated from `train_phs`), and forecasted values (over the holdout)
plt.plot(phs, color="gray", label="Actual")
plt.plot(model_train.fittedvalues, color="black", linestyle="dashed", label="Fitted")
plt.plot(forecast, color="black", linestyle="dotted", label="Forecasted")
plt.legend()
plt.show()

# Print the summary of the fitted model (including the values of the smoothing constants)
print(model_train.summary())