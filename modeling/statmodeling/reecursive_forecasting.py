import numpy as np

def difference(dataset, interval=1):
    difference = []
    for i in range(interval, len(dataset)):
        difference.append(dataset[i] - dataset[i - interval])
    return difference


def inverse_difference(history, forecasts, interval):
    inverse_difference = []
    for i in range(len(forecasts)):
        index = -interval + i
        inverse_difference.append(forecasts[i] + history[index])
    return np.array(inverse_difference)

def recursive_forecast_ar(self, history, fitted_params, max_horizons):
    forecasts = list(history[-self.args.p:])

    if self.args.trend == 'c':
        const = fitted_params.get('const', 0)
    else:
        const = 0

    for i in range(max_horizons):
        forecast = const
        for j in range(self.args.p):
            forecast += fitted_params.get(f'{self.args.target}.L{j + 1}', 0) * forecasts[-j - 1]
        forecasts.append(forecast)
    return np.array(forecasts[self.args.p:])


def recursive_forecast_arima(self, history, next_observed, fitted_params, residuals, max_horizon):
    if self.args.d > 0:
        differenced = difference(history, self.args.d)
    else:
        differenced = history
    forecasts = list(differenced[-self.args.p:])
    residuals = list(residuals)
    next_observed = list(next_observed)

    for i in range(max_horizon):
        if self.args.trend == 'c':
            const = fitted_params.get('const', 0)
        else:
            const = 0
        ar_forecast = 0
        ma_forecast = 0

        for j in range(self.args.p):
            ar_forecast += fitted_params.get(f'ar.L{j + 1}', 0) * forecasts[-j - 1]
        # i = 0
        # 1 <= self.args.q (2) yes self.args.q - i 2 last two residuals. we have residuals to use.

        # i = 1
        # 2 <= self.args.q (2) yes self.args.q - i 1 last one residuals. we still have residuals to use while avoiding information leakage.

        # i = 2
        # 3 <= self.args.q (2) no won't be in the next function. we ran out of allowed residuals.
        if i + 1 <= self.args.q:
            for j in range(self.args.q - i):
                # self.args.q - i
                # as we get the recursive forecasts, we don't have access to some recent residuals, so we want to use the latest allowed residuals.
                ma_forecast += fitted_params.get(f'ma.L{j + 1}', 0) * residuals[(-j - 1)]

        forecast = const + ar_forecast + ma_forecast
        forecasts.append(forecast)
        if i == 0:
            residuals.append(next_observed[0] - forecast)

    if self.args.d > 0:
        return inverse_difference(history, forecasts[self.args.p:], self.args.d)
    else:
        return np.array(forecasts[self.args.p:])
        j
def recursive_forecast_sarima(self, history, next_observed, fitted_params, residuals, max_horizon):
        if self.args.d > 0:
            differenced = difference(history, self.args.d)
        else:
            differenced = history

        if self.args.D > 0:
            seasonal_difference = self.difference(differenced, self.args.s)
        else:
            seasonal_difference = differenced

        forecasts = list(seasonal_difference)
        residuals = list(residuals)
        next_observed = list(next_observed)

        for i in range(max_horizon):
            if self.args.trend == 'c':
                const = fitted_params.get('const', 0)
            else:
                const = 0

            ar_forecast = 0
            ma_forecast = 0
            ar_seasonal_forecast = 0
            ma_seasonal_forecast = 0

            for j in range(self.args.p):
                ar_forecast += fitted_params.get(f'ar.L{j + 1}', 0) * forecasts[-j - 1]

            #i = 0
            #1 <= self.args.q (2) yes self.args.q - i 2 last two residuals. we have residuals to use.

            # i = 1
            # 2 <= self.args.q (2) yes self.args.q - i 1 last one residuals. we still have residuals to use while avoiding information leakage.

            # i = 2
            # 3 <= self.args.q (2) no won't be in the next function. we ran out of allowed residuals.
            if i + 1 <= self.args.q:
                for j in range(self.args.q - i):
                    # self.args.q - i
                    # as we get the recursive forecasts, we don't have access to some recent residuals, so we want to use the latest allowed residuals.
                    ma_forecast += fitted_params.get(f'ma.L{j + 1}', 0) * residuals[(-j - 1)]

            for j in range(self.args.P):
                ar_seasonal_forecast += fitted_params.get(f'sar.L{j + 1}', 0) * forecasts[-self.args.s * (j + 1)]

            for j in range(self.args.Q):
                if i < self.args.s * (j + 1):
                    ma_seasonal_forecast += fitted_params.get(f'sma.L{j + 1}', 0) * residuals[-(self.args.s * (j + 1))]

            forecast = const + ar_forecast + ma_forecast + ar_seasonal_forecast + ma_seasonal_forecast
            forecasts.append(forecast)
            if i == 0:
                residuals.append(next_observed[0] - forecast) # append first horizon forecast from each window for next window forecasts.
                # we are not appending later residuals to avoid information leakage

        if self.args.D > 0:
            seasonal_difference = inverse_difference(history,
                                                   forecasts[self.args.p:],
                                                   self.args.s)
        else:
            seasonal_difference = np.array(forecasts[self.args.p:])

        if self.args.d > 0:
            return inverse_difference(history, seasonal_difference, self.args.d)[-max_horizon:]
        else:
            return seasonal_difference[-max_horizon:]