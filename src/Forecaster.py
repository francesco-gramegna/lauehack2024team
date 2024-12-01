import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
from typing import List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
    ExpSineSquared,
    DotProduct,
    Matern,
)
from statsmodels.tsa.vector_ar.var_model import VAR
import shap


def fill_missing_values_with_gp(df):
    """
    Fill missing values in the dataframe using Gaussian Processes.
    """
    with warnings.catch_warnings(action="ignore"):

        for col in tqdm(df.columns):
            if df[col].isna().sum() > 0:  # Process columns with missing values
                # Extract observed data
                observed = df[col].dropna()
                X_observed = np.array(observed.index).reshape(-1, 1)  # Reshape index
                y_observed = observed.values

                # Extract missing data locations
                X_missing = np.array(df[df[col].isna()].index).reshape(-1, 1)

                # Define Gaussian Process
                # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                PARAM_RANGES = (1e-1, 1e1)

                kernel = WhiteKernel(0.1, PARAM_RANGES)
                kernel += ConstantKernel(1, PARAM_RANGES) * RBF(1, PARAM_RANGES)
                kernel += ConstantKernel(1, PARAM_RANGES) * DotProduct(1, PARAM_RANGES)
                gp = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=10, alpha=1e-2
                )

                # Fit GP to observed data
                gp.fit(X_observed, y_observed)

                # Predict missing values
                y_missing_pred, sigma = gp.predict(X_missing, return_std=True)

                # Fill missing values in the original dataframe
                df.loc[df[col].isna(), col] = y_missing_pred
    return df


class ExternalAction:
    def __init__(self, var: str, amount, time, duration, type: str):
        self.var = var
        self.amount = amount
        self.time = time
        self.duration = duration
        self.type = type

        if type not in ["set", "update"]:
            raise Exception()

    def is_applicable(self, time):
        return time >= self.time and time <= self.time + self.duration


class Forecaster:
    def __init__(self):
        # kernel = WhiteKernel(0.1, (1e-2, 1e2))
        # kernel += ConstantKernel(1, (1e-8, 1e3)) * RBF(1, (1e-3, 1e2))
        # kernel += ConstantKernel(1, (1e-8, 1e3)) * ExpSineSquared(1, 1, (1e-2, 1e2), (1e-2, 1e2))
        # kernel += ConstantKernel(1, (1e-8, 1e3)) * DotProduct(1, (1e-2, 1e2))
        # kernel = WhiteKernel(0.1, (1e-3, 1e3))
        # kernel += ConstantKernel(1, (1e-4, 1e4)) * RBF(1, (1e-3, 1e3))
        # kernel += ConstantKernel(1, (1e-4, 1e4)) * ExpSineSquared(1, 1, (1e-2, 1e2), (1e-2, 1e2))
        # kernel += ConstantKernel(1, (1e-4, 1e4)) * DotProduct(1, (1e-3, 1e3))

        PARAM_RANGES = (1e-9, 1e9)
        kernel = WhiteKernel(0.1, PARAM_RANGES)
        kernel += ConstantKernel(1, PARAM_RANGES) * RBF(1, PARAM_RANGES)
        kernel += ConstantKernel(1, PARAM_RANGES) * DotProduct(1, PARAM_RANGES)

        # PARAM_RANGES = (1e-4, 1e4)
        # kernel = WhiteKernel(0.1, PARAM_RANGES)
        # kernel += ConstantKernel(1, PARAM_RANGES) * RBF(1, PARAM_RANGES)
        # kernel += ConstantKernel(1, PARAM_RANGES) * DotProduct(1, PARAM_RANGES)

        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9, random_state=42
        )
        self.y_mean = None
        self.y_std = None

    def fit(self, df):
        dff = df.set_index(["Date"], drop=False)

        X = dff.drop(columns=["ex_factory_volumes"])
        y = dff["ex_factory_volumes"]

        X = X.fillna(X.mean())
        # X = fill_missing_values_with_gp(X)

        self.X_mean = X.mean()
        self.X_std = X.std()
        orig_date = X["Date"].copy()
        self.X = (X - self.X_mean) / (self.X_std + 1e-9)
        self.X["Date"] = orig_date

        self.y_mean = y.mean()
        self.y_std = y.std()
        self.y = (y - self.y_mean) / (self.y_std)

    def forecast(
        self,
        steps: int,
        external_actions: List[ExternalAction] = [],
        maxlags=15,
        verbose=False,
    ):
        stds = []
        preds = []
        start_time = int(self.X.iloc[-1]["Date"]) + 1

        with warnings.catch_warnings():
            for _ in tqdm(range(steps), disable=not verbose):
                model = VAR(self.X)
                results = model.fit(maxlags=maxlags)

                if not results.is_stable:
                    raise ValueError("Model is unstable. Check data or adjust lags.")

                forecast_input = self.X.values[-results.k_ar :]
                forecast = results.forecast(y=forecast_input, steps=1)
                new_sample = {col: val for col, val in zip(self.X.columns, forecast[0])}
                new_sample["Date"] = start_time

                # Apply feasible actions
                for e in external_actions:
                    if e.is_applicable(start_time):
                        # Apply action
                        if e.type == "set":
                            new_sample[e.var] = e.var
                        elif e.type == "update":
                            # De-normalize
                            new_sample[e.var] = (
                                new_sample[e.var] * self.X_std[e.var]
                                + self.X_mean[e.var]
                            )
                            new_sample[e.var] *= e.amount
                            # Ri-normalize
                            new_sample[e.var] = (
                                new_sample[e.var] - self.X_mean[e.var]
                            ) / self.X_std[e.var]

                self.X = pd.concat([self.X, pd.DataFrame.from_dict([new_sample])])

                start_time += 1

            orig_date = self.X["Date"].copy()
            self.X = self.X * self.X_std + self.X_mean
            self.X["Date"] = orig_date

            self.y = self.y * self.y_std + self.y_mean

            self.gp.fit(self.X.iloc[: len(self.y)], self.y)
            preds, stds = self.gp.predict(self.X.iloc[len(self.y) :], return_std=True)
            self.y = pd.concat([self.y, pd.Series(preds)])

        return preds, stds

    def explain_with_shap(self, samples, plot_type="bar"):

        if self.gp is None or not hasattr(self.gp, "kernel_"):
            raise ValueError(
                "Il modello GP non è stato addestrato. Chiama 'fit' prima di utilizzare SHAP."
            )

        # Seleziona un sottoinsieme di dati da spiegare
        X_sample = (self.X).sample(n=samples, random_state=42)

        # Crea un Explainer SHAP per GaussianProcessRegressor
        explainer = shap.KernelExplainer(self.gp.predict, X_sample.values)

        # Calcola i valori SHAP
        shap_values = explainer.shap_values(X_sample.values)

        if "Date" in X_sample.columns:
            date_idx = X_sample.columns.get_loc("Date")
            X_sample = X_sample.drop(columns=["Date"])
            shap_values = np.delete(shap_values, date_idx, axis=1)

        shap_dict = {
            "X": [feature for feature in X_sample.columns],
            "y": [
                float(np.mean(np.abs(shap_values[:, idx])))
                for idx in range(len(X_sample.columns))
            ],
        }

        # Visualizza i risultati
        # if plot_type == "bar":
        #    shap.summary_plot(shap_values, X_sample, plot_type="bar")
        # elif plot_type == "summary":
        #    shap.summary_plot(shap_values, X_sample)
        # elif plot_type == "force":
        #    shap.force_plot(explainer.expected_value, shap_values, self.X.iloc[:samples])
        # else:
        #    raise ValueError(f"Tipo di plot SHAP '{plot_type}' non supportato.")

        return shap_dict
