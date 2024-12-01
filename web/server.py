import sys

sys.path.append("../src")

from typing import Union
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from data_loader import (
    load_innovix_floresland,
    load_innovix_elbonie,
    load_bristor_zegoland,
)
from Forecaster import Forecaster, ExternalAction
import dateutil.parser
import datetime
from copy import deepcopy
from dateutil.relativedelta import relativedelta

df = load_innovix_floresland("../src/data")
first_date = datetime.datetime(2018, 1, 1)
forecaster_orig = Forecaster()
forecaster_orig.fit(df)

curr_forecaster = None


app = FastAPI()


class Request(BaseModel):
    var: str
    amount: float
    time: str
    duration: int
    actionType: str


def iso_to_timestep(iso_time, first_date):
    query_time = dateutil.parser.isoparse(iso_time)
    return (
        (first_date.year - query_time.year) * 12 + first_date.month - query_time.month
    ) * -1


@app.post("/forecast/{num_steps}")
async def forecast_req(num_steps: int, requests: list[Request]):
    global curr_forecaster
    curr_forecaster = deepcopy(forecaster_orig)

    preds, stds = curr_forecaster.forecast(
        num_steps,
        external_actions=[
            ExternalAction(
                req.var,
                req.amount,
                iso_to_timestep(req.time, first_date),
                req.duration,
                req.actionType,
            )
            for req in requests
        ],
        maxlags=15,
        verbose=False,
    )

    return {
        "x": [*range(0, len(preds))],
        "y": preds.tolist(),
        "std": (stds * 1.96).tolist(),
        "first_date": first_date + relativedelta(months=df.iloc[-1]["Date"] + 1),
    }


@app.post("/shap/")
async def shap_req():
    result = curr_forecaster.explain_with_shap(20)
    # Sort the result by the shapley values
    features = result["X"]
    values = result["y"]

    # Sort the features by the shapley values
    features = [x for _, x in sorted(zip(values, features), reverse=True)]
    values = sorted(values, reverse=True)
    result["X"] = features[:5]
    result["y"] = values[:5]
    return result


app.mount("/", StaticFiles(directory="static"), name="static")
