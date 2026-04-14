from typing import Annotated, Dict, List, Literal

import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from stable_baselines3 import DQN, PPO

ACTIONS = [
    "EXIT", "HOLD", "AGGRESSIVE_LONG",
    "CONSERVATIVE_LONG", "TREND_FOLLOW", "MEAN_REVERT"
]
HOLDING_DAYS = [1, 3, 5, 10]
MODEL_PATHS = {
    "sentiment": "models/saved/sentiment_agent.zip",
}

models: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models["sentiment"] = load_model(MODEL_PATHS["sentiment"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load sentiment model from {MODEL_PATHS['sentiment']}: {exc}"
        ) from exc
    yield


app = FastAPI(
    title="RL Trading Agent API",
    description="API for serving prediction results from the sentiment trading agent.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class PredictRequest(BaseModel):
    state: Annotated[List[float], Field(min_length=30, max_length=30)]


class PredictResponse(BaseModel):
    model: str
    action_id: int
    strategy: str
    holding_days: int


class CompareResponse(BaseModel):
    predictions: Dict[str, PredictResponse]


def load_model(path: str):
    return PPO.load(path)


def predict_from_model(model, state: List[float]) -> PredictResponse:
    state_array = np.array(state, dtype=np.float32).reshape(1, 30)
    action, _ = model.predict(state_array)
    action_id = int(action[0])

    if action_id < 0 or action_id >= len(ACTIONS) * len(HOLDING_DAYS):
        raise ValueError(f"Unexpected action_id: {action_id}")

    return PredictResponse(
        model="",
        action_id=action_id,
        strategy=ACTIONS[action_id % len(ACTIONS)],
        holding_days=HOLDING_DAYS[action_id // len(HOLDING_DAYS)],
    )


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": "sentiment"}


@app.get("/web", tags=["Web UI"])
def web_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web/test", tags=["Web UI"])
def web_test(request: Request):
    try:
        # Sample state: 30 features (technical + sentiment)
        # Example: RSI=50, MACD=0, etc., sentiment scores around 0
        sample_state = [
            50.0,  # RSI
            0.0,   # MACD
            0.0,   # MACD signal
            0.0,   # MACD hist
            100.0, # SMA short
            100.0, # SMA long
            0.02,  # Volatility short
            0.02,  # Volatility long
            100.0, # BB upper
            100.0, # BB middle
            100.0, # BB lower
            0.5,   # ATR
            0.0,   # Trend strength
            0.0,   # Regime
        ] + [0.0] * 16  # Sentiment features (16 sentiment scores)
        
        prediction = predict_from_model(models["sentiment"], sample_state)
        prediction.model = "sentiment"
        
        result = {
            "status": "success",
            "message": "Model test successful",
            "sample_input": sample_state,
            "prediction": prediction.model_dump()
        }
        
        # For debugging, return JSON instead of template
        return result
        
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/web/predict", tags=["Web UI"])
def web_predict(request: Request, model: str = Form(...), state: str = Form(...)):
    try:
        state_list = [float(x.strip()) for x in state.split(",")]
        if len(state_list) != 30:
            raise ValueError(f"Expected 30 features, got {len(state_list)}")
        
        prediction = predict_from_model(models["sentiment"], state_list)
        prediction.model = "sentiment"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "status": "success",
                "model": model,
                "input_state": state_list,
                "prediction": prediction.dict()
            }
        })
    except Exception as exc:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "status": "error",
                "message": str(exc)
            }
        })


@app.get("/test", tags=["Testing"])
def test_model():
    """
    Run a simple test prediction on the sentiment model.
    Returns the prediction result for demonstration.
    """
    try:
        # Sample state: 30 features (technical + sentiment)
        # Example: RSI=50, MACD=0, etc., sentiment scores around 0
        sample_state = [
            50.0,  # RSI
            0.0,   # MACD
            0.0,   # MACD signal
            0.0,   # MACD hist
            100.0, # SMA short
            100.0, # SMA long
            0.02,  # Volatility short
            0.02,  # Volatility long
            100.0, # BB upper
            100.0, # BB middle
            100.0, # BB lower
            0.5,   # ATR
            0.0,   # Trend strength
            0.0,   # Regime
        ] + [0.0] * 16  # Sentiment features (16 sentiment scores)
        
        prediction = predict_from_model(models["sentiment"], sample_state)
        prediction.model = "sentiment"
        
        return {
            "status": "success",
            "message": "Model test successful",
            "sample_input": sample_state,
            "prediction": prediction.model_dump()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
