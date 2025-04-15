from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.schemas.predict import SubmitRequest

from web.utils.csv_writer import save_to_csv
from web.models.predict import predict_label, get_suggested_categories

router = APIRouter()


@router.get('/api/model/prediction')
async def predict(title: str):
    prediction = predict_label(title)
    suggestions = get_suggested_categories(prediction)
    return {'prediction': prediction, 'suggestions': suggestions}
