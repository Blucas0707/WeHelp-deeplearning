from fastapi import APIRouter

from web.schemas.predict import SubmitRequest

from web.utils.csv_writer import save_to_csv

router = APIRouter()


@router.post('/api/model/feedback')
async def feedback(req: SubmitRequest):
    save_to_csv(req.title, req.label)
    return {'message': 'ok'}
