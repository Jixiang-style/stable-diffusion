import logging


from fastapi.responses import FileResponse
from fastapi import FastAPI

from ldm.api.model import Txt2imgOption

from ldm.api.processing import select_seed_randomly, Processing

app = FastAPI()


@app.post("/txt2img")
async def txt2img(opt: Txt2imgOption):
    if opt.seed is None:
        opt.seed = select_seed_randomly()
    logging.info('Path: /txt2img, Body: %s' % opt)
    img_path = Processing.txt2img(opt)
    if img_path is not None:
        return {'status': 'success', 'result': img_path}
    return {'status': 'failed'}


@app.get("/get_image")
async def get_image(image_path: str):
    return FileResponse(image_path, media_type="image/png")
