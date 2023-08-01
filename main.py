import uvicorn
from fastapi import FastAPI

import global_server_average
import global_server_median
from logger_config import logger
import test_model_median

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from global server"}


@app.get("/initiate_fl/average")
async def initiate_process_average():
    logger.info("Initiated FL type - Average")
    global_server_average.run_global_server()
    return {"message": "Average FL completed. Check log for more details"}


@app.get("/initiate_fl/median")
async def initiate_process_median():
    logger.info("Initiated FL type - Median")
    global_server_median.run_global_server()
    return {"message": "Median FL completed. Check log for more details"}

@app.get("/test_model/average")
async def test_average():
    logger.info("Initiated testing FL type - Average")
    test_model_median.test_model()
    return {"message": "Average test FL completed. Check log for more details"}

@app.get("/test_model/median")
async def test_median():
    logger.info("Initiated testing FL type - Median")
    test_model_median.test_model()
    return {"message": "Median test FL completed. Check log for more details"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)