import uvicorn
from fastapi import FastAPI

import global_server_average
import global_server_median

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello from global server"}


@app.get("/initiate_fl_average")
async def initiate_process_average():
    print("Initiated FL type - Average")
    global_server_average.run_global_server()
    return {"message": "Average FL completed. Check log for more details"}


@app.get("/initiate_fl_median")
async def initiate_process_median():
    print("Initiated FL type - Median")
    global_server_median.run_global_server()
    return {"message": "Median FL completed. Check log for more details"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)