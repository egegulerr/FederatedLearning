import json
from enum import Enum
from fastapi import FastAPI, Request
import pickle
import uvicorn
from fastapi.responses import FileResponse
from model import ModelWrapper
import numpy as np
import redis
from codecarbon import EmissionsTracker


NUMBER_OF_CLIENTS = 4


class ModelType(str, Enum):
    # TODO Bu classa ihtiyacmiz olmayabilir
    protobuffer = "protobuffer"
    json = "json"
    pickle = "pickel"


app = FastAPI()
redix = redis.Redis(host="localhost", port=6379)


@app.get("/geweights/{model_type}", description="Get ML model weights for JSON Model")
def get_weights(model_type: str):

    if model_type == ModelType.json.name:
        return FileResponse("../saved_models/model_json/weights_json.json")

    if model_type == ModelType.pickle.name:
        return FileResponse("../saved_models/model_pickel/model_weights.pkl")


@app.get("/getavgweights/{model_type}")
def get_avg_weights(model_type: str):

    if model_type == ModelType.json.name:
        weights = pickle.loads(redix.get("new_weights"))
        weights_dict = {}
        for i, weight in enumerate(weights):
            weights_dict["weights_" + str(i)] = weight.tolist()

        return json.dumps(weights_dict)


@app.post("/sendweights/{serialized_type}/{client_id}/{trained_data_length}")
async def send_weights(
    serialized_type: str, client_id: str, trained_data_length: int, request: Request
):

    file_content = await request.body()

    increment_client_count()
    redix.delete("new_weights")
    client_count = int(redix.get("count").decode())
    if client_count <= NUMBER_OF_CLIENTS:
        redix.set(client_id, file_content)
        redix.set(f"{client_id}_length", trained_data_length)
    if client_count == NUMBER_OF_CLIENTS:
        redix.set("count", 0)

        weights, data_lengths = prepare_weights()
        new_weights = aggregate_weights(weights, data_lengths)
        redix.set("new_weights", pickle.dumps(new_weights))

    print("Got the weights from the client. Setting weights")


@app.get("/", description="Health Check")
async def check_new_weights():
    available = False
    if redix.exists("new_weights") == 1:
        available = True
    return available


def prepare_weights():
    weights_np_array = []
    weights_bytes = [
        redix.get(str(client_index)) for client_index in range(NUMBER_OF_CLIENTS + 1)
    ]
    filtered_weights_bytes = list(filter(lambda x: x is not None, weights_bytes))
    weights_dict = [
        json.loads(json.loads(content)) for content in filtered_weights_bytes
    ]
    data_point_leghts = []
    for index, weights in enumerate(weights_dict, start=1):
        temp_array = []
        data_point_leghts.append(int(redix.get(f"{index}_length").decode()))
        for w in weights.values():
            temp_array.append(np.array(w))
        weights_np_array.append(temp_array)
    return weights_np_array, data_point_leghts


def increment_client_count():
    count = redix.get("count")
    if count:
        redix.set("count", int(count.decode()) + 1)
    else:
        redix.set("count", 1)

def aggregate_weights(weights_dict, data_point_legths):
    total_data_points = sum(data_point_legths)
    for index in range(len(weights_dict)):
        scalar = data_point_legths[index] / total_data_points
        for n_array_index in range(len(weights_dict[index])):
            weights_dict[index][n_array_index] *= scalar
    new_weights = [sum(x) for x in zip(*weights_dict)]
    return new_weights


def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    run_api()

    tracker.stop()