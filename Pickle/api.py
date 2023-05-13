from enum import Enum
from fastapi import FastAPI, HTTPException, Request
import pickle
import uvicorn
from fastapi.responses import FileResponse, Response
from model import ModelWrapper
import redis
from codecarbon import EmissionsTracker

NUMBER_OF_CLIENTS = 4


class ModelType(str, Enum):
    # TODO Bu classa ihtiyacmiz olmayabilir
    protobuffer = "protobuffer"
    json = "json"
    pickle = "pickle"


app = FastAPI()
redix = redis.Redis(host="localhost", port=6379)


@app.get("/", description="Weights available check")
async def check_new_weights():
    available = False
    if redix.exists("new_weights") == 1:
        available = True
    return available


@app.get(
    "/getmodel/{model_type}",
    description="Get ML model depending on chosen serialized method",
)
def get_model(model_type: ModelType):

    if model_type is ModelType.pickle:
        return FileResponse("saved_models/model_pickel/model.pkl")

    return HTTPException(
        status_code=404, detail=f"Wanted serialization method does not exist"
    )


@app.get("/geweights/{model_type}", description="Get ML model weights for JSON Model")
def get_weights(model_type: str):

    if model_type == ModelType.pickle.name:
        return FileResponse("saved_models/model_pickel/model_weights.pkl")


@app.get("/getavgweights/{model_type}")
def get_avg_weights(model_type: str):

    if model_type == ModelType.pickle.name:
        return Response(redix.get("new_weights"))


@app.post("/sendweights/{serialized_type}/{client_id}/{trained_data_length}")
async def send_weights(
    serialized_type: str, client_id: str, trained_data_length: int, request: Request
):

    data: bytes = await request.body()

    increment_client_count()
    redix.delete("new_weights")
    client_count = int(redix.get("count").decode())
    if client_count <= NUMBER_OF_CLIENTS:
        redix.set(client_id, data)
        redix.set(f"{client_id}_length", trained_data_length)
    if client_count == NUMBER_OF_CLIENTS:
        redix.set("count", 0)

        weights, data_lengths = prepare_weights()
        new_weights = aggregate_weights(weights, data_lengths)
        redix.set("new_weights", pickle.dumps(new_weights))

    print("Got the weights from the client. Setting weights")


def prepare_weights():
    weights_bytes = [
        redix.get(str(client_index)) for client_index in range(NUMBER_OF_CLIENTS + 1)
    ]
    filtered_weights_bytes = list(filter(lambda x: x is not None, weights_bytes))
    weights_dict = [pickle.loads(content) for content in filtered_weights_bytes]
    data_point_leghts = []
    for index, _ in enumerate(weights_dict, start=1):
        data_point_leghts.append(int(redix.get(f"{index}_length").decode()))
    return weights_dict, data_point_leghts


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
