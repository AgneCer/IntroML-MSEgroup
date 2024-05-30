from http.client import responses
import requests
import json
import utils

def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")



if __name__ == "__main__":

    model = None
    filename = None
    current_model = None

    test_loader = None
    device = None

    best_model = utils.load_model(model, filename, current_model)
    
    preds = utils.return_predictions_dict(best_model, test_loader, device)

    res = {
        "images": preds,
        "groupname": "MSE-MagnificheSireneEnterprise"
    }

    submit(res)