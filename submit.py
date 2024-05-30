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
    
    # Passa `keep_full_label=True` per mantenere le etichette complete (id_name)
    # Passa `keep_full_label=False` per ottenere solo l'ID dalle etichette
    preds = utils.return_predictions_dict(best_model, test_loader, device, keep_full_label=False)

    res = {
        "images": preds,
        "groupname": "MSE-MagnificheSireneEnterprise"
    }

    submit(res)