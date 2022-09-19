import numpy as np
import torch
from tqdm import tqdm
from train import StarModelFPN
from utils import DEVICE, score_iou, synthesize_data


def load_model():
    model = StarModelFPN()
    model.to(DEVICE)
    with open("./model.pickle", "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def eval(*, n_examples: int = 1024) -> None:
    model = load_model()
    scores = []
    for _ in tqdm(range(n_examples)):
        image, label = synthesize_data()
        with torch.no_grad():
            pred = model(torch.Tensor(image[None, None]).to(DEVICE))
        np_pred = pred[0].detach().cpu().numpy()
        scores.append(score_iou(np_pred, label))

    ious = np.asarray(scores, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print((ious > 0.7).mean())


if __name__ == "__main__":
    eval()
