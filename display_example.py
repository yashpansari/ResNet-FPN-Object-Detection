import matplotlib.pyplot as plt
from utils import _corners, synthesize_data


def plot(ax, image, label, text):
    ax.imshow(image, cmap="gray")
    ax.set_title(text)
    if label.size > 0:
        bbox = _corners(*label)
        ax.fill(bbox[:, 0], bbox[:, 1], facecolor="none", edgecolor="r")


def prettify_np(a):
    return list(map(lambda x: int(x) if x.is_integer() else round(x, 2), a.tolist()))


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    image, label = synthesize_data(has_star=True)
    plot(ax[0], image, label, f"star with label {prettify_np(label)}")

    image, label = synthesize_data(has_star=False)
    plot(ax[1], image, label, "no star")

    image, label = synthesize_data(has_star=True, noise_level=0.1)
    plot(ax[2], image, label, f"star (less noise) with label {prettify_np(label)}")

    fig.savefig("example.png")
