import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import umap

sns.set(style="white", context="notebook", rc={"figure.figsize": (14, 10)})
penguins = pd.read_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
)
penguins = penguins.dropna()

reducer = umap.UMAP()
penguin_data = penguins[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

embedding = reducer.fit_transform(scaled_penguin_data)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[
        sns.color_palette()[x]
        for x in penguins.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
    ],
)
plt.gca().set_aspect("equal", "datalim")
plt.title("UMAP projection of the Penguin dataset", fontsize=24)
plt.show()
