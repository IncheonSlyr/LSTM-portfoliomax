from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


DATA_PATH = Path(__file__).with_name("institution_data.csv")


class GCN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 2)

    def forward(self, data: Data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


def build_graph_data(df: pd.DataFrame) -> Data:
    features = torch.tensor(df[["Performance", "Collaboration", "Resources"]].values, dtype=torch.float)
    labels = torch.tensor(df["GrowthLabel"].values, dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
             5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    return Data(x=features, edge_index=edge_index, y=labels)


def train_model(data: Data, epochs: int = 100) -> GCN:
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
    return model


def plot_predictions(data: Data, df: pd.DataFrame, predictions: torch.Tensor) -> None:
    graph = nx.Graph()
    graph.add_edges_from(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    positions = nx.spring_layout(graph, seed=42)
    colors = ["green" if predictions[i] == 1 else "red" for i in range(len(predictions))]
    labels = {i: df["Name"][i] for i in range(len(df))}

    plt.figure(figsize=(9, 7))
    nx.draw(
        graph,
        positions,
        with_labels=True,
        labels=labels,
        node_color=colors,
        node_size=800,
        font_color="white",
        font_size=8,
    )
    plt.title("Institution Growth Graph (Green = Improving, Red = Stagnant)")
    plt.show()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    data = build_graph_data(df)
    model = train_model(data)
    model.eval()
    predictions = model(data).argmax(dim=1)
    df["PredictedGrowth"] = predictions.tolist()
    print(df[["Name", "Type", "GrowthLabel", "PredictedGrowth"]])
    plot_predictions(data, df, predictions)


if __name__ == "__main__":
    main()
