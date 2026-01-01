import torch
import torch.nn as nn
import time
from model import RecurrentNet


def train(train_loader, learn_rate, hidden_dim, EPOCHS=100, model_type="GRU"):
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2

    # Unified model
    model = RecurrentNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=0.2,
        rnn_type=model_type,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.process_time()
        h = model.init_hidden(train_loader.batch_size)
        avg_loss = 0
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])

            model.zero_grad()
            x, label = x.float(), label.float()

            out, h = model(x, h)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss / len(train_loader):.4f}")
        epoch_times.append(time.process_time() - start_time)

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds")
    return model