import torch

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from src.dataset import load_dataset
from src.model import DeepfakeAudioCNN



# Load dataset


X, y = load_dataset(
    "data/raw/real-vs-fake-human-voice-deepfake-audio"
)

print("Features:", X.shape)
print("Labels:", y.shape)



# Convert numpy to tensor

X = torch.tensor(
    X,
    dtype=torch.float32
)


y = torch.tensor(
    y,
    dtype=torch.long
)



# CNN requires:
# (samples, channels, features)

X = X.unsqueeze(1)



# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



train_data = TensorDataset(
    X_train,
    y_train
)


train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)



# Model

model = DeepfakeAudioCNN()



# Calculate class weights
real_count = (y_train == 0).sum()
fake_count = (y_train == 1).sum()


weights = torch.tensor(
    [
        1.0 / real_count,
        1.0 / fake_count
    ],
    dtype=torch.float32
)


loss_function = torch.nn.CrossEntropyLoss(
    weight=weights
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



# Training

epochs = 30


for epoch in range(epochs):

    total_loss = 0


    for audio, label in train_loader:


        optimizer.zero_grad()


        prediction = model(audio)


        loss = loss_function(
            prediction,
            label
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()



    print(
        f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}"
    )



# Save model

torch.save(
    model.state_dict(),
    "models/deepfake_audio_model.pth"
)


print("Model saved successfully!")
# Testing accuracy

model.eval()

with torch.no_grad():

    output = model(X_test)

    prediction = torch.argmax(
        output,
        dim=1
    )


    accuracy = (
        prediction == y_test
    ).float().mean()


print(
    f"Test Accuracy: {accuracy.item()*100:.2f}%"
)