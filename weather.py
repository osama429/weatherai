import requests
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time

# Logger
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

# Fetch weather data
def get_weather(city="Jakarta", retries=3):
    attempt = 0
    while attempt < retries:
        try:
            url = f"https://wttr.in/{city}?format=%l+%c+%t"
            log(f"🔍 Requesting URL: {url}")
            res = requests.get(url, timeout=10)

            if res.status_code != 200:
                raise ValueError(f"Failed to fetch data, status code {res.status_code}")

            raw_text = res.text.strip()
            log(f"🧚 Raw response: {raw_text}")

            parts = raw_text.split()
            if len(parts) < 3:
                raise ValueError(f"Unexpected response format: {raw_text}")

            location = parts[0]
            condition = parts[1]
            temperature = int(parts[2].replace("+", "").replace("°C", ""))

            weather_data = {
                "location": location,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "condition": condition,
                "temperature_c": temperature
            }

            log(f"🌤️ Weather in {weather_data['location']} ({weather_data['time']}): {weather_data['condition']}, {weather_data['temperature_c']}°C")
            return weather_data

        except Exception as e:
            log(f"⚠️ Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(2)

    log("❌ All attempts to fetch weather data failed.")
    return None

# Weather Dataset
class WeatherDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        temperature = torch.tensor([item['temperature_c']], dtype=torch.float32)
        condition_encoded = torch.tensor([1.0 if '☔' in item['condition'] else 0.0], dtype=torch.float32)
        return condition_encoded, temperature  # Features and target

# Simple Neural Network Model
class WeatherPredictor(nn.Module):
    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # From weather condition (binary) to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Output: predicted temperature

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)
        return x

# Main flow
if __name__ == "__main__":
    log("📡 Fetching weather data...")
    weather_entry = get_weather("Jakarta")
    if weather_entry:
        log("📦 Preparing dataset...")
        dataset = WeatherDataset([weather_entry])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Initialize the model, loss function, and optimizer
        model = WeatherPredictor()
        criterion = nn.MSELoss()  # Mean Squared Error Loss (for regression)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training loop (for demonstration, training on a single data point)
        log("📈 Starting training...")
        for epoch in range(10):  # Dummy epochs for example
            for condition, temperature in dataloader:
                optimizer.zero_grad()
                prediction = model(condition)
                loss = criterion(prediction, temperature)
                loss.backward()
                optimizer.step()

            log(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Test the model (predict the temperature for a condition)
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            test_condition = torch.tensor([1.0])  # Assume it's rainy (for example)
            predicted_temp = model(test_condition)
            log(f"Predicted temperature for rainy condition: {predicted_temp.item():.2f}°C")
        log("✅ Done displaying weather data")