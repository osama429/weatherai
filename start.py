import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import re
import random
import string
from collections import Counter
from sklearn.model_selection import train_test_split

# Step 1: Create a synthetic dataset class
class StudentDataset(Dataset):
    def __init__(self, num_samples):
        print("Initializing Student Dataset...")  # Debug print
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self.universities = ['ITB', 'UGM', 'MIT', 'Harvard']
        self.generate_data()

    def generate_data(self):
        print("Generating Data...")  # Debug print
        for _ in range(self.num_samples):
            student_data = {
                'age': random.randint(18, 30),
                'GPA': round(random.uniform(2.0, 4.0), 2),
                'major': self.random_string(length=10)
            }

            university = random.choice(self.universities)
            label = self.universities.index(university)

            self.data.append(student_data)
            self.labels.append(label)

    def random_string(self, length=10):
        return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        student_data = self.data[idx]
        label = self.labels[idx]

        features = torch.tensor([student_data['age'], student_data['GPA']], dtype=torch.float32)

        return features, label

# Step 2: Define the neural network model
class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

# Step 3: Define the LSTM Model for Text Classification
class SAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, output_dim=4):  # 4 because there are 4 universities
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x

# Step 4: Tokenize and build vocabulary
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vocab(data):
    words = [word for text, _ in data for word in tokenize(text)]
    vocab = {word: i+2 for i, (word, _) in enumerate(Counter(words).most_common())}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab

def encode_text(text, vocab, max_len=6):
    tokenized = tokenize(text)
    encoded = [vocab.get(word, vocab['<unk>']) for word in tokenized]
    padded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
    return padded[:max_len]

# Step 5: Custom Dataset for Text Classification
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=6):
        self.data = [(encode_text(text, vocab, max_len), label) for text, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Step 6: GPA-based Recommendation Function with Roadmap for AP Classes
def recommend_based_on_gpa(gpa, university):
    if gpa >= 3.5:
        return {
            "message": f"Your GPA is excellent! For {university}, consider the following AP classes:",
            "roadmap": {
                "AP Physics": {
                    "description": "An advanced placement course in physics, ideal for students with a strong grasp of math and science.",
                    "prerequisites": "Algebra, basic chemistry",
                    "difficulty": "Hard",
                    "tips": "Study the math and conceptual fundamentals before diving into the experiments."
                },
                "AP Calculus": {
                    "description": "A college-level calculus course that covers differential and integral calculus.",
                    "prerequisites": "Precalculus, strong algebra skills",
                    "difficulty": "Hard",
                    "tips": "Focus on understanding the core concepts rather than memorizing formulas."
                },
                "AP Computer Science": {
                    "description": "An introductory college-level course in computer science, covering topics like algorithms and data structures.",
                    "prerequisites": "Basic programming skills",
                    "difficulty": "Moderate",
                    "tips": "Practice coding regularly and try solving real-world problems to strengthen your skills."
                },
                "AP Chemistry": {
                    "description": "A college-level chemistry course covering topics such as thermodynamics, equilibrium, and kinetics.",
                    "prerequisites": "Algebra, basic physics",
                    "difficulty": "Hard",
                    "tips": "Make sure to thoroughly understand the chemical reactions and their underlying principles."
                }
            }
        }
    elif gpa >= 2.5:
        return {
            "message": f"Your GPA is good. Focus on improving the subjects you find challenging. For {university}, consider these AP classes:",
            "roadmap": {
                "AP Computer Science": {
                    "description": "An introductory college-level course in computer science.",
                    "prerequisites": "Basic programming skills",
                    "difficulty": "Moderate",
                    "tips": "Review coding exercises and familiarize yourself with common algorithms."
                },
                "AP Biology": {
                    "description": "An introductory biology course covering topics like cell biology, genetics, and ecology.",
                    "prerequisites": "Basic knowledge of high school biology",
                    "difficulty": "Moderate",
                    "tips": "Focus on understanding key processes in the cell and the systems of the body."
                }
            }
        }
    else:
        return {
            "message": f"Your GPA is below average. Consider revisiting foundational subjects. For {university}, here are some AP classes to focus on:",
            "roadmap": {
                "AP Statistics": {
                    "description": "An introductory course in statistics, ideal for students who need to improve their math foundation.",
                    "prerequisites": "Basic algebra",
                    "difficulty": "Moderate",
                    "tips": "Try to understand the concepts of probability and sampling to make the subject easier."
                },
                "AP Biology": {
                    "description": "An introductory biology course.",
                    "prerequisites": "Basic knowledge of high school biology",
                    "difficulty": "Moderate",
                    "tips": "Use visual aids like diagrams to understand complex processes."
                }
            }
        }

# Step 7: Function for User Input Prediction with GPA and Roadmap
def predict_text_with_gpa(input_text, gpa, model, vocab):
    encoded_text = torch.tensor([encode_text(input_text, vocab)], dtype=torch.long)

    with torch.no_grad():
        output = model(encoded_text)
        prediction = torch.argmax(output, dim=1).item()

    universities = ["ITB", "UGM", "MIT", "Harvard"]
    university = universities[prediction]

    gpa_recommendation = recommend_based_on_gpa(gpa, university)

    print(f"\nInput: {input_text}")
    print(f"Predicted University: {university}")
    print(f"GPA-based AP Class Recommendation: {gpa_recommendation['message']}")

    # Printing the roadmap
    for subject, details in gpa_recommendation['roadmap'].items():
        print(f"\n{subject}:")
        print(f"  Description: {details['description']}")
        print(f"  Prerequisites: {details['prerequisites']}")
        print(f"  Difficulty: {details['difficulty']}")
        print(f"  Tips: {details['tips']}")

# Step 8: Main Training and Execution
def train_text_classification_model(text_data):
    vocab = build_vocab(text_data)
    train_data, test_data = train_test_split(text_data, test_size=0.2, random_state=42)

    train_dataset = TextDataset(train_data, vocab)
    test_dataset = TextDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    vocab_size = len(vocab)
    model = SAE(vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x_batch, y_batch = batch
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            outputs = model(x_batch)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return model, vocab

# Example text data
text_data = [
    ("how to get to mit", 2),
    ("how to get to harvard", 3),
    ("how to get in itb", 0),
    ("how to apply to ugm", 1),
    ("how to enroll in stanford", 3),
    ("how to learn coding", 0),
    ("how to draw myself", 0),
    ("how to be a software engineer", 0),
    ("how to become a software developer", 0),
    ("how to start programming", 0),
    ("how to learn python", 0),
    ("how to get a job as a programmer", 0),
    ("how to get an internship in software development", 0),
    ("how to become a game developer", 0),
    ("how to get into the technical university of munich", 2),
    ("how to apply for engineering courses at oxford", 2),
    ("how to join university of engineering and technology", 2),
    ("how to get into california institute of technology", 2),
    ("how to enroll in an engineering university", 2),
]

# Train the model
model, vocab = train_text_classification_model(text_data)

# Step 9: Taking User Input in a Loop
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break

    gpa = float(input("Enter your GPA (from 0.0 to 4.0): "))
    predict_text_with_gpa(user_input, gpa, model, vocab) 
    