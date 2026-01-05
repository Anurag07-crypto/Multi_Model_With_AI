# 🚀 Multi‑Mode AI Playground (Streamlit App)

A **multi‑page Streamlit application** that brings together several AI/ML models into one clean, interactive interface. This project is designed as an **AI showcase hub**—chatting with a bot, classifying plant leaves, detecting AI‑generated images, and identifying Indian birds, all from a single app.

Built with ❤️ using **Python, PyTorch, and Streamlit**.

---

## ✨ Features at a Glance

- 🤖 **Chat With Bot** – Seq2Seq LSTM‑based conversational chatbot
- 🌿 **Leaf Disease Classifier** – CNN model trained on plant leaf images
- 🧠 **Real vs AI Image Detector** – CNN to detect AI‑generated images
- 🐦 **Indian Bird Classifier** – Deep CNN for Indian bird species recognition
- 🧭 **Multi‑Page Navigation** – Clean navigation using Streamlit Pages
- 🎨 **Modern UI** – Custom styling, icons, and smooth UX

---

## 🗂️ Project Structure

```
Multi_Mode_With_A/
│
├── main.py                  # Main entry point (navigation + homepage)
├── page1.py                 # Chatbot page (Seq2Seq LSTM)
├── page2.py                 # Leaf classification page
├── page3.py                 # Real vs AI image detection page
├── page4.py                 # Indian bird classification page
│
├── models/
│   ├── bot_checkpoint.pth
│   ├── plant_leaf_classification.pth
│   ├── ai_vs_real.pth
│   └── Indian_Bird_Indentifier_model.pth
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Used

### 🤖 Chatbot (page1.py)
- Architecture: **Seq2Seq (LSTM Encoder–Decoder)**
- Tokenization: `nltk.word_tokenize`
- Trained with teacher forcing
- Streaming word‑by‑word responses for realism

### 🌿 Leaf Classifier (page2.py)
- Architecture: **Custom CNN**
- Classes: 38 plant disease categories
- Input Size: 244×244
- Output: Class name + confidence score

### 🧠 AI vs Real Image Detector (page3.py)
- Architecture: **CNN**
- Binary Classification: `AI` vs `Real`
- Input Size: 64×64
- Use Case: Detect AI‑generated images

### 🐦 Indian Bird Classifier (page4.py)
- Architecture: **Deep CNN (5 Conv Blocks)**
- Classes: 25 Indian bird species
- Input Size: 64×64

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – Web UI
- **PyTorch** – Model inference
- **Torchvision** – Image transforms
- **NLTK** – NLP preprocessing
- **PIL** – Image handling

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Multi_Mode_With_A.git
cd Multi_Mode_With_A
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
streamlit run main.py
```

---

## 📸 How It Works

1. Launch the app
2. Enter your name and discovery source
3. Choose a model from the homepage or navigation bar
4. Upload an image or start chatting
5. Get predictions with confidence scores

Simple. Clean. Powerful. ✨

---

## 🚧 Known Limitations

- Chatbot responses may contain `<UNK>` tokens (vocab limitation)
- Models are inference‑only (no live training)
- CPU inference by default

---

## 🔮 Future Improvements

- 🔥 Transformer‑based chatbot
- 🚀 GPU acceleration toggle
- 📊 Confidence visualizations
- 🌐 Deployment on Hugging Face / AWS
- 🧠 Unified model manager

---

## 🙌 Author

**Anurag**  
AI / ML Engineer in the making 🧠⚡  
Building, breaking, and rebuilding intelligent systems.

---

## ⭐ Support

If you like this project:
- ⭐ Star the repo
- 🍴 Fork it
- 💬 Share feedback

Because building AI is cool — but building **multi‑mode AI** is cooler 😎

---

Happy hacking 🚀

