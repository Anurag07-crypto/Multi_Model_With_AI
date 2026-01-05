import streamlit as st
import time
import torch 
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")
# Recreating all the classes
# ---------------------------------------------------------------------------------------
class encoder(nn.Module):
    def __init__(self,vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True)
    def forward(self, src):
        embedding = self.embedding(src)
        output, (h,c) = self.lstm(embedding)
        return output, (h,c)
# -------------------------------------------------------------------------------------
class decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)
    def forward(self, token, hidden, cell):
        embedded = self.embedding(token)
        output, (h,c) = self.lstm(embedded, (hidden, cell))
        pred = self.fc(output.squeeze(1))
        return pred, (h, c)
# -------------------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, teacher_forcing=0.7):
        encoder_output, (h,c) = self.encoder(src)
        input_token = trg[:,0]
        output = []
        for t in range(1, trg.size(1)):
            decoder_output, (h,c) = self.decoder(input_token.unsqueeze(1), h,c)
            output.append(decoder_output.unsqueeze(1))
            teacher = torch.rand(1).item() < teacher_forcing
            top_1 = decoder_output.argmax(1)
            
            input_token = trg[:,t] if teacher else top_1
        return torch.cat(output,dim=1)
# ---------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(r"models\bot_checkpoint.pth", map_location=device)
vocab = checkpoint["vocab"]
vocab_length=len(vocab)
enc = encoder(vocab_size=vocab_length)
dec = decoder(vocab_size=vocab_length)
model = Seq2Seq(enc, dec)
model.load_state_dict(checkpoint["model_state_dict"])
inv_vocab = {v: k for k, v in vocab.items()}
model.eval()
# ---------------------------------------------------------------------------------------
def text_to_indics(text, vocab):
    tokens = ["<SOS>"] + word_tokenize(text.lower()) + ["<EOS>"]
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
def response(model, prompt, max_len=70):
    src = text_to_indics(prompt, vocab)
    with torch.no_grad():
        _, (h,c) = model.encoder(src)
        input_token = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long)
        output_token = []
        for _ in range(max_len):
            pred, (h,c) = model.decoder(input_token, h,c)
            next_token = pred.argmax(dim=1).item()
            if next_token == vocab["<EOS>"]:
                break
            word = inv_vocab.get(next_token, "<UNK>")
            output_token.append(word)
            input_token = torch.tensor([[next_token]], dtype=torch.long)
    # clean_tokens = [w for w in output_token if w != "<UNK>"]
    return " ".join(output_token)
# -------------------------------------------------------

# -------------------------------------------------------
st.title("Hi user Chat Bot in your service")
# -------------------------------------------------------
st.markdown(
    """
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3, h4, h5, h6 {color: #fafafa;}
    .st-bb {background-color: transparent;}
    .css-1d391kg {padding-top: 1rem;}
    .stButton>button {
        background: linear-gradient(90deg, #ff4b1f 0%, #ff9068 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {transform: scale(1.03);}
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# -------------------------------------------------------------------
def response_generator(prompt):
    respond = response(model,prompt)
    for word in respond.split():
        yield word + " "
        time.sleep(0.005)
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
# -------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# -------------------------------------------------------
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
# -------------------------------------------------------
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
# -------------------------------------------------------