# 🇪🇬 Digital Egypt Assistant

**An AI-powered chatbot that helps users navigate Egypt's digital government services in Arabic.**  
Built with Streamlit, LangChain, and multilingual embeddings, this assistant provides accurate, real-time answers about services available on the [Digital Egypt Portal](https://digital.gov.eg) — without linking to external sites.

---

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG) using Chroma vector store
- 💬 Arabic-language support via HuggingFace multilingual embeddings
- 🧠 Session-aware conversation memory using LangChain
- 🧾 Predefined quick-access service buttons (e.g., “Check vehicle fines”)
- 🔐 Supports multiple LLMs: OpenAI GPT-4o, Together LLaMA 3, Google Gemini
- ❌ No web links — full answers are provided from internal context only

---

## 📦 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/your-username/digital-egypt-assistant.git
cd digital-egypt-assistant
```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
3. **Run the app**
   ```bash
   streamlit run app.py
    ```
## 🔑 API Keys Required
To use the assistant, you must provide one of the following in the sidebar:

- `OpenAI API Key` (for GPT-4o)

- `Together AI API Key` (for LLaMA 3)

- `Google API Key` (for Gemini)

⚠️ Note: Gemini 2.5 Pro currently does not offer a free tier. Use `gemini-pro` or upgrade your plan.

📁 Project Structure
```bash
📦 digital-egypt-assistant/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── inflow/                 # ChromaDB persistence directory
└── README.md
```
## 🧠 Technologies
- Streamlit

- LangChain

- Chroma

- HuggingFace Embeddings


## 📝 License
MIT License.
Feel free to use, modify, and deploy.

## 🤝 Acknowledgments
Built by Mohamed Orabi as a prototype to enhance accessibility to Egypt's digital services through conversational AI.









