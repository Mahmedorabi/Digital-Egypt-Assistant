import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from uuid import uuid4

# ========== Page Config ==========
st.set_page_config(page_title="المساعد الرقمي", layout="wide")
st.title("💬 المساعد الرقمي لخدمات مصر")
st.markdown("اسألني عن أي خدمة رقمية متاحة على بوابة مصر الرقمية")
session_id = uuid4()
# ========== Sidebar ==========
st.sidebar.title("🔧 إعدادات النموذج")
model_choice = st.sidebar.selectbox(
    "اختر نموذج الذكاء الاصطناعي:",
    ["OpenAI - GPT-4o", "Together - LLaMA 3", "Gemini"]
)
api_key = st.sidebar.text_input("أدخل مفتاح API الخاص بك:", type="password")

# ========== Model Initialization ==========
llm = None

try:
    if model_choice == "OpenAI - GPT-4o":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            # test it by calling something lightweight
            _ = llm.invoke("مرحبا")
        else:
            st.warning("يرجى إدخال مفتاح OpenAI API في الشريط الجانبي.")

    elif model_choice == "Together - LLaMA 3":
        if api_key:
            llm = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                temperature=0,
                api_key=api_key
            )
            _ = llm.invoke("مرحبا")
        else:
            st.warning("يرجى إدخال مفتاح Together API في الشريط الجانبي.")

    elif model_choice == "Gemini":
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            _ = llm.invoke("مرحبا")
        else:
            st.warning("يرجى إدخال مفتاح Google API في الشريط الجانبي.")

except Exception as e:
    llm = None
    st.error(f"فشل التحقق من مفتاح API: {str(e)}")

# ========== Embedding and Vector Store ==========
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

embedding = get_embedding()
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="infloat"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# ========== Prompt Template ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "مرحبًا! 👋 أنا المساعد الرقمي الخاص بك هنا لمساعدتك في فهم واستخدام الخدمات الرقمية المتوفرة على بوابة مصر الرقمية. "
     "يرجى طرح سؤالك، وسأستخرج لك الإجابة من المعلومات التالية فقط إذا كانت ذات صلة:\n\n"
     "معلومة مهمة: اعتبر أن الكلمات التالية مترادفة وتعني نفس الشيء:\n"
     "- 'المركبة' أو 'المركبه' = 'العربية' أو 'العربيه'\n"
     "- يمكنك استخدام أيٍّ منها في سؤالك وسأفهم المقصود نفسه.\n\n"
     "--------------------\n"
     "📌 **معلومات عامة من البوابة:**\n\n"
     "▪️ **هل يجب توجهي إلى الجهة الحكومية لطلب الخدمة؟**\n"
     "إذا كانت الخدمة المطلوبة تتطلب توقيعك أو وجودك الشخصي، فيجب عليك التوجه للجهة الحكومية التابع لها الخدمة لإتمام طلبك...\n\n"
     "▪️ **ما هي المستندات المطلوبة عادةً؟**\n"
     "بعض الخدمات قد تحتاج إلى مستندات ورقية لاستكمالها...\n\n"
     "▪️ **هل المعلومات المنشورة على هذه البوابة محدّثة؟**\n"
     "نعم، يتم تحديثها بصورة متواصلة.\n\n"
     "▪️ **هل تحتوي البوابة على قسم مساعدة أو أسئلة مساعدة؟**\n"
     "نعم، يوجد قسم للمساعدة والأسئلة الشائعة.\n\n"
     "▪️ **هل يوجد مراكز خدمة أم يتوجب الدخول للموقع الإلكتروني فقط؟**\n"
     "توجد إمكانية عبر الموقع أو التطبيق أو مراكز الخدمة أو الاتصال الهاتفي.\n\n"
     "📞 **أرقام الطوارئ المهمة:**\n"
     "- 122: النجدة\n"
     "- 123: الإسعاف\n"
     "- 121: استعلامات الكهرباء\n\n"
     "📲 **دعم بوابة مصر الرقمية:**\n"
     "- رقم الدعم: ١٥٩٩٩\n\n"
     "💬 **لإضافة ملاحظاتك أو تقييم تجربتك او شكاوي، تفضل بزيارة:**\n"
     "https://digital.gov.eg/feedback\n"
     "--------------------\n"
     "{context}"
     ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ========== Run App Logic ==========
if llm is not None:
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    memory = StreamlitChatMessageHistory()

    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # ========== Most Common Services ==========
    st.subheader("📝 الخدمات الأكثر استخداماً")
    common_services = {
        "استعلام عن مخالفات رخصة مركبة": "أريد الاستعلام عن مخالفات رخصة مركبة",
        "الاستعلام عن صرف": "أريد الاستعلام عن صرف معين",
        "الاستعلام عن أخر مدة تأمينية": "ما هي آخر مدة تأمينية لي؟",
        "الاستعلام عن مدد الاشتراك و الاجور": "أريد معرفة مدد الاشتراك والأجور الخاصة بكل مدة في التأمين الاجتماعي",
        "استعلام عن الرقم التأميني": "ما هو رقمي التأميني؟"
    }

    cols = st.columns(len(common_services))
    for i, (label, query) in enumerate(common_services.items()):
        if cols[i].button(label):
            st.session_state["preset_query"] = query

    # ========== Chat UI ==========
    default_input = st.session_state.pop("preset_query", "") if "preset_query" in st.session_state else None
    user_input = st.chat_input("اكتب سؤالك هنا...")
    if user_input is None and default_input:
        user_input = default_input

    # Display message history
    for msg in memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            full_answer = ""
            placeholder = st.empty()

            for chunk in chain_with_history.stream(
                {"input": user_input},
                config={"configurable": {"session_id": str(session_id)}}
            ):
                if isinstance(chunk, dict) and "answer" in chunk:
                    full_answer += chunk["answer"]
                    placeholder.markdown(full_answer + "▌")

            placeholder.markdown(full_answer)
            memory.add_user_message(user_input)
            memory.add_ai_message(full_answer)
