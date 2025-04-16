import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Streamlit UI
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar for API key input
with st.sidebar:
    hf_api_key = st.text_input("🔐 Hugging Face API Token", value="", type="password")

# Input URL (YouTube or website)
generic_url = st.text_input("Enter a YouTube or Website URL", label_visibility="visible")

# When the button is clicked
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("❗ Please provide both the API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("❗ The URL is invalid. Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing content..."):
                # Load content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # Initialize LLM only after key is provided
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                    task="text-generation",
                    hf_api_key=hf_api_key,
                    temperature=0.7,
                    model_kwargs={"max_length": 150}
                )

                # Summarization prompt
                prompt_template = """
                Provide a summary of the following content in 300 words:
                Content: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display result
                st.success("✅ Summary generated successfully!")
                st.markdown(f"### 📄 Summary:\n\n{output_summary}")
        except Exception as e:
            st.error("⚠️ Something went wrong during summarization.")
            st.exception(e)
