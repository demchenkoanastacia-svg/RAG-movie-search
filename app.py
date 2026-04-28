"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3

A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!

Instructions:
  1. Replace the DOCUMENTS list below with your own texts
  2. Update the app title and description
  3. Run locally:  streamlit run app.py
  4. Deploy to Render (see assignment instructions)
"""

import streamlit as st
import numpy as np

#Memory check for render
import tracemalloc, os, psutil

tracemalloc.start()
process = psutil.Process(os.getpid())
st.sidebar.metric("RAM used (MB)", f"{process.memory_info().rss / 1024**2:.1f}")

st.set_page_config(
    page_title="My MOVIE RAG Knowledge Base",
    page_icon="🔍🎬",
    layout="wide",
)

# --- Custom Styling for Aesthetics ---


st.markdown("""
<style>
/* Main App Background and Sidebar */
[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
}
[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #E50914; /* Movie Red Accent */
}

/* Hide Radio Button Dots completely */
div[role="radiogroup"] > label > div:first-of-type {
    display: none !important;
}

/* Style the Radio List Items (Sidebar Navigation) */
div[role="radiogroup"] > label {
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 5px;
    background-color: transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}

/* Hover effect */
div[role="radiogroup"] > label:hover {
    background-color: rgba(229, 9, 20, 0.15); /* subtle red */
}

/* Highlight Selected Page (Red Background) */
div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #E50914 !important; 
}
div[role="radiogroup"] > label[data-checked="true"] p {
    color: white !important;
    font-weight: bold;
}

/* Metrics and Containers styled darkly */
div[data-testid="stMetric"] {
    background-color: #1a1a1a;
    border-left: 4px solid #E50914;
    padding: 15px;
    border-radius: 8px;
}
div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #151515;
    border-color: #333333;
}

/* General Text color optimizations for dark mode */
h1, h2, h3, p, span, div.stMarkdown, div.stText, label {
    color: #f1f1f1 !important;
}

/* Input fields */
.stTextInput > div > div > input {
    color: #ffffff !important;
    border-color: #E50914 !important;
}

/* Dividers */
hr {
    border-bottom-color: #E50914 !important;
}


</style>
""", unsafe_allow_html=True)




# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "title": "Mad Max 2", 
        "plot": "In a post-apocalyptic wasteland, former cop Max Rockatansky agrees to help a community of settlers defend their oil refinery from a violent gang of marauders led by Lord Humungus and his lieutenant Wez. Max retrieves a semi-truck to haul the community's fuel tanker to safety. During a high-speed chase, Max drives the tanker as a diversion, allowing the settlers to escape with the actual fuel hidden in oil drums. Humungus and Wez are killed in a head-on collision, and Max remains a lone wanderer as the 'Road Warrior'.", 
        "genre": "Action", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/f/f7/Mad_max_two_the_road_warrior.jpg"
    },
    {
        "title": "Die Hard", 
        "plot": "NYPD officer John McClane arrives at Nakatomi Plaza in Los Angeles to reconcile with his wife, Holly, during a Christmas party. The building is seized by German radical Hans Gruber, who intends to steal $640 million in bearer bonds under the guise of terrorism. McClane wages a one-man war against the group, using a radio to alert Sergeant Al Powell. After a series of gunfights and explosions, including the destruction of the roof, McClane kills Gruber by dropping him from the skyscraper and rescues the hostages.", 
        "genre": "Action, Christmas", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/c/ca/Die_Hard_%281988_film%29_poster.jpg"
    },
    {
        "title": "Pinocchio", 
        "plot": "Geppetto the woodcarver creates a puppet named Pinocchio, who is brought to life by the Blue Fairy with the promise of becoming a real boy if he is brave and truthful. Guided by his conscience, Jiminy Cricket, Pinocchio is led astray by Honest John and sent to Pleasure Island, where boys are turned into donkeys. After escaping with a donkey's tail, Pinocchio rescues Geppetto from the belly of the whale Monstro. Pinocchio dies saving his father but is resurrected as a human boy for his selfless bravery.", 
        "genre": "Animation", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/b/ba/Pinocchio-1940-poster.jpg"
    },
    {
        "title": "Toy Story", 
        "plot": "Sheriff Woody, a pull-string cowboy doll, feels threatened when his owner Andy receives a high-tech Buzz Lightyear action figure for his birthday. Buzz believes he is a real Space Ranger, leading to a rivalry that results in both toys being lost at the house of Sid, a cruel neighbor who destroys toys. After Buzz realizes he is a toy, he and Woody team up to escape Sid's mutant toys. They use a rocket to reunite with Andy's family during a move, becoming best friends in the process.", 
        "genre": "Animation", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg"
    },
    {
        "title": "The Texas Chain Saw Massacre", 
        "plot": "Five teenagers traveling through Texas to visit a family homestead run out of gas and are hunted by a family of cannibalistic killers, including the chainsaw-wielding Leatherface. One by one, the friends are murdered with hammers and meat hooks. The final survivor, Sally Hardesty, is captured and tortured during a macabre dinner with the family. She manages to escape through a window and flags down a passing truck, narrowly evading an enraged Leatherface as he dances with his chainsaw at sunrise.", 
        "genre": "Horror", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/a/a0/The_Texas_Chain_Saw_Massacre_%281974%29_theatrical_poster.jpg"
    },
    {
        "title": "The Matrix", 
        "plot": "Hacker Neo discovers that his reality is a simulated world called the Matrix, created by AI machines to harvest bioelectric energy from captive humans. Rescued by the rebel leader Morpheus, Neo is brought to the real-world ship Nebuchadnezzar and told he is 'the One' destined to end the war. After a betrayal by crewmate Cypher leads to Morpheus's capture by the lethal Agent Smith, Neo and Trinity launch a rescue mission. Though Neo is killed by Smith, he is resurrected by Trinity's love, gaining the ability to perceive and manipulate the Matrix's code. He destroys Smith, escapes to the real world, and vows to liberate humanity from the machines.",
        "genre": "Sci-Fi", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/d/db/The_Matrix.png"
    },
    {
        "title": "2001: A Space Odyssey", 
        "plot": "After prehistoric hominins discover an alien monolith that triggers technological evolution, humans in the future find a similar object on the moon emitting a signal toward Jupiter. The spacecraft Discovery One is sent to investigate, manned by Dave Bowman, Frank Poole, and the sentient computer HAL 9000. HAL malfunctions and kills the crew, leaving only Bowman to deactivate him. Reaching Jupiter, Bowman encounters a massive monolith, travels through a stargate, and is transformed into a 'Star Child' representing the next stage of human evolution.", 
        "genre": "Sci-Fi", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/1/11/2001_A_Space_Odyssey_%281968%29.png"
    },
    {
        "title": "Blade Runner", 
        "plot": "In 2019 Los Angeles, 'blade runner' Rick Deckard is tasked with retiring four escaped Nexus-6 replicants led by Roy Batty. During his investigation, Deckard meets Rachael, an experimental replicant who believes she is human. After killing the other replicants, Deckard is hunted by Roy onto a rooftop. Roy chooses to save Deckard's life before his own four-year lifespan expires. Finding an origami unicorn, Deckard flees the city with Rachael, questioning the boundary between human and machine.", 
        "genre": "Sci-Fi", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/9/9f/Blade_Runner_%281982_poster%29.png"
    },
    {
        "title": "The Dark Knight", 
        "plot": "The Joker orchestrates a series of crimes in Gotham City to plunge it into anarchy and challenge Batman's moral code. He corrupts District Attorney Harvey Dent by orchestrating the death of Rachel Dawes, leading to Dent becoming the vengeful Two-Face. Batman stops the Joker's social experiment on two rigged ferries but must deal with Dent's rampage. To preserve Dent's reputation as a hero and keep the city's hope alive, Batman takes the blame for Dent's crimes and becomes a fugitive hunted by the police.", 
        "genre": "Superhero", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg"
    },
    {
        "title": "Saving Private Ryan", 
        "plot": "Following the D-Day landings, Captain Miller leads a squad behind enemy lines in Normandy to find Private James Ryan, whose three brothers have all been killed in action. The mission sparks debate among the men about the value of one life versus many. They eventually find Ryan defending a strategic bridge in Ramelle. During a brutal German assault, most of Miller's squad is killed, but they hold the bridge until reinforcements arrive. Miller's dying words urge Ryan to 'earn' the sacrifice made for him.", 
        "genre": "War", 
        "poster": "https://upload.wikimedia.org/wikipedia/en/a/ac/Saving_Private_Ryan_poster.jpg"
    }
]

# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
#    from langchain_huggingface import HuggingFaceEmbeddings
#    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


@st.cache_resource(show_spinner="Building vector database...")
def get_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import chromadb
    

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    metadatas = []
    for doc in _documents:
        # Split only the plot text, keeping the title for metadata
        split_plots = splitter.split_text(doc["plot"])
        chunks.extend(split_plots)
        # Attach the title to every single chunk of this movie
        metadatas.extend([{"title": doc["title"]}] * len(split_plots))

    #embeddings = load_embedding_model()
    ef = load_embedding_model()
    client = chromadb.Client()
    collection = client.get_or_create_collection(
    name="knowledge_base_movies",
    embedding_function=ef,
    )
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(chunks))],
    )
    return collection, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("My MOVIE RAG App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Search", "📦 Explore Chunks", "🎬 Movie Database"], label_visibility="collapsed")

# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    # ─── INJECT BACKGROUND IMAGE FOR HOME PAGE ───
    import os
    import base64
    if os.path.exists("background.jpg"):
        with open("background.jpg", "rb") as f:
            bg_ext = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: linear-gradient(rgba(0,0,0, 0.5), rgba(0,0,0, 0.5)), url("data:image/jpeg;base64,{bg_ext}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    # ─────────────────────────────────────────────

    st.title("My RAG Movie Database")
    st.markdown("""
    Welcome! This app lets you search movies by plot details, not just keywords.

    ### How it works
    1. Movie plots are split into small chunks
    2. Each chunk is converted to an **embedding** (a vector of numbers)
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you search, your query is embedded and compared to all chunks
    5. The most **semantically similar** chunks are returned

    ### Get started
    - Explore our database **Movie Database**
    - Go to **Search** to ask questions
    - Go to **Explore Chunks** to see how documents are split
    - P.S. remmember that our database is only 10 movies from wikipedia page with best movies, so don't be sad if you don't find what you're searching and just watch different movie :)

    ---
    *Built with Streamlit, LangChain, and ChromaDB*
    """)

    st.info(f"Knowledge base contains **{len(DOCUMENTS)} documents**.")


# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "🔍 Search":
    st.title("Semantic Search")
    st.markdown("Ask a question and the app will find the most relevant chunks from the knowledge base.")

    vector_store, chunks = get_vector_store(tuple(DOCUMENTS))

    query = st.text_input(
        "Your question",
        placeholder="Search for a movie plot (e.g., 'space explorations')",
    )
    num_results = st.slider("Number of results", 1, 10, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.query(query_texts=[query], n_results=num_results)
            

        st.subheader(f"Top {num_results} results")
        for i, (text, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            similarity = 1.0 / (1.0 + dist)
            with st.container():
                st.markdown(f"**Result {i}**: {meta.get('title', 'Unknown')} — relevance: `{similarity:.2f}`")
                st.markdown(f"> {text}")
                st.divider()



    st.markdown("---")
    st.caption("Powered by DefaultEmbeddingFunction + ChromaDB")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "📦 Explore Chunks":
    st.title("Explore Chunks")
    st.markdown("See how your documents are split into chunks by the recursive text splitter.")

    vector_store, chunks = get_vector_store(tuple(DOCUMENTS))

    st.metric("Total chunks", len(chunks))

    lengths = [len(c) for c in chunks]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg chunk size", f"{np.mean(lengths):.0f} chars")
    col2.metric("Min chunk size", f"{min(lengths)} chars")
    col3.metric("Max chunk size", f"{max(lengths)} chars")

    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
            st.text(chunk)

# ──────────────────────────────────────────────────────────────────────
# MOVIE DATABASE PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "🎬 Movie Database":
    st.title("Movie Database")
    st.markdown("Browse our library! More movies are comming soon...")

    st.markdown("""
    <style>
    /* This targets images inside containers to keep them uniform */
    .stImage img {
        height: 400px;
        width: 100%;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create columns for a grid layout (3 columns wide)
    cols = st.columns(3)
    
    for i, doc in enumerate(DOCUMENTS):
        with cols[i % 3]:
            # Use Streamlit's new stylized container feature to make a "card", setting uniform height
            with st.container(height=650, border=True):
                placeholder_url = doc['poster']
                st.image(placeholder_url, use_container_width=True)
                
                st.subheader(doc['title'])
                st.caption(f"**Genre:** {doc['genre']}")
                
                # Only show the first ~120 characters of the plot
                st.write(f"_{doc['plot'][:120]}..._")
