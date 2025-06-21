<div align="center">
  <h1 align="center">
    📜 Maqasid (مقاصد) 📜
  </h1>
  <p align="center">
    <strong>A Deep Learning Framework for Nuanced Thematic Classification of Arabic Poetry</strong>
  </p>
  <p align="center">
    <a href="https://your-streamlit-app-link.streamlit.app">
      <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version">
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
  </p>
</div>

**Maqasid** is an end-to-end research framework designed to address the critical challenges in the computational thematic analysis of Arabic poetry. It provides a robust methodology and a suite of tools for researchers, developers, and digital humanists to explore the rich, multifaceted themes inherent in one of the world's oldest literary traditions.

This project moves beyond simple single-label classification by introducing a novel hierarchical thematic taxonomy and a powerful hybrid deep learning model capable of understanding thematic complexity and overlap.

[**➡️ Live Demo**](https://colab.research.google.com/drive/1Z6r-Q37jYzyjHcR-nAtRXRfr9BTAWLYG?usp=drive_link)
<br>
<p align="center">
  <img src="https://your-link-to-a-demo-gif.com/demo.gif" alt="Maqasid Demo" width="80%">
</p>

---

## 📖 Table of Contents

- [✨ Key Features](#-key-features)
- [📊 The Maqasid Corpus](#-the-maqasid-corpus)
- [🔬 Interactive Exploration with Google Colab](#-interactive-exploration-with-google-colab)
- [🛠️ Technology Stack](#-technology-stack)
- [🚀 Getting Started](#-getting-started)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [🎓 Citing This Work](#-citing-this-work)
- [📄 License](#-license)

---

## ✨ Key Features

- **Multi-Label Classification:** Accurately assigns multiple, co-occurring themes to a single poem, reflecting its true literary nature.
- **Hierarchical Thematic Schema:** A novel taxonomy based on seven authoritative works of Arabic literary criticism, capturing thematic nuances with up to four levels of specificity.
- **Poetry-Specific Embeddings:** Utilizes a custom `FastText` model trained from scratch on our poetry corpus to understand archaic and metaphorical language.
- **Reproducible MLOps Pipeline:** The entire project is structured with modern MLOps practices, using `DVC` for data versioning to ensure full reproducibility.
- **Interactive Tools:** A user-friendly web application and a Google Colab notebook for model demonstration and in-depth corpus exploration.

---

## 📊 The Maqasid Corpus

At the heart of this project is the **Maqasid Corpus**, a new, large-scale, and richly annotated dataset of Arabic poetry. It was constructed to overcome the limitations of existing resources and to serve as a foundational tool for computational literary analysis.

- **Size:** Contains [الرجاء إدخال عدد القصائد، مثلاً +20,000] poems from various historical eras.
- **Annotation:** Each poem is annotated with multiple thematic labels from our hierarchical schema, capturing thematic co-occurrence.
- **Grounding:** The thematic taxonomy was developed through a rigorous synthesis of classical and modern Arabic literary criticism, ensuring scholarly validity.
- **Accessibility:** The corpus is made available to the research community to foster new, data-driven inquiries into Arabic literature.

---

## 🔬 Interactive Exploration with Google Colab

To enhance the accessibility and promote hands-on analysis of the **Maqasid Corpus**, we have developed an interactive Google Colab notebook. This tool empowers anyone—from students to seasoned researchers—to visually explore, filter, and analyze the dataset directly in their browser with zero setup.

➡️ **[Open the Interactive Explorer in Google Colab](https://colab.research.google.com/drive/1Z6r-Q37jYzyjHcR-nAtRXRfr9BTAWLYG?usp=drive_link)**

The notebook features two powerful, user-friendly dashboards:

#### 1. Thematic Poem Browser

This dashboard provides an intuitive way to navigate the corpus through its rich thematic hierarchy. It allows you to:

-   **Drill-Down Through Themes**: Start from broad categories (e.g., "Love Poetry") and progressively narrow your focus to highly specific sub-themes (e.g., "Chaste Love" → "Love from a distance").
-   **Instantly Access Poems**: As you select a theme, the interface immediately populates a list of all poems annotated with that specific theme.
-   **View Detailed Poem Analysis**: Clicking on a poem reveals its full text, essential metadata (poet, era), and an interactive pie chart that visualizes its complete thematic composition.

<p align="center">
  <em>The interactive poem browser in action, allowing users to filter poems by theme and view a detailed analysis with a dynamic chart.</em>
  <br>
  <img src="https://raw.githubusercontent.com/NoorBayan/Maqasid/main/images/ThematicPoem.png" width="500px"/>
</p>

#### 2. Cross-Era Thematic Analysis Dashboard

Designed for comparative literary studies, this advanced analytical tool enables data-driven investigation into the evolution of poetic themes across different historical periods. Its key functionalities include:

-   **Targeted Analysis**: Select a primary theme (e.g., "Praise Poetry") and a specific historical era (e.g., "Umayyad Period") to focus your inquiry.
-   **Dynamic Visualization**: The tool automatically generates a series of hierarchical bar charts that break down the chosen theme into its sub-themes, displaying the frequency of each within the selected era.
-   **Uncover Literary Trends**: This dashboard facilitates empirical answers to complex research questions, such as: *"Which sub-themes of Satire were most prevalent in the Abbasid era compared to the Modern era?"*

<p align="center">
  <em>The Cross-Era Analysis Dashboard generating hierarchical bar charts to compare sub-theme frequencies within a selected era.</em>
  <br>
  <img src="https://raw.githubusercontent.com/NoorBayan/Maqasid/main/images/ThematicAnalysis.png" width="400px"/>
</p>

This powerful feature transforms the Maqasid Corpus from a static dataset into a dynamic laboratory for literary and historical inquiry.

---

## 🛠️ Technology Stack

- **Backend & ML:** Python, PyTorch, Gensim, Scikit-learn
- **Web Framework & Notebooks:** Streamlit, Google Colab, Plotly
- **Data Versioning:** DVC
- **HPO:** Optuna
- **Code Quality:** Black, isort, Flake8
- **Testing:** Pytest

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Java Development Kit (JDK)](https://www.oracle.com/java/technologies/downloads/) (required by `pyfarasa`)
- [Git](https://git-scm.com/downloads/) & [DVC](https://dvc.org/doc/install)

### Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/maqasid.git
    cd maqasid
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull Data and Models with DVC**
    *This step downloads the large data and model files tracked by DVC.*
    ```bash
    dvc pull
    ```

5.  **Run the Interactive Web Application**
    ```bash
    streamlit run web_app/app.py
    ```
    Your browser should open a new tab with the **Maqasid** dashboard!

---

## 📂 Project Structure

The project follows a modular and scalable structure to separate concerns:

```
├── data/              # Raw, processed, and annotated datasets (tracked by DVC)
├── saved_models/      # Trained models and embeddings (tracked by DVC)
├── src/               # Core Python source code for the classification pipeline
├── web_app/           # Source code for the Streamlit interactive application
├── scripts/           # Executable scripts for running pipeline stages
├── tests/             # Unit tests for the project
├── .dvc/              # DVC metadata
├── dvc.yaml           # DVC pipeline definition
└── requirements.txt   # Project dependencies
```

---

## 🤝 Contributing

Contributions are welcome! Whether it's improving the model, enhancing the web app, or expanding the dataset, your help is appreciated. Please check our [contribution guidelines](CONTRIBUTING.md) (you can create this file later) or open an issue to get started.

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some Amazing Feature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
