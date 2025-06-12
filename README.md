# Maqasid (مقاصد): A Deep Learning Framework for Nuanced Thematic Classification of Arabic Poetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link.streamlit.app) <!-- 👈 **مهم:** استبدل هذا الرابط برابط تطبيقك الفعلي عند نشره -->

**Maqasid** is an end-to-end framework designed to address the critical challenges in the computational thematic analysis of Arabic poetry. It provides a robust methodology and a suite of tools for researchers, developers, and digital humanists to explore the rich, multifaceted themes inherent in one of the world's oldest literary traditions.

This project moves beyond simple single-label classification by introducing a novel hierarchical thematic taxonomy and a powerful hybrid deep learning model capable of understanding thematic complexity and overlap.

---

## 📜 Project Overview

The computational analysis of Arabic poetry has been hindered by its thematic intricacy and a significant lack of comprehensively annotated corpora. **Maqasid** directly tackles these issues by delivering:

1.  **A Richly Annotated Corpus:** A new, large-scale, multi-label Arabic poetry dataset built upon a novel, literature-grounded hierarchical thematic schema.
2.  **An Optimized Hybrid Model:** A `CNN-BiLSTM` architecture that synergistically captures both local linguistic features and long-range sequential context, powered by custom-trained `FastText` embeddings.
3.  **An Interactive Platform:** An intuitive web application built with Streamlit that allows users to analyze new poems and interactively explore the annotated corpus.

![Maqasid Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)
<!-- 👈 **مهم:** قم بإنشاء GIF قصير يوضح كيفية عمل التطبيق واستبدل هذا الرابط. يمكنك استخدام أدوات مثل ScreenToGif. -->

## ✨ Key Features

- **Multi-Label Classification:** Accurately assigns multiple, co-occurring themes to a single poem, reflecting its true literary nature.
- **Hierarchical Thematic Schema:** Based on seven authoritative works of Arabic literary criticism, our schema captures thematic nuances with up to four levels of specificity.
- **Poetry-Specific Embeddings:** Utilizes a custom `FastText` model trained from scratch on our poetry corpus to understand archaic and metaphorical language.
- **End-to-End MLOps Pipeline:** The entire project is structured with modern MLOps practices, using `DVC` for data versioning and providing a reproducible pipeline.
- **Interactive Dashboard:** A user-friendly web app to both demonstrate the model's capabilities and serve as a powerful tool for scholarly research.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- [Python 3.9](https://www.python.org/downloads/) or higher
- [Java Development Kit (JDK)](https://www.oracle.com/java/technologies/downloads/) (required by `pyfarasa`)
- [Git](https://git-scm.com/downloads/) for cloning the repository.
- [DVC](https://dvc.org/doc/install) for pulling the data and models.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/maqasid.git
cd maqasid
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages for development, testing, and running the app.

```bash
pip install -r requirements.txt
```

### 4. Pull Data and Models with DVC

The large data and model files are tracked with DVC. Pull them from the remote storage.

```bash
dvc pull
```
*(Note: You may need to configure your DVC remote first if you are contributing.)*

### 5. Run the Interactive Web Application

Launch the Streamlit application with the following command:

```bash
streamlit run web_app/app.py
```

Your browser should open a new tab with the **Maqasid** dashboard running locally!

---


## 🤝 Contributing

Contributions are welcome! Whether it's improving the model, enhancing the web app, or expanding the dataset, your help is appreciated. Please feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
