# NeuralArticleFinder

NeuralArticleFinder is a project aimed at preprocessing articles, training neural models, and recommending articles based on content similarity.

## Technologies Used

- **Python**: The core programming language used for this project.
- **NLTK**: Used for text preprocessing.
- **Gensim**: Used for training the Word2Vec model and article recommendation.
- **NumPy**: Used for numerical operations.

---

## Getting Started

### 1. Clone the Repository

First, you need to clone the repository to your local machine. You can do this using:

```bash
git clone https://github.com/renan-siqueira/NeuralArticleFinder.git
```

Access the project root folder:
```bash
cd NeuralArticleFinder
```

---

### 2. Setting Up a Virtual Environment

It's a good idea to create a virtual environment to manage dependencies. Here's how you can do that:

```bash
python -m venv env
```

Activate the virtual environment:

- Windows:

```bash
env\Scripts\activate
```

- macOS and Linux:

```bash
source env/bin/activate
```

---

### 3. Install Dependencies

Once you have your virtual environment set up and running, you can install the dependencies:

```bash
pip install -r requirements.txt
```

---

### 4. Run the Project

Now, you can run the specific modules, for example:

```bash
python -m src.preprocessing.preprocess
```

---

## License

This project is open-sourced and available to everyone under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you find any bugs or have suggestions for improvements.
