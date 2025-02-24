# Content-Based Movie Recommendation System

## Overview
This project implements a simple content-based recommendation system for movies. It uses TF-IDF vectorization and cosine similarity to recommend movies that match a user’s textual description of their preferences. The system uses both movie overviews and keywords to compute similarity.

## Dataset
- **Source:** Kaggle IMDB 500 Movie Datasets.
- **Details:** For simplicity and speed, I am using the top 700 movies from the dataset.
- **Loading:**  
  Place the `movie_dataset.csv` file containing columns such as `overview`, `keywords`, and `original_title` or `title` in the repository’s root directory. The code will automatically load and preprocess the dataset.

## Setup
- **Python Version:**  
  This project requires Python 3.7 or higher.
  
- **Virtual Environment (Recommended):**  
  To create and activate a virtual environment, run:
  ```bash
  python -m venv venv
  # On macOS/Linux:
  source venv/bin/activate
  # On Windows:
  venv\Scripts\activate

- **Install Dependencies:**
Install the required packages with:
pip install -r requirements.txt

The dependencies include:
pandas
scikit-learn

## Running
- **Running the Code:**
To run the recommendation system, execute:
python recommend.py

You will be prompted to enter a description of your movie preferences. The system will then output the top 5 matching movies along with their similarity scores and overviews.
For example:
```bash
Enter a description of your movie preferences:
>> I like romantic comedy
```

## Results
- **Sample Output:**
Here’s an example of what the output might look like for the sample query above:
```bash
Recommended Movies:

1. 16 to Life
   Similarity: 0.2261
   Overview: romantic comedy  a small town teenager s angst about sexual inexperience drives a comic quest for love and understanding on a birthday to end all birthdays

2. Christmas Mail
   Similarity: 0.2218
   Overview: in this holiday romantic comedy  a mysterious woman who works at the post office answering santa s mail captures the heart of a disillusioned postal carrier

3. Midnight in Paris
   Similarity: 0.2183
   Overview: a romantic comedy about a family traveling to the french capital for business  the party includes a young engaged couple forced to confront the illusion that a life different from their own is better

4. About Last Night
   Similarity: 0.2100
   Overview: a modern reimagining of the classic romantic comedy  this contemporary version closely follows new love for two couples as they journey from the bar to the bedroom and are eventually put to the test in the real world

5. One Day
   Similarity: 0.2028
   Overview: a romantic comedy centered on dexter and emma  who first meet during their graduation in 1988 and proceed to keep in touch regularly  the film follows what they do on july 15 annually  usually doing something together
```
- **The link to watch the demo for the content based recommendation system is in demo.md file.**
