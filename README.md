# 🎬 CineMatch: AI-Powered Movie Recommender

CineMatch is a premium movie recommendation system that uses machine learning to suggest films based on your preferences. Built with Streamlit and the TMDB dataset, it features a modern glassmorphism UI and robust error handling.

## ✨ Features

- **AI-Powered Recommendations**: Uses CountVectorizer and Cosine Similarity to find movies with similar tags.
- **Premium UI**: Modern dark theme with glassmorphism effects and smooth animations.
- **Dynamic Content**: Fetches real-time posters, ratings, and release years from the TMDB API.
- **AI-Powered Recommendations**: Uses CountVectorizer and Cosine Similarity to find the best matches.
- **Premium UI**: Modern dark theme with Glassmorphism and smooth animations.
- **Real-time Data**: Fetches posters and ratings directly from TMDB API.
- **Fast Search**: Optimized recommendation logic with ThreadPoolExecutor for concurrent API calls.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd movie-recommender-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add your TMDB API Key:
```env
TMDB_API_KEY=your_api_key_here
```

### 4. Generate the Model
If you don't have the pickle files, run the generation script:
```bash
python generate_model.py
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (CountVectorizer, Cosine Similarity)
- **Data Manipulation**: Pandas, Numpy
- **API**: TMDB API

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `generate_model.py`: Script to process data and generate recommendation models.
- `download_data.py`: Script to fetch the TMDB dataset.
- `model/`: Directory containing generated pickle files.
- `.env`: Configuration for API keys.
- `requirements.txt`: Project dependencies.

---
Developed with ❤️ by Ishan Panwar
