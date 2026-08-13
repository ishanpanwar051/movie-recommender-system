import requests
import os

# Try more sources for TMDB dataset
sources = [
    {
        'movies': 'https://raw.githubusercontent.com/krishnaik06/Movie-Recommendation-System/main/tmdb_5000_movies.csv',
        'credits': 'https://raw.githubusercontent.com/krishnaik06/Movie-Recommendation-System/main/tmdb_5000_credits.csv'
    },
    {
        'movies': 'https://raw.githubusercontent.com/harshtewari98/Movie-Recommendation-System/master/tmdb_5000_movies.csv',
        'credits': 'https://raw.githubusercontent.com/harshtewari98/Movie-Recommendation-System/master/tmdb_5000_credits.csv'
    },
    {
        'movies': 'https://raw.githubusercontent.com/codebasics/movie-recommender-system/master/tmdb_5000_movies.csv',
        'credits': 'https://raw.githubusercontent.com/codebasics/movie-recommender-system/master/tmdb_5000_credits.csv'
    },
    {
        'movies': 'https://raw.githubusercontent.com/ashishjain1547/PublicDatasets/master/movies.csv',
        'credits': 'https://raw.githubusercontent.com/ashishjain1547/PublicDatasets/master/credits.csv'
    },
    {
        'movies': 'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/tmdb_5000_movies.csv',
        'credits': 'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/tmdb_5000_credits.csv'
    }
]

for i, source in enumerate(sources):
    print(f"Trying source {i+1}...")
    try:
        print("Downloading movies dataset...")
        response = requests.get(source['movies'])
        if response.status_code == 200 and len(response.content) > 1000:
            with open('tmdb_5000_movies.csv', 'wb') as f:
                f.write(response.content)
            print(f"Movies dataset downloaded successfully! Size: {len(response.content)} bytes")
            
            print("Downloading credits dataset...")
            response = requests.get(source['credits'])
            if response.status_code == 200 and len(response.content) > 1000:
                with open('tmdb_5000_credits.csv', 'wb') as f:
                    f.write(response.content)
                print(f"Credits dataset downloaded successfully! Size: {len(response.content)} bytes")
                print("Download completed successfully!")
                break
            else:
                print(f"Credits download failed with status: {response.status_code}")
        else:
            print(f"Movies download failed with status: {response.status_code}, size: {len(response.content)}")
    except Exception as e:
        print(f"Error with source {i+1}: {e}")
else:
    print("All sources failed. Trying Kaggle API...")
    
    # Try using kagglehub to download
    try:
        import kagglehub
        print("Downloading from Kaggle using kagglehub...")
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
        print(f"Dataset downloaded to: {path}")
        
        # Find and copy the CSV files
        import shutil
        import glob
        
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        for csv_file in csv_files:
            if 'movies' in csv_file.lower():
                shutil.copy(csv_file, 'tmdb_5000_movies.csv')
                print(f"Copied {csv_file} to tmdb_5000_movies.csv")
            elif 'credits' in csv_file.lower():
                shutil.copy(csv_file, 'tmdb_5000_credits.csv')
                print(f"Copied {csv_file} to tmdb_5000_credits.csv")
        
        if os.path.exists('tmdb_5000_movies.csv') and os.path.exists('tmdb_5000_credits.csv'):
            print("Dataset downloaded successfully from Kaggle!")
        else:
            print("Could not find required files in Kaggle download")
    except ImportError:
        print("kagglehub not installed. Installing...")
        os.system("pip install kagglehub")
        print("Please run this script again after installation")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
