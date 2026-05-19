import os
import shutil
import kagglehub

def main():
    competition_name = "rogii-wellbore-geology-prediction"
    
    # Ensure username environment variable is set for kagglehub
    if "KAGGLE_USERNAME" not in os.environ:
        os.environ["KAGGLE_USERNAME"] = "nustanishant"
    
    print(f"Downloading data for competition: {competition_name}...")
    try:
        # Download the competition data
        cache_path = kagglehub.competition_download(competition_name)
        print(f"Downloaded successfully to cache path: {cache_path}")
        
        # Move or copy the files to our local data directory
        dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"Copying files from {cache_path} to {dest_dir}...")
        for item in os.listdir(cache_path):
            s = os.path.join(cache_path, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Data files copied successfully to local data folder!")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please check if you have accepted the competition rules on Kaggle: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules")

if __name__ == "__main__":
    main()
