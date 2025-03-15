!/bin/bash

pip install gdown

FILENAMES_URL="https://drive.google.com/uc?id=1bWoBSU5Hq1wNQUYuP_BS565P4C3Rr7Bp"  
IMAGE_FEATURES_URL="https://drive.google.com/uc?id=1iq7xKxz_LUZDIT0wCix0fIr4KmrjgjK1" 

gdown "$FILENAMES_URL" -O filenames.pkl
gdown "$IMAGE_FEATURES_URL" -O image_features.pkl