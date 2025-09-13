# Fashion Recommendation Web Application

## Introduction
This project is a smart AI-driven web application that helps users discover clothing items visually similar to an input image. Users can upload images from online, social media, or stores, and receive curated fashion recommendations powered by deep learning models.

## Main Goals
- Deploy advanced AI models: ResNet50 & VAE
- Similarity-based recommendation system
- User-centric web application using Flask (with RTL support)
- Flexible & scalable design
- Real-world use cases in e-commerce or digital wardrobes

## Dataset
- Fashion Product Images Dataset by paramaggarwal on Kaggle
- 44,000 images of tops, pants, dresses, shoes, bags, etc.
- Only images used for visual similarity
- Preprocessed & subset of 500 images used for feature extraction

## Models
### ResNet50
- Feature extraction: 2048D vectors
- Pre-trained on ImageNet
- Cosine similarity for ranking

### Variational Autoencoder (VAE)
- Latent space: 64D vectors
- Trained encoder weights saved (`vae_encoder.h5`)
- Cosine similarity in latent space

## System Architecture
1. User uploads image
2. Preprocessing (ResNet: 224×224, VAE: 64×64)
3. Feature extraction
4. Similarity search in dataset
5. Top matches returned

## Evaluation Metrics
- Cosine similarity score [-1,1]
- Top-N accuracy (Precision@N)
- Mean Average Precision (MAP)
- F1 Score (adapted for retrieval)

## Challenges & Solutions
- VAE training instability → saved encoder weights
- Evaluating quality without ground truth → visual validation
- Integrating two models → dropdown selector in frontend

## Conclusion
A functional AI-powered fashion recommendation web application integrating ResNet50 and VAE for flexible and intelligent recommendations.

## References
1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114)
3. [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
4. [TensorFlow](https://www.tensorflow.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/)

