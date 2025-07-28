# 🔍 Fake News Detector

A beautiful and interactive web application built with Streamlit that uses 4 different machine learning models to detect fake news articles.

## 🚀 Features

- **4 ML Models**: Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest
- **Real-time Analysis**: Instant prediction with confidence scores
- **Beautiful UI**: Modern, responsive design with interactive visualizations
- **Confidence Scores**: Each model provides confidence percentage
- **Visual Analytics**: Bar charts and pie charts for easy interpretation
- **Overall Prediction**: Aggregated result from all models
- **Fast Loading**: Optimized for quick startup and predictions

## 📊 Model Performance

Based on the training data:
- **Logistic Regression**: 97.0% accuracy
- **Decision Tree**: 89.2% accuracy  
- **Gradient Boosting**: 92.3% accuracy
- **Random Forest**: 94.0% accuracy
- **Average Accuracy**: 93.1%

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have these files in your directory:
   - app.py
   - requirements.txt
   - train_models.py
   - Fake.csv
   - True.csv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pre-train models (FAST SETUP - Recommended)**
   ```bash
   python train_models.py
   ```
   This will train and save models to disk, making the app load instantly.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

### ⚡ Performance Optimizations

- **Model Persistence**: Models are saved to disk after first training
- **Reduced Data Size**: Uses 8,000 samples instead of full dataset for faster training
- **Optimized Parameters**: Model parameters tuned for speed vs accuracy balance
- **Caching**: Data preprocessing is cached for faster subsequent runs

## 📝 Usage

1. **Enter News Text**: Paste any news article text into the text area
2. **Click Analyze**: Press the "🔍 Analyze News" button
3. **View Results**: 
   - Overall prediction (Fake/True)
   - Individual model predictions with confidence scores
   - Interactive visualizations
   - Model agreement statistics

## 🎨 Features Explained

### Overall Prediction
- Aggregates results from all 4 models
- Shows majority vote (how many models predicted fake vs true)
- Displays average confidence score

### Individual Model Results
- Each model shows its prediction (Fake/True)
- Confidence score as a percentage
- Visual confidence bar
- Color-coded results (Red for Fake, Green for True)

### Visualizations
- **Bar Chart**: Compares confidence scores across all models
- **Pie Chart**: Shows the distribution of model predictions

## 📁 File Structure

```
FakeNews/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── train_models.py     # Model training script
├── README.md          # This file
├── models.pkl         # Pre-trained models (created by train_models.py)
├── Fake.csv           # Fake news dataset
└── True.csv           # True news dataset
```

## 🔧 Technical Details

### Data Processing
- Text preprocessing: lowercase, remove URLs, punctuation, numbers
- TF-IDF vectorization for feature extraction
- Train/test split (75%/25%)

### Models Used
1. **Logistic Regression**: Linear classification model
2. **Decision Tree**: Tree-based classification
3. **Gradient Boosting**: Ensemble method with boosting
4. **Random Forest**: Ensemble method with bagging

### Performance
- Models are cached using `@st.cache_resource` for faster loading
- Real-time prediction without retraining
- Responsive design for different screen sizes

## 🎯 Example Usage

Try these sample texts:

**Fake News Example:**
```
"Scientists discover that drinking coffee makes you immortal! 
A new study shows that coffee drinkers live forever and never age. 
This revolutionary finding will change medicine forever."
```

**True News Example:**
```
"WASHINGTON (Reuters) - The Federal Reserve raised interest rates 
by 0.25 percentage points on Wednesday, citing strong economic 
growth and low unemployment as reasons for the increase."
```

## 🐛 Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   - Make sure you've installed all requirements: `pip install -r requirements.txt`

2. **Data loading errors**
   - Ensure `Fake.csv` and `True.csv` are in the same directory as `app.py`

3. **Slow loading**
   - First run takes longer as models need to train
   - Subsequent runs will be faster due to caching

4. **Port already in use**
   - Try: `streamlit run app.py --server.port 8502`

## 📈 Future Enhancements

- Add more ML models (BERT, transformers)
- Include model explanation features
- Add batch processing for multiple articles
- Export results functionality
- User feedback collection

## 🤝 Contributing

Feel free to contribute by:
- Adding new models
- Improving the UI/UX
- Enhancing the preprocessing pipeline
- Adding more visualizations

## 📄 License

This project is open source and available under the MIT License.

---

**🔍 Fake News Detector | Powered by Machine Learning** 