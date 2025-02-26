# Next Word Prediction - Flask Web Application

This is a Flask web application that uses NLP to predict the next word(s) in a sentence. The application uses pre-trained language models from Hugging Face's Transformers library.

## Features

- Predict the next word(s) for any input text
- Adjust parameters like number of words to predict, number of predictions, and temperature
- Interactive web interface
- RESTful API for integration with other applications

## Requirements

- Python 3.8+
- Flask
- PyTorch
- Transformers
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
next-word-prediction/
├── app.py                 # Flask application
├── next_word_predictor.py # Prediction logic
├── requirements.txt       # Dependencies
├── Dockerfile             # For containerization
├── templates/
│   └── index.html         # Web interface
└── README.md              # This file
```

## API Usage

The application provides a REST API endpoint for predictions.

### Endpoint: `/predict`

**Method**: POST

**Request Body**:
```json
{
  "text": "The weather today is",
  "num_words": 1,
  "num_predictions": 5,
  "temperature": 1.0
}
```

**Response**:
```json
{
  "original_text": "The weather today is",
  "predictions": [
    {
      "text": "nice",
      "confidence": 0.0854
    },
    {
      "text": "good",
      "confidence": 0.0723
    },
    ...
  ],
  "processing_time": 0.453
}
```

## Docker Support

Build the Docker image:
```
docker build -t next-word-prediction .
```

Run the container:
```
docker run -p 5000:5000 next-word-prediction
```

## Customization

### Changing the Model

You can use different models by modifying the model initialization in `app.py`:

```python
# For a smaller, faster model
predictor = NextWordPredictor(model_name="distilgpt2")

# For a more accurate model
predictor = NextWordPredictor(model_name="gpt2-medium")
```

### Performance Considerations

- The first prediction might take longer as the model is loaded into memory
- Using a GPU will significantly improve prediction speed
- For production use, consider using a smaller model or model quantization

## License

MIT
