from transformers import pipeline

company_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

sentiment_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def adjust_output(sentiment, sentiment_score):
    if sentiment_score > 0.9:
        if sentiment == 'positive':
            return 0.1
        elif sentiment == 'negative':
            return -0.1
    return 0.0

def analyze_sentiment_and_stock(text):
    # 1. Company Detection
    company_result = company_classifier(text, ['NVDA', 'ING', 'SAN', 'PFE', 'CSCO'])
    detected_company = company_result['labels'][0] if company_result['scores'][0] > 0.5 else None

    # 2. Sentiment Analysis
    sentiment_result = sentiment_classifier(text, ['positive', 'negative', 'neutral'])
    predicted_sentiment = sentiment_result['labels'][0]
    sentiment_score = sentiment_result['scores'][0]

    # 3. Adjust Output
    adjusted_output = adjust_output(predicted_sentiment, sentiment_score)

    return detected_company, predicted_sentiment, sentiment_score, adjusted_output