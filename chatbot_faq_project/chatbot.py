import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

faqs = {
    "How can I track my order?": "You can track your order in the 'My Orders' section of the app.",
    "What payment methods are accepted?": "We accept credit/debit cards, UPI, and digital wallets.",
    "How do I cancel my order?": "Go to 'My Orders' and select the order you want to cancel.",
    "What should I do if my food is cold?": "You can raise a complaint in the Help section for a quick resolution.",
    "Is contactless delivery available?": "Yes, contactless delivery is available at checkout."
}

stop_words = set(stopwords.words('english'))

# def preprocess(text):
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))
#     tokens = word_tokenize(text)
#     filtered = [word for word in tokens if word not in stop_words]
#     return " ".join(filtered)
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, preserve_line=True)  # 👈 Add preserve_line=True here
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered)

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def find_best_answer(user_question):
    user_processed = preprocess(user_question)
    user_vector = vectorizer.transform([user_processed])
    
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]
    
    if best_score > 0.2:
        return faqs[questions[best_match_index]]
    else:
        return "Sorry, I couldn't understand your question. Please try again."

def run_chatbot():
    print("Welcome to the FoodApp FAQ Chatbot! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Thank you! Have a great day!")
            break
        response = find_best_answer(user_input)
        print("Chatbot:", response)

run_chatbot()
