# Machine Learning Classification and Question Answering System
# Part 2 of KEC Education Analysis

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import pickle
from collections import Counter

class EducationContentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.subject_keywords = {
            'mathematics': ['math', 'algebra', 'geometry', 'calculus', 'equation', 'formula', 'number', 'calculate'],
            'science': ['biology', 'chemistry', 'physics', 'experiment', 'hypothesis', 'theory', 'cell', 'atom'],
            'english': ['grammar', 'literature', 'essay', 'poem', 'novel', 'writing', 'reading', 'language'],
            'history': ['historical', 'past', 'ancient', 'civilization', 'war', 'empire', 'culture', 'timeline'],
            'geography': ['climate', 'continent', 'country', 'map', 'region', 'population', 'environment', 'location']
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def classify_by_keywords(self, text):
        """Classify content based on subject keywords"""
        text_lower = text.lower()
        subject_scores = {}
        
        for subject, keywords in self.subject_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            subject_scores[subject] = score
        
        if max(subject_scores.values()) > 0:
            return max(subject_scores, key=subject_scores.get)
        return 'general'
    
    def extract_grade_level(self, title, content):
        """Extract grade/form level from title or content"""
        text = (title + " " + content).lower()
        
        # Look for form indicators
        for i in range(1, 5):
            if f'form {i}' in text or f'form{i}' in text:
                return f'form_{i}'
        
        # Look for grade indicators
        for i in range(1, 13):
            if f'grade {i}' in text or f'grade{i}' in text:
                return f'grade_{i}'
        
        return 'unknown'
    
    def prepare_training_data(self, df):
        """Prepare data for machine learning"""
        # Preprocess content
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Extract features
        df['subject'] = df['content'].apply(self.classify_by_keywords)
        df['grade_level'] = df.apply(lambda x: self.extract_grade_level(x['title'], x['content']), axis=1)
        df['content_length'] = df['content'].str.len()
        df['word_count'] = df['processed_content'].str.split().str.len()
        
        return df
    
    def train_classifier(self, df, target_column='subject'):
        """Train the classification model"""
        # Prepare features
        X = df['processed_content']
        y = df[target_column]
        
        # Remove samples with unknown labels
        mask = y != 'unknown'
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("No valid training data available")
            return None
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5)
            avg_score = scores.mean()
            print(f"{name}: {avg_score:.3f} (+/- {scores.std() * 2:.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model
        self.classifier = best_model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print(f"\nBest Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.classifier
    
    def predict_content(self, text):
        """Predict subject and other attributes for new content"""
        if self.classifier is None or self.vectorizer is None:
            return "Model not trained"
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(vectorized_text)[0]
        probability = self.classifier.predict_proba(vectorized_text)[0].max()
        
        return {
            'predicted_subject': prediction,
            'confidence': probability,
            'grade_level': self.extract_grade_level('', text)
        }

class QuestionAnsweringSystem:
    def __init__(self, df_content):
        self.content_df = df_content
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_vectors = None
        self.setup_qa_system()
    
    def setup_qa_system(self):
        """Setup the question answering system"""
        # Combine title and content for better context
        self.content_df['full_text'] = self.content_df['title'] + " " + self.content_df['content']
        
        # Vectorize all content
        self.content_vectors = self.vectorizer.fit_transform(self.content_df['full_text'])
    
    def find_relevant_content(self, question, top_k=5):
        """Find most relevant content for a question"""
        # Vectorize the question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarity
        similarities = (self.content_vectors * question_vector.T).toarray().flatten()
        
        # Get top k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'content': self.content_df.iloc[idx]['content'][:500] + "...",
                    'title': self.content_df.iloc[idx]['title'],
                    'similarity': similarities[idx],
                    'url': self.content_df.iloc[idx].get('url', 'N/A')
                })
        
        return results
    
    def answer_question(self, question):
        """Generate answer for a question based on scraped content"""
        relevant_content = self.find_relevant_content(question)
        
        if not relevant_content:
            return "No relevant content found for this question."
        
        # Simple extractive answer generation
        answer = f"Based on the educational content from KEC:\n\n"
        
        for i, content in enumerate(relevant_content[:3], 1):
            answer += f"{i}. From '{content['title']}':\n"
            answer += f"   {content['content']}\n"
            answer += f"   (Relevance: {content['similarity']:.3f})\n\n"
        
        return answer

def analyze_content_topics(df, n_topics=10):
    """Perform topic modeling on the content"""
    # Preprocess text
    classifier = EducationContentClassifier()
    processed_texts = df['content'].apply(classifier.preprocess_text)
    
    # Remove empty texts
    processed_texts = processed_texts[processed_texts.str.len() > 0]
    
    if len(processed_texts) == 0:
        print("No valid text content for topic analysis")
        return None
    
    # Vectorize
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    
    # LDA Topic Modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display topics
    print("Discovered Topics:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return lda, vectorizer

def create_visualizations(df):
    """Create visualizations for the analysis"""
    plt.figure(figsize=(15, 10))
    
    # Subject distribution
    plt.subplot(2, 3, 1)
    if 'subject' in df.columns:
        subject_counts = df['subject'].value_counts()
        plt.pie(subject_counts.values, labels=subject_counts.index, autopct='%1.1f%%')
        plt.title('Subject Distribution')
    
    # Grade level distribution
    plt.subplot(2, 3, 2)
    if 'grade_level' in df.columns:
        grade_counts = df['grade_level'].value_counts()
        plt.bar(grade_counts.index, grade_counts.values)
        plt.title('Grade Level Distribution')
        plt.xticks(rotation=45)
    
    # Content length distribution
    plt.subplot(2, 3, 3)
    if 'content_length' in df.columns:
        plt.hist(df['content_length'], bins=20, alpha=0.7)
        plt.title('Content Length Distribution')
        plt.xlabel('Character Count')
    
    # Word count distribution
    plt.subplot(2, 3, 4)
    if 'word_count' in df.columns:
        plt.hist(df['word_count'], bins=20, alpha=0.7)
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
    
    # Word cloud
    plt.subplot(2, 3, 5)
    if 'processed_content' in df.columns:
        all_text = ' '.join(df['processed_content'].dropna())
        if all_text:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud')
    
    plt.tight_layout()
    plt.savefig('education_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(classifier, filename='education_classifier.pkl'):
    """Save the trained model"""
    with open(filename, 'wb') as f:
        pickle.dump({
            'classifier': classifier.classifier,
            'vectorizer': classifier.vectorizer,
            'subject_keywords': classifier.subject_keywords
        }, f)
    print(f"Model saved as {filename}")

def load_model(filename='education_classifier.pkl'):
    """Load a saved model"""
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = EducationContentClassifier()
        classifier.classifier = model_data['classifier']
        classifier.vectorizer = model_data['vectorizer']
        classifier.subject_keywords = model_data['subject_keywords']
        
        return classifier
    except FileNotFoundError:
        print(f"Model file {filename} not found")
        return None

# Example usage and main execution
if __name__ == "__main__":
    print("Education Content ML Analysis System")
    print("=" * 50)
    
    # This would be used with scraped data
    # df = pd.read_csv('scraped_content.csv')  # Load your scraped data
    
    # For demonstration with sample data
    sample_data = {
        'title': ['Form 1 Mathematics', 'Form 2 Biology', 'Form 3 Chemistry', 'Form 4 Physics'],
        'content': [
            'Introduction to algebra and basic equations. Numbers and calculations.',
            'Study of living organisms, cells, and biological processes.',
            'Chemical reactions, atoms, and molecular structures.',
            'Physics concepts, motion, energy, and forces in nature.'
        ],
        'level': ['secondary'] * 4,
        'url': ['http://example.com'] * 4
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize classifier
    classifier = EducationContentClassifier()
    
    # Prepare data
    df = classifier.prepare_training_data(df)
    print("Data prepared successfully")
    
    # Train classifier
    if len(df) > 1:
        classifier.train_classifier(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Setup QA system
        qa_system = QuestionAnsweringSystem(df)
        
        # Example questions
        questions = [
            "What is algebra?",
            "How do cells work?",
            "What are chemical reactions?",
            "Explain physics concepts"
        ]
        
        print("\nQuestion Answering Examples:")
        print("=" * 30)
        for question in questions:
            answer = qa_system.answer_question(question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
        
        # Save model
        save_model(classifier)
        
    print("Analysis complete!")
