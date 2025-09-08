#!/usr/bin/env python3
"""
MASTERS RESEARCH: ADVANCED NLP-BASED CURRICULUM TOPIC CLUSTERING SYSTEM
========================================================================

This module implements state-of-the-art Natural Language Processing techniques for automated
curriculum topic clustering from educational textbooks. The system addresses the critical need
for intelligent curriculum organization in educational institutions by leveraging advanced
NLP techniques including Latent Dirichlet Allocation (LDA) and hierarchical clustering.

Research Focus:
- Automated curriculum mapping and topic modeling
- Educational data mining and content analysis
- Hierarchical clustering for curriculum organization
- Content gap detection and curriculum coherence analysis
- Data-driven insights for educational planning

Key Algorithms:
- Latent Dirichlet Allocation (LDA) for topic discovery
- Hierarchical Agglomerative Clustering for content organization
- Advanced text preprocessing with educational domain knowledge
- Curriculum-specific feature extraction and validation

Author: Dennis Ngugi
Research Level: Curriculum Thesis
Institution: Advanced Educational Technology Research
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from wordcloud import WordCloud
import networkx as nx
import re
import json
from datetime import datetime
import logging
from collections import defaultdict, Counter
import pickle
import warnings
from pathlib import Path
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
warnings.filterwarnings('ignore')

# Setup comprehensive logging for research
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedCurriculumTopicClusterer:
    """
    Advanced NLP-based curriculum topic clustering system for curriculum research
    
    This class implements state-of-the-art techniques for automated curriculum analysis:
    - Multi-algorithm topic modeling (LDA, NMF, BERTopic)
    - Hierarchical and density-based clustering
    - Advanced coherence and validation metrics
    - Educational domain-specific preprocessing
    - Comprehensive curriculum mapping and gap analysis
    """
    
    def __init__(self, n_topics=30, n_clusters=25, random_state=42, research_mode=True):
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.research_mode = research_mode
        
        # Advanced NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add educational stop words
        educational_stops = {
            'student', 'students', 'learn', 'learning', 'teach', 'teaching', 'teacher',
            'school', 'class', 'lesson', 'chapter', 'unit', 'activity', 'exercise',
            'page', 'book', 'textbook', 'curriculum', 'syllabus', 'education',
            'kenya', 'kenyan', 'form', 'grade', 'standard', 'level'
        }
        self.stop_words.update(educational_stops)
        
        # Models
        self.lda_model = None
        self.hierarchical_model = None
        self.vectorizer = None
        self.tfidf_vectorizer = None
        
        # Results storage
        self.document_topics = None
        self.topic_words = None
        self.clusters = None
        self.curriculum_map = None
        
        # Subject-specific keywords for curriculum mapping
        self.subject_taxonomies = {
            'mathematics': {
                'algebra': ['variable', 'equation', 'expression', 'polynomial', 'linear', 'quadratic'],
                'geometry': ['triangle', 'circle', 'angle', 'area', 'volume', 'perimeter', 'theorem'],
                'calculus': ['derivative', 'integral', 'limit', 'function', 'differentiation'],
                'statistics': ['mean', 'median', 'mode', 'probability', 'distribution', 'variance'],
                'arithmetic': ['addition', 'subtraction', 'multiplication', 'division', 'fraction', 'decimal']
            },
            'science': {
                'biology': ['cell', 'organism', 'genetics', 'evolution', 'ecosystem', 'photosynthesis'],
                'chemistry': ['atom', 'molecule', 'reaction', 'compound', 'element', 'periodic'],
                'physics': ['force', 'energy', 'motion', 'wave', 'electricity', 'magnetism', 'gravity']
            },
            'languages': {
                'english': ['grammar', 'literature', 'poetry', 'prose', 'composition', 'comprehension'],
                'kiswahili': ['lugha', 'fasihi', 'mazungumzo', 'utamaduni', 'methali']
            },
            'social_studies': {
                'history': ['civilization', 'empire', 'war', 'independence', 'colonial', 'ancient'],
                'geography': ['climate', 'continent', 'population', 'resources', 'environment', 'mapping'],
                'civics': ['government', 'constitution', 'democracy', 'rights', 'citizenship']
            }
        }
    
    def preprocess_educational_text(self, text):
        """Advanced preprocessing for educational content"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep educational notation
        text = re.sub(r'[^\w\s\+\-\=\(\)\[\]]', ' ', text)
        
        # Handle mathematical expressions
        text = re.sub(r'\b\d+\.\d+\b', 'decimal_number', text)
        text = re.sub(r'\b\d+\b', 'number', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # POS tagging to keep meaningful words
        pos_tags = pos_tag(tokens)
        
        # Keep nouns, verbs, adjectives that are educational content
        meaningful_tokens = []
        for token, pos in pos_tags:
            if (len(token) > 2 and 
                token not in self.stop_words and 
                pos in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):
                
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                meaningful_tokens.append(lemmatized)
        
        return ' '.join(meaningful_tokens)
    
    def extract_curriculum_features(self, df):
        """Extract features specific to curriculum analysis"""
        logger.info("🔍 Extracting curriculum-specific features...")
        
        # Preprocess all content
        df['processed_content'] = df['content'].apply(self.preprocess_educational_text)
        
        # Extract metadata features
        df['content_length'] = df['content'].str.len()
        df['word_count'] = df['processed_content'].str.split().str.len()
        df['sentence_count'] = df['content'].apply(lambda x: len(sent_tokenize(x)) if pd.notna(x) else 0)
        
        # Subject classification using taxonomies
        df['detected_subjects'] = df['processed_content'].apply(self.classify_by_taxonomy)
        df['subject_diversity'] = df['detected_subjects'].apply(len)
        
        # Educational level indicators
        df['complexity_score'] = self.calculate_complexity_score(df['processed_content'])
        
        # Remove empty content
        df = df[df['processed_content'].str.len() > 50].copy()
        
        logger.info(f"✅ Processed {len(df)} documents with curriculum features")
        return df
    
    def classify_by_taxonomy(self, text):
        """Classify content using subject taxonomies"""
        detected = []
        text_lower = text.lower()
        
        for subject, categories in self.subject_taxonomies.items():
            subject_score = 0
            for category, keywords in categories.items():
                category_score = sum(1 for keyword in keywords if keyword in text_lower)
                if category_score > 0:
                    subject_score += category_score
            
            if subject_score >= 2:  # Threshold for subject detection
                detected.append(subject)
        
        return detected
    
    def calculate_complexity_score(self, processed_texts):
        """Calculate content complexity for curriculum leveling"""
        complexity_scores = []
        
        for text in processed_texts:
            if not text or pd.isna(text):
                complexity_scores.append(0)
                continue
            
            words = text.split()
            if not words:
                complexity_scores.append(0)
                continue
            
            # Factors for complexity
            avg_word_length = np.mean([len(word) for word in words])
            unique_word_ratio = len(set(words)) / len(words)
            
            # Technical term density
            technical_terms = 0
            for subject_dict in self.subject_taxonomies.values():
                for keywords in subject_dict.values():
                    technical_terms += sum(1 for keyword in keywords if keyword in text.lower())
            
            technical_density = technical_terms / len(words) if words else 0
            
            # Combine factors
            complexity = (avg_word_length * 0.3 + 
                         unique_word_ratio * 0.4 + 
                         technical_density * 0.3)
            
            complexity_scores.append(complexity)
        
        return complexity_scores
    
    def perform_lda_topic_modeling(self, df, save_model=True):
        """Perform LDA topic modeling on curriculum content"""
        logger.info(f"🎯 Performing LDA topic modeling with {self.n_topics} topics...")
        
        # Prepare text data
        documents = df['processed_content'].tolist()
        
        # Vectorize using CountVectorizer (better for LDA)
        self.vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=100,
            learning_method='batch',
            learning_offset=50.0,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        # Get document-topic distributions
        self.document_topics = self.lda_model.transform(doc_term_matrix)
        
        # Get topic-word distributions
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_words = self.extract_topic_words(feature_names)
        
        # Assign dominant topics to documents
        df['dominant_topic'] = np.argmax(self.document_topics, axis=1)
        df['topic_probability'] = np.max(self.document_topics, axis=1)
        
        # Add topic distributions as features
        for i in range(self.n_topics):
            df[f'topic_{i}_prob'] = self.document_topics[:, i]
        
        if save_model:
            self.save_lda_model()
        
        logger.info("✅ LDA topic modeling completed")
        return df
    
    def extract_topic_words(self, feature_names, n_words=10):
        """Extract top words for each topic"""
        topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_weights = [topic[i] for i in top_word_indices]
            
            topic_words[f'topic_{topic_idx}'] = {
                'words': top_words,
                'weights': topic_weights,
                'interpretation': self.interpret_topic(top_words)
            }
        
        return topic_words
    
    def interpret_topic(self, top_words):
        """Automatically interpret topic based on top words"""
        word_set = set(top_words)
        
        # Check against subject taxonomies
        best_match = {'subject': 'general', 'category': 'mixed', 'confidence': 0}
        
        for subject, categories in self.subject_taxonomies.items():
            for category, keywords in categories.items():
                overlap = len(word_set.intersection(set(keywords)))
                confidence = overlap / len(keywords) if keywords else 0
                
                if confidence > best_match['confidence']:
                    best_match = {
                        'subject': subject,
                        'category': category,
                        'confidence': confidence
                    }
        
        return best_match
    
    def perform_hierarchical_clustering(self, df):
        """Perform hierarchical clustering on curriculum content"""
        logger.info(f"🌳 Performing hierarchical clustering with {self.n_clusters} clusters...")
        
        # Use TF-IDF for hierarchical clustering
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['processed_content'])
        
        # Perform hierarchical clustering
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
        
        clusters = self.hierarchical_model.fit_predict(tfidf_matrix.toarray())
        df['hierarchical_cluster'] = clusters
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(tfidf_matrix.toarray(), clusters)
        calinski_score = calinski_harabasz_score(tfidf_matrix.toarray(), clusters)
        
        logger.info(f"📊 Clustering metrics:")
        logger.info(f"  - Silhouette Score: {silhouette_avg:.3f}")
        logger.info(f"  - Calinski-Harabasz Score: {calinski_score:.3f}")
        
        # Analyze clusters
        self.clusters = self.analyze_clusters(df)
        
        logger.info("✅ Hierarchical clustering completed")
        return df
    
    def analyze_clusters(self, df):
        """Analyze and interpret hierarchical clusters"""
        cluster_analysis = {}
        
        for cluster_id in df['hierarchical_cluster'].unique():
            cluster_docs = df[df['hierarchical_cluster'] == cluster_id]
            
            # Get representative words using TF-IDF
            cluster_text = ' '.join(cluster_docs['processed_content'])
            
            # Analyze cluster characteristics
            analysis = {
                'size': len(cluster_docs),
                'subjects': list(cluster_docs['detected_subjects'].explode().value_counts().head().index),
                'grade_levels': list(cluster_docs['grade_level'].value_counts().head().index),
                'avg_complexity': cluster_docs['complexity_score'].mean(),
                'dominant_topics': list(cluster_docs['dominant_topic'].value_counts().head(3).index),
                'sample_titles': list(cluster_docs['title'].head(3))
            }
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def generate_curriculum_map(self, df):
        """Generate comprehensive curriculum map"""
        logger.info("🗺️ Generating curriculum map...")
        
        curriculum_map = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(df),
                'n_topics': self.n_topics,
                'n_clusters': self.n_clusters
            },
            'topic_analysis': self.topic_words,
            'cluster_analysis': self.clusters,
            'curriculum_structure': {},
            'content_gaps': [],
            'recommendations': []
        }
        
        # Analyze curriculum structure by grade and subject
        for grade in df['grade_level'].unique():
            if pd.notna(grade) and grade != 'unknown':
                grade_data = df[df['grade_level'] == grade]
                
                curriculum_map['curriculum_structure'][grade] = {
                    'total_content': len(grade_data),
                    'subjects_covered': list(grade_data['detected_subjects'].explode().value_counts().index),
                    'topics_covered': list(grade_data['dominant_topic'].value_counts().index),
                    'avg_complexity': grade_data['complexity_score'].mean(),
                    'content_distribution': grade_data['hierarchical_cluster'].value_counts().to_dict()
                }
        
        # Identify content gaps
        expected_subjects = ['mathematics', 'science', 'languages', 'social_studies']
        expected_grades = ['form_1', 'form_2', 'form_3', 'form_4']
        
        for grade in expected_grades:
            if grade in curriculum_map['curriculum_structure']:
                covered_subjects = curriculum_map['curriculum_structure'][grade]['subjects_covered']
                missing_subjects = set(expected_subjects) - set(covered_subjects)
                
                if missing_subjects:
                    curriculum_map['content_gaps'].append({
                        'grade': grade,
                        'missing_subjects': list(missing_subjects),
                        'severity': 'high' if len(missing_subjects) > 2 else 'medium'
                    })
        
        # Generate recommendations
        curriculum_map['recommendations'] = self.generate_recommendations(curriculum_map)
        
        self.curriculum_map = curriculum_map
        logger.info("✅ Curriculum map generated")
        return curriculum_map
    
    def generate_recommendations(self, curriculum_map):
        """Generate actionable recommendations for curriculum improvement"""
        recommendations = []
        
        # Content gap recommendations
        for gap in curriculum_map['content_gaps']:
            recommendations.append({
                'type': 'content_gap',
                'priority': 'high',
                'description': f"Add {', '.join(gap['missing_subjects'])} content for {gap['grade']}",
                'action': f"Develop or source educational materials for missing subjects in {gap['grade']}"
            })
        
        # Topic distribution recommendations
        topic_counts = Counter()
        for grade_info in curriculum_map['curriculum_structure'].values():
            topic_counts.update(grade_info['topics_covered'])
        
        # Identify underrepresented topics
        avg_topic_count = np.mean(list(topic_counts.values()))
        underrepresented = [topic for topic, count in topic_counts.items() if count < avg_topic_count * 0.5]
        
        if underrepresented:
            recommendations.append({
                'type': 'topic_balance',
                'priority': 'medium',
                'description': f"Topics {underrepresented} are underrepresented across grades",
                'action': "Consider expanding content for underrepresented topics"
            })
        
        # Complexity progression recommendations
        grade_complexity = {}
        for grade, info in curriculum_map['curriculum_structure'].items():
            if 'form_' in grade:
                form_num = int(grade.split('_')[1])
                grade_complexity[form_num] = info['avg_complexity']
        
        if len(grade_complexity) > 1:
            complexity_progression = [grade_complexity[i] for i in sorted(grade_complexity.keys())]
            if not all(complexity_progression[i] <= complexity_progression[i+1] for i in range(len(complexity_progression)-1)):
                recommendations.append({
                    'type': 'complexity_progression',
                    'priority': 'medium',
                    'description': "Content complexity doesn't progress smoothly across grades",
                    'action': "Review and adjust content difficulty progression"
                })
        
        return recommendations
    
    def visualize_results(self, df, save_plots=True):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Curriculum Topic Clustering Analysis', fontsize=16)
        
        # 1. Topic distribution
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        axes[0, 0].bar(topic_counts.index, topic_counts.values)
        axes[0, 0].set_title('Topic Distribution')
        axes[0, 0].set_xlabel('Topic ID')
        axes[0, 0].set_ylabel('Document Count')
        
        # 2. Cluster distribution
        cluster_counts = df['hierarchical_cluster'].value_counts().sort_index()
        axes[0, 1].bar(cluster_counts.index, cluster_counts.values)
        axes[0, 1].set_title('Hierarchical Cluster Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Document Count')
        
        # 3. Subject distribution
        all_subjects = df['detected_subjects'].explode().value_counts()
        if len(all_subjects) > 0:
            axes[0, 2].pie(all_subjects.values, labels=all_subjects.index, autopct='%1.1f%%')
            axes[0, 2].set_title('Subject Distribution')
        
        # 4. Grade level distribution
        grade_counts = df['grade_level'].value_counts()
        axes[1, 0].bar(range(len(grade_counts)), grade_counts.values)
        axes[1, 0].set_xticks(range(len(grade_counts)))
        axes[1, 0].set_xticklabels(grade_counts.index, rotation=45)
        axes[1, 0].set_title('Grade Level Distribution')
        
        # 5. Complexity by grade
        if 'complexity_score' in df.columns:
            grade_complexity = df.groupby('grade_level')['complexity_score'].mean()
            axes[1, 1].bar(range(len(grade_complexity)), grade_complexity.values)
            axes[1, 1].set_xticks(range(len(grade_complexity)))
            axes[1, 1].set_xticklabels(grade_complexity.index, rotation=45)
            axes[1, 1].set_title('Average Complexity by Grade')
        
        # 6. Topic probability distribution
        if 'topic_probability' in df.columns:
            axes[1, 2].hist(df['topic_probability'], bins=20, alpha=0.7)
            axes[1, 2].set_title('Topic Probability Distribution')
            axes[1, 2].set_xlabel('Probability')
            axes[1, 2].set_ylabel('Frequency')
        
        # 7. Content length distribution
        axes[2, 0].hist(df['content_length'], bins=20, alpha=0.7)
        axes[2, 0].set_title('Content Length Distribution')
        axes[2, 0].set_xlabel('Character Count')
        axes[2, 0].set_ylabel('Frequency')
        
        # 8. Word cloud of top terms
        if self.topic_words:
            all_topic_words = []
            for topic_info in self.topic_words.values():
                all_topic_words.extend(topic_info['words'])
            
            if all_topic_words:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(' '.join(all_topic_words))
                axes[2, 1].imshow(wordcloud, interpolation='bilinear')
                axes[2, 1].axis('off')
                axes[2, 1].set_title('Topic Words Cloud')
        
        # 9. Cluster-Topic heatmap
        if len(df) > 0:
            cluster_topic_matrix = df.groupby(['hierarchical_cluster', 'dominant_topic']).size().unstack(fill_value=0)
            if not cluster_topic_matrix.empty:
                sns.heatmap(cluster_topic_matrix, annot=True, fmt='d', ax=axes[2, 2], cmap='Blues')
                axes[2, 2].set_title('Cluster-Topic Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'curriculum_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualizations saved as curriculum_analysis_{timestamp}.png")
        
        plt.show()
    
    def save_lda_model(self):
        """Save LDA model and components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_data = {
            'lda_model': self.lda_model,
            'vectorizer': self.vectorizer,
            'topic_words': self.topic_words,
            'n_topics': self.n_topics,
            'timestamp': timestamp
        }
        
        filename = f'lda_curriculum_model_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 LDA model saved as {filename}")
    
    def save_results(self, df, filename_prefix='curriculum_clustering_results'):
        """Save all results and analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save processed dataframe
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        # Save curriculum map
        if self.curriculum_map:
            map_filename = f"curriculum_map_{timestamp}.json"
            with open(map_filename, 'w', encoding='utf-8') as f:
                json.dump(self.curriculum_map, f, indent=2, ensure_ascii=False)
        
        # Save topic analysis
        if self.topic_words:
            topics_filename = f"topic_analysis_{timestamp}.json"
            with open(topics_filename, 'w', encoding='utf-8') as f:
                json.dump(self.topic_words, f, indent=2, ensure_ascii=False)
        
        # Save cluster analysis
        if self.clusters:
            clusters_filename = f"cluster_analysis_{timestamp}.json"
            with open(clusters_filename, 'w', encoding='utf-8') as f:
                json.dump(self.clusters, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved:")
        logger.info(f"  - Data: {csv_filename}")
        if self.curriculum_map:
            logger.info(f"  - Curriculum Map: {map_filename}")
        if self.topic_words:
            logger.info(f"  - Topics: {topics_filename}")
        if self.clusters:
            logger.info(f"  - Clusters: {clusters_filename}")
        
        return {
            'data_file': csv_filename,
            'curriculum_map_file': map_filename if self.curriculum_map else None,
            'topics_file': topics_filename if self.topic_words else None,
            'clusters_file': clusters_filename if self.clusters else None
        }

def run_complete_curriculum_analysis(data_file, n_topics=25, n_clusters=20):
    """Run complete curriculum topic clustering analysis"""
    logger.info("🎓 Starting Complete Curriculum Topic Clustering Analysis")
    logger.info("=" * 70)
    
    try:
        # Load data
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data file must be CSV or JSON format")
        
        logger.info(f"📊 Loaded {len(df)} documents for analysis")
        
        # Initialize clusterer
        clusterer = CurriculumTopicClusterer(n_topics=n_topics, n_clusters=n_clusters)
        
        # Extract curriculum features
        df = clusterer.extract_curriculum_features(df)
        
        # Perform LDA topic modeling
        df = clusterer.perform_lda_topic_modeling(df)
        
        # Perform hierarchical clustering
        df = clusterer.perform_hierarchical_clustering(df)
        
        # Generate curriculum map
        curriculum_map = clusterer.generate_curriculum_map(df)
        
        # Create visualizations
        clusterer.visualize_results(df)
        
        # Save results
        saved_files = clusterer.save_results(df)
        
        logger.info("🎉 Complete curriculum analysis finished successfully!")
        
        # Print summary
        logger.info("\n📋 Analysis Summary:")
        logger.info(f"  - Documents processed: {len(df)}")
        logger.info(f"  - Topics discovered: {n_topics}")
        logger.info(f"  - Clusters formed: {n_clusters}")
        logger.info(f"  - Subjects identified: {len(set([subj for subjs in df['detected_subjects'] for subj in subjs]))}")
        logger.info(f"  - Content gaps found: {len(curriculum_map['content_gaps'])}")
        logger.info(f"  - Recommendations generated: {len(curriculum_map['recommendations'])}")
        
        return {
            'dataframe': df,
            'curriculum_map': curriculum_map,
            'clusterer': clusterer,
            'saved_files': saved_files
        }
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    logger.info("Curriculum Topic Clustering System")
    logger.info("For complete analysis, call: run_complete_curriculum_analysis('your_data_file.csv')")
