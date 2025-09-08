# PROFESSIONAL CURRICULUM TOPIC CLUSTERING ANALYSIS
# Educational Content Analysis using Advanced NLP Techniques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality professional output
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_research_results():
    """Load research results with comprehensive error handling"""
    
    try:
        # Look for the comprehensive report file
        report_files = list(Path('.').glob('*comprehensive_report*.json'))
        
        if report_files:
            latest_file = max(report_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded: {latest_file.name}")
            return data, latest_file.name
        
        print("❌ No comprehensive report found.")
        return None, None
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None

def display_professional_summary(data):
    """Display professional research summary"""
    
    if not data:
        return
    
    print("=" * 80)
    print("📊 CURRICULUM TOPIC CLUSTERING RESEARCH RESULTS")
    print("   Advanced NLP Analysis of Educational Content")
    print("=" * 80)
    
    # Research metadata
    metadata = data.get('research_metadata', {})
    print(f"\n📋 EXECUTION SUMMARY:")
    print(f"   Author: {metadata.get('author', 'Dennis Ngugi')}")
    print(f"   Duration: {metadata.get('execution_duration', 'N/A')}")
    print(f"   Phases Completed: {metadata.get('total_phases_completed', 0)}/4")
    print(f"   Errors: {metadata.get('errors_encountered', 0)}")
    
    # Key metrics
    methodology = data.get('methodology', {})
    data_collection = methodology.get('data_collection', {})
    eval_metrics = methodology.get('evaluation_metrics', {})
    
    print(f"\n📊 KEY PERFORMANCE METRICS:")
    print(f"   Documents Processed: {data_collection.get('total_documents', 0):,}")
    print(f"   Topic Coherence Score: {eval_metrics.get('topic_coherence', 0):.4f}")
    
    # Results
    results = data.get('results', {})
    data_results = results.get('data_collection_results', {})
    print(f"   Items Collected: {data_results.get('total_items_scraped', 0):,}")
    print(f"   Quality Items: {data_results.get('quality_items_retained', 0):,}")
    
    # Research objectives
    objectives = data.get('research_objectives', {})
    achieved = objectives.get('achieved', False)
    status = "✅ COMPLETED" if achieved else "⏳ IN PROGRESS"
    print(f"\n🎯 PROJECT STATUS: {status}")
    
    print("=" * 80)

def create_data_collection_chart(data):
    """Create data collection results chart"""
    
    results = data.get('results', {})
    data_results = results.get('data_collection_results', {})
    
    plt.figure(figsize=(12, 8))
    scraped = data_results.get('total_items_scraped', 0)
    retained = data_results.get('quality_items_retained', 0)
    
    categories = ['Items Collected', 'Quality Items']
    values = [scraped, retained]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    plt.title('Educational Content Collection Results\nKenya Education Cloud Portal Analysis', 
              fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Number of Documents', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig('1_data_collection_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Saved: 1_data_collection_results.png")

def create_coherence_analysis_chart(data):
    """Create topic coherence analysis chart"""
    
    methodology = data.get('methodology', {})
    eval_metrics = methodology.get('evaluation_metrics', {})
    
    plt.figure(figsize=(12, 8))
    coherence = eval_metrics.get('topic_coherence', 0)
    coherence_data = [coherence, 1.0 - coherence]
    labels = [f'Coherence Score\n{coherence:.4f}', f'Potential Improvement\n{1-coherence:.4f}']
    colors = ['#F18F01', '#C73E1D']
    
    wedges, texts, autotexts = plt.pie(coherence_data, labels=labels, colors=colors, 
                                      autopct='%1.2f%%', startangle=90, 
                                      textprops={'fontweight': 'bold', 'fontsize': 14},
                                      wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    
    plt.title('Topic Coherence Analysis\nLatent Dirichlet Allocation Model Performance', 
              fontsize=18, fontweight='bold', pad=25)
    
    plt.tight_layout()
    plt.savefig('2_topic_coherence_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Saved: 2_topic_coherence_analysis.png")

def create_nlp_configuration_chart(data):
    """Create NLP configuration chart"""
    
    methodology = data.get('methodology', {})
    nlp_techniques = methodology.get('nlp_techniques', {})
    
    plt.figure(figsize=(12, 8))
    topics_str = nlp_techniques.get('topic_modeling', 'LDA with 30 topics')
    clusters_str = nlp_techniques.get('clustering', 'Hierarchical clustering with 25 clusters')
    
    topics = 30
    clusters = 25
    
    # Extract actual numbers
    try:
        import re
        topic_match = re.search(r'(\d+)\s*topics?', topics_str, re.IGNORECASE)
        if topic_match:
            topics = int(topic_match.group(1))
        
        cluster_match = re.search(r'(\d+)\s*clusters?', clusters_str, re.IGNORECASE)
        if cluster_match:
            clusters = int(cluster_match.group(1))
    except:
        pass
    
    config_labels = ['LDA Topics', 'Hierarchical Clusters']
    config_values = [topics, clusters]
    colors = ['#3A86FF', '#FF006E']
    
    bars = plt.bar(config_labels, config_values, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    plt.title('NLP Analysis Configuration\nTopic Modeling and Clustering Parameters', 
              fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Parameter Count', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, config_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(config_values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    plt.ylim(0, max(config_values) * 1.15)
    plt.tight_layout()
    plt.savefig('3_nlp_configuration.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Saved: 3_nlp_configuration.png")

def create_execution_summary_chart(data):
    """Create pipeline execution summary chart"""
    
    metadata = data.get('research_metadata', {})
    
    plt.figure(figsize=(12, 8))
    phases = metadata.get('total_phases_completed', 0)
    errors = metadata.get('errors_encountered', 0)
    max_phases = 4
    
    summary_labels = ['Phases Completed', 'Errors Encountered']
    summary_values = [phases, errors]
    colors = ['#06D6A0', '#EF476F']
    
    bars = plt.bar(summary_labels, summary_values, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    duration = metadata.get('execution_duration', 'N/A')
    plt.title(f'Pipeline Execution Summary\nTotal Processing Time: {duration}', 
              fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.ylim(0, max(max_phases + 1, max(summary_values) + 1))
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, summary_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    # Add phase completion indicator
    plt.text(0, phases + 0.3, f'{phases}/{max_phases}', ha='center', va='bottom', 
            fontweight='bold', fontsize=14, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('4_pipeline_execution_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Saved: 4_pipeline_execution_summary.png")

# MAIN EXECUTION
if __name__ == "__main__":
    print("🔍 Loading Research Results...")
    
    # Load data
    research_data, filename = load_research_results()
    
    if research_data:
        # Display summary
        display_professional_summary(research_data)
        
        # Create individual visualizations
        print("\n📈 Creating professional visualizations...")
        create_data_collection_chart(research_data)
        create_coherence_analysis_chart(research_data)
        create_nlp_configuration_chart(research_data)
        create_execution_summary_chart(research_data)
        
        print(f"\n🎉 ANALYSIS COMPLETE")
        print(f"📁 Source Report: {filename}")
        print(f"🖼️ Generated Visualizations:")
        print(f"   • 1_data_collection_results.png")
        print(f"   • 2_topic_coherence_analysis.png") 
        print(f"   • 3_nlp_configuration.png")
        print(f"   • 4_pipeline_execution_summary.png")
        print(f"🚀 Ready for professional presentation and publication")
        
    else:
        print("\n❌ No research results found.")
        print("Please ensure the pipeline has been executed successfully.")
