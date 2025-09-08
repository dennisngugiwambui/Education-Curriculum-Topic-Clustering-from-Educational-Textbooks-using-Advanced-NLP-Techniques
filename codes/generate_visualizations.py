#!/usr/bin/env python3
"""
Generate Professional Visualizations for README
Creates high-quality charts for curriculum topic clustering research
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Configure matplotlib for high-quality output
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def create_data_collection_visualization():
    """Create data collection results visualization"""
    
    plt.figure(figsize=(12, 8))
    
    # Sample data based on actual results
    categories = ['Items Collected', 'Quality Items Retained', 'Processing Success Rate']
    values = [96, 95, 98.96]  # 95/96 = 98.96%
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    plt.title('Educational Content Collection and Processing Results\nKenya Education Cloud Comprehensive Analysis', 
              fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Count / Percentage', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if i < 2:  # First two are counts
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=16)
        else:  # Third is percentage
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig('data_collection_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Generated: data_collection_analysis.png")

def create_nlp_performance_visualization():
    """Create NLP performance and configuration visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Topic Coherence Analysis
    coherence = 0.6322  # Actual coherence score
    coherence_data = [coherence, 1.0 - coherence]
    labels = [f'Coherence Score\n{coherence:.4f}', f'Potential Improvement\n{1-coherence:.4f}']
    colors = ['#06D6A0', '#EF476F']
    
    wedges, texts, autotexts = ax1.pie(coherence_data, labels=labels, colors=colors, 
                                      autopct='%1.2f%%', startangle=90, 
                                      textprops={'fontweight': 'bold', 'fontsize': 12},
                                      wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    
    ax1.set_title('Topic Coherence Analysis\nLDA Model Performance', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Right subplot: NLP Configuration
    config_labels = ['LDA Topics', 'Hierarchical\nClusters', 'Documents\nProcessed']
    config_values = [30, 25, 95]
    colors2 = ['#3A86FF', '#FF006E', '#8338EC']
    
    bars = ax2.bar(config_labels, config_values, color=colors2, alpha=0.85, 
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_title('NLP Analysis Configuration\nSystem Parameters', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, config_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(config_values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax2.set_ylim(0, max(config_values) * 1.15)
    
    plt.tight_layout()
    plt.savefig('nlp_performance_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Generated: nlp_performance_analysis.png")

if __name__ == "__main__":
    print("🎨 Generating professional visualizations for README...")
    
    # Set style for professional appearance
    plt.style.use('seaborn-v0_8-whitegrid')
    
    create_data_collection_visualization()
    create_nlp_performance_visualization()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Files created:")
    print("   • data_collection_analysis.png")
    print("   • nlp_performance_analysis.png")
