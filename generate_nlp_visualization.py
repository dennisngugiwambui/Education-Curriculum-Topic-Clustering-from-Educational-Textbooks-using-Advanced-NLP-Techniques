#!/usr/bin/env python3
"""
Generate NLP Performance Visualization for README
Creates the dashboard-style visualization matching the second image
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure matplotlib for high-quality output
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def create_nlp_dashboard():
    """Create the comprehensive NLP dashboard visualization"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Add main title
    fig.suptitle('Masters Research: Curriculum Topic Clustering Results', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 1. Data Collection Results (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Items\nScraped', 'Quality\nRetained']
    values = [95, 94]
    colors = ['#3498db', '#2ecc71']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    ax1.set_title('Data Collection Results', fontweight='bold', fontsize=14, pad=15)
    ax1.set_ylabel('Number of Documents', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Topic Coherence Analysis (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    coherence = 0.627
    remaining = 1 - coherence
    
    sizes = [coherence, remaining]
    labels = [f'Coherence\n{coherence:.3f}', f'Remaining\n{remaining:.3f}']
    colors2 = ['#e74c3c', '#ecf0f1']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors2, 
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontweight': 'bold', 'fontsize': 10})
    ax2.set_title('Topic Coherence Analysis', fontweight='bold', fontsize=14, pad=15)
    
    # 3. NLP Analysis Configuration (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    config_labels = ['LDA\nTopics', 'Hierarchical\nClusters']
    config_values = [30, 25]
    colors3 = ['#9b59b6', '#f39c12']
    
    bars3 = ax3.bar(config_labels, config_values, color=colors3, alpha=0.8,
                    edgecolor='black', linewidth=1)
    ax3.set_title('NLP Analysis Configuration', fontweight='bold', fontsize=14, pad=15)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_ylim(0, 35)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, config_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 4. Pipeline Execution Summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    pipeline_labels = ['Phases\nCompleted', 'Errors\nEncountered']
    pipeline_values = [4, 0]
    colors4 = ['#27ae60', '#e67e22']
    
    bars4 = ax4.bar(pipeline_labels, pipeline_values, color=colors4, alpha=0.8,
                    edgecolor='black', linewidth=1)
    ax4.set_title('Pipeline Execution Summary\nDuration: 0:09:32.418482', 
                  fontweight='bold', fontsize=14, pad=15)
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_ylim(0, 5)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, pipeline_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add summary statistics text box (bottom span)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = """
    Research Summary Statistics:
    • Processing Success Rate: 98.94% (94/95 documents successfully analyzed)
    • Topic Coherence Score: 0.627 (indicating high-quality topic separation)
    • Analysis Configuration: 30 LDA topics, 25 hierarchical clusters optimized for educational content
    • Pipeline Execution: 4 phases completed successfully with zero errors in 9 minutes 32 seconds
    • Data Quality: Comprehensive Kenya Education Cloud content analysis with robust validation
    """
    
    ax5.text(0.05, 0.8, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('nlp_performance_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("📊 Generated: nlp_performance_analysis.png")

if __name__ == "__main__":
    print("🎨 Generating NLP Performance Analysis Dashboard...")
    
    # Set style for professional appearance
    plt.style.use('default')
    
    create_nlp_dashboard()
    
    print("\n✅ NLP Performance visualization generated successfully!")
    print("📁 File created: nlp_performance_analysis.png")
