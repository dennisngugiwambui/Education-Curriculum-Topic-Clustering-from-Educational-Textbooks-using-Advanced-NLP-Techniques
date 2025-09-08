#!/usr/bin/env python3
"""
Comprehensive KEC Educational Content Scraper
Designed for curriculum topic clustering and NLP analysis
Extracts ALL textbooks and educational materials from Kenyan high school curriculum
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import json
import os
from urllib.parse import urljoin, urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveKECScraper:
    def __init__(self, max_workers=5, delay_between_requests=1):
        self.base_urls = {
            'main': 'https://kec.ac.ke/',
            'lms': 'https://lms.kec.ac.ke/',
            'elimika': 'https://elimika.kec.ac.ke/',
            'oer': 'https://oer.kec.ac.ke/',
            'resources': 'https://resources.kec.ac.ke/',
            'orangebook': 'https://orangebook.kec.ac.ke/'
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.max_workers = max_workers
        self.delay = delay_between_requests
        self.scraped_data = []
        self.discovered_urls = set()
        self.failed_urls = []
        self.lock = threading.Lock()
        
        # Category IDs discovered from KEC structure
        self.category_mappings = {
            # Pre-Primary
            69: {'level': 'pre_primary', 'grades': ['pp1', 'pp2']},
            
            # Primary (CBC)
            74: {'level': 'primary', 'grade': 'grade_1', 'curriculum': 'cbc'},
            80: {'level': 'primary', 'grade': 'grade_2', 'curriculum': 'cbc'},
            90: {'level': 'primary', 'grade': 'grade_3', 'curriculum': 'cbc'},
            92: {'level': 'primary', 'grade': 'grade_4', 'curriculum': 'cbc'},
            94: {'level': 'primary', 'grade': 'grade_5', 'curriculum': 'cbc'},
            151: {'level': 'primary', 'grade': 'grade_6', 'curriculum': 'cbc'},
            
            # Junior Secondary (CBC)
            102: {'level': 'junior_secondary', 'grade': 'grade_7', 'curriculum': 'cbc'},
            107: {'level': 'junior_secondary', 'grade': 'grade_8', 'curriculum': 'cbc'},
            108: {'level': 'junior_secondary', 'grade': 'grade_9', 'curriculum': 'cbc'},
            
            # Secondary (8-4-4 System)
            113: {'level': 'secondary', 'grade': 'form_1', 'curriculum': '8-4-4'},
            114: {'level': 'secondary', 'grade': 'form_2', 'curriculum': '8-4-4'},
            115: {'level': 'secondary', 'grade': 'form_3', 'curriculum': '8-4-4'},
            116: {'level': 'secondary', 'grade': 'form_4', 'curriculum': '8-4-4'},
            
            # Special categories
            344: {'level': 'special', 'type': 'accessible_digital_textbook'},
            530: {'level': 'special', 'type': 'child_protection'},
            547: {'level': 'special', 'type': 'parental_resources'}
        }
        
        # Subject keywords for better classification
        self.subject_patterns = {
            'mathematics': ['math', 'algebra', 'geometry', 'calculus', 'arithmetic', 'statistics', 'trigonometry'],
            'english': ['english', 'literature', 'grammar', 'composition', 'reading', 'writing', 'language'],
            'kiswahili': ['kiswahili', 'lugha', 'fasihi', 'utamaduni'],
            'biology': ['biology', 'life science', 'botany', 'zoology', 'ecology', 'genetics'],
            'chemistry': ['chemistry', 'chemical', 'organic', 'inorganic', 'physical chemistry'],
            'physics': ['physics', 'mechanics', 'thermodynamics', 'optics', 'electricity', 'magnetism'],
            'geography': ['geography', 'climate', 'weather', 'map', 'population', 'environment'],
            'history': ['history', 'historical', 'civilization', 'government', 'civics'],
            'business': ['business', 'commerce', 'economics', 'accounting', 'entrepreneurship'],
            'computer': ['computer', 'ict', 'programming', 'technology', 'digital'],
            'agriculture': ['agriculture', 'farming', 'crops', 'livestock', 'soil'],
            'home_science': ['home science', 'nutrition', 'food', 'clothing', 'family'],
            'art': ['art', 'drawing', 'design', 'creative', 'visual'],
            'music': ['music', 'singing', 'instruments', 'rhythm'],
            'physical_education': ['physical education', 'sports', 'games', 'fitness', 'health'],
            'religious_education': ['religious', 'cre', 'ire', 'hre', 'moral']
        }
    
    def setup_selenium_driver(self):
        """Setup Selenium WebDriver with optimal settings"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')  # Speed up loading
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)
    
    def discover_all_urls(self):
        """Discover all possible URLs from KEC ecosystem"""
        logger.info("🔍 Discovering all educational content URLs...")
        
        discovered = set()
        
        # Base category URLs
        for cat_id, info in self.category_mappings.items():
            url = f"https://lms.kec.ac.ke/course/index.php?categoryid={cat_id}"
            discovered.add(url)
        
        # Additional discovery patterns
        additional_patterns = [
            # Course listings
            "https://lms.kec.ac.ke/course/index.php",
            "https://lms.kec.ac.ke/course/search.php",
            
            # Resource portals
            "https://resources.kec.ac.ke/docs/",
            "https://orangebook.kec.ac.ke/",
            
            # Elimika courses
            "https://elimika.kec.ac.ke/",
            
            # OER Portal
            "https://oer.kec.ac.ke/"
        ]
        
        discovered.update(additional_patterns)
        
        # Try to discover more URLs dynamically
        try:
            driver = self.setup_selenium_driver()
            
            for base_url in self.base_urls.values():
                try:
                    logger.info(f"Exploring {base_url} for additional URLs...")
                    driver.get(base_url)
                    time.sleep(2)
                    
                    # Find all links
                    links = driver.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href")
                        if href and any(domain in href for domain in ['kec.ac.ke', 'kicd.ac.ke']):
                            discovered.add(href)
                    
                except Exception as e:
                    logger.warning(f"Error exploring {base_url}: {e}")
                    continue
            
            driver.quit()
            
        except Exception as e:
            logger.warning(f"Selenium discovery failed: {e}")
        
        self.discovered_urls = discovered
        logger.info(f"✅ Discovered {len(discovered)} URLs to scrape")
        return discovered
    
    def extract_course_content(self, url):
        """Extract detailed content from course pages"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract course information
            course_data = {
                'url': url,
                'title': '',
                'description': '',
                'content': '',
                'subjects': [],
                'grade_level': '',
                'curriculum': '',
                'resources': [],
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                course_data['title'] = title_elem.get_text().strip()
            
            # Extract description
            desc_elem = soup.find('div', class_='course-description') or soup.find('meta', attrs={'name': 'description'})
            if desc_elem:
                course_data['description'] = desc_elem.get('content', '') if desc_elem.name == 'meta' else desc_elem.get_text().strip()
            
            # Extract main content
            content_areas = soup.find_all(['div', 'section', 'article'], class_=re.compile(r'content|main|body'))
            all_text = []
            
            for area in content_areas:
                # Remove script and style elements
                for script in area(["script", "style"]):
                    script.decompose()
                text = area.get_text()
                all_text.append(text)
            
            # Also get general page text
            page_text = soup.get_text()
            all_text.append(page_text)
            
            # Clean and combine content
            combined_text = ' '.join(all_text)
            lines = (line.strip() for line in combined_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            course_data['content'] = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract grade level and curriculum from URL and content
            course_data.update(self.classify_content(url, course_data['content'], course_data['title']))
            
            # Find downloadable resources
            resource_links = soup.find_all('a', href=re.compile(r'\.(pdf|doc|docx|ppt|pptx)$'))
            for link in resource_links:
                course_data['resources'].append({
                    'url': urljoin(url, link.get('href')),
                    'title': link.get_text().strip(),
                    'type': link.get('href').split('.')[-1]
                })
            
            return course_data
            
        except Exception as e:
            logger.error(f"Error extracting course content from {url}: {e}")
            return None
    
    def classify_content(self, url, content, title):
        """Classify content by subject, grade level, and curriculum"""
        classification = {
            'subjects': [],
            'grade_level': 'unknown',
            'curriculum': 'unknown',
            'education_level': 'unknown'
        }
        
        # Extract from URL parameters
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        if 'categoryid' in query_params:
            cat_id = int(query_params['categoryid'][0])
            if cat_id in self.category_mappings:
                mapping = self.category_mappings[cat_id]
                classification.update(mapping)
        
        # Classify by content analysis
        text_to_analyze = (title + ' ' + content).lower()
        
        # Subject classification
        for subject, keywords in self.subject_patterns.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                classification['subjects'].append(subject)
        
        # Grade level detection
        grade_patterns = {
            'form_1': ['form 1', 'form one', 'class 9'],
            'form_2': ['form 2', 'form two', 'class 10'],
            'form_3': ['form 3', 'form three', 'class 11'],
            'form_4': ['form 4', 'form four', 'class 12'],
            'grade_1': ['grade 1', 'standard 1', 'class 1'],
            'grade_2': ['grade 2', 'standard 2', 'class 2'],
            'grade_3': ['grade 3', 'standard 3', 'class 3'],
            'grade_4': ['grade 4', 'standard 4', 'class 4'],
            'grade_5': ['grade 5', 'standard 5', 'class 5'],
            'grade_6': ['grade 6', 'standard 6', 'class 6'],
            'grade_7': ['grade 7', 'standard 7', 'class 7'],
            'grade_8': ['grade 8', 'standard 8', 'class 8'],
            'grade_9': ['grade 9', 'standard 9', 'class 9']
        }
        
        for grade, patterns in grade_patterns.items():
            if any(pattern in text_to_analyze for pattern in patterns):
                classification['grade_level'] = grade
                break
        
        # Curriculum detection
        if any(term in text_to_analyze for term in ['cbc', 'competency', 'competence']):
            classification['curriculum'] = 'cbc'
        elif any(term in text_to_analyze for term in ['8-4-4', 'kcse', 'kcpe']):
            classification['curriculum'] = '8-4-4'
        
        return classification
    
    def scrape_single_url(self, url):
        """Scrape a single URL with error handling"""
        try:
            logger.info(f"Scraping: {url}")
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            content = self.extract_course_content(url)
            
            if content and content['content'].strip():
                with self.lock:
                    self.scraped_data.append(content)
                logger.info(f"✅ Successfully scraped: {content['title'][:50]}...")
                return True
            else:
                logger.warning(f"⚠️ No content found at: {url}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {e}")
            with self.lock:
                self.failed_urls.append(url)
            return False
    
    def scrape_all_content(self):
        """Scrape all discovered URLs using multithreading"""
        logger.info("🚀 Starting comprehensive content scraping...")
        
        # Discover all URLs first
        urls = self.discover_all_urls()
        
        # Filter out duplicate and invalid URLs
        valid_urls = []
        for url in urls:
            if url and url.startswith('http') and 'kec.ac.ke' in url:
                valid_urls.append(url)
        
        logger.info(f"📋 Scraping {len(valid_urls)} URLs with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.scrape_single_url, url): url for url in valid_urls}
            
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(valid_urls)} URLs processed")
        
        logger.info(f"🎉 Scraping completed! Collected {len(self.scraped_data)} content items")
        logger.info(f"❌ Failed URLs: {len(self.failed_urls)}")
        
        return self.scraped_data
    
    def save_data(self, filename_prefix='comprehensive_kec_data'):
        """Save scraped data in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        if self.scraped_data:
            df = pd.DataFrame(self.scraped_data)
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            # Save summary statistics
            stats = {
                'total_items': len(self.scraped_data),
                'subjects_found': list(set([subj for item in self.scraped_data for subj in item.get('subjects', [])])),
                'grade_levels_found': list(set([item.get('grade_level', 'unknown') for item in self.scraped_data])),
                'curricula_found': list(set([item.get('curriculum', 'unknown') for item in self.scraped_data])),
                'failed_urls': self.failed_urls,
                'scraping_completed': datetime.now().isoformat()
            }
            
            stats_filename = f"{filename_prefix}_stats_{timestamp}.json"
            with open(stats_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Data saved:")
            logger.info(f"  - JSON: {json_filename}")
            logger.info(f"  - CSV: {csv_filename}")
            logger.info(f"  - Stats: {stats_filename}")
            
            return {
                'json_file': json_filename,
                'csv_file': csv_filename,
                'stats_file': stats_filename,
                'dataframe': df
            }
        
        return None
    
    def generate_curriculum_report(self):
        """Generate comprehensive curriculum analysis report"""
        if not self.scraped_data:
            logger.warning("No data available for curriculum report")
            return None
        
        df = pd.DataFrame(self.scraped_data)
        
        report = {
            'summary': {
                'total_content_items': len(df),
                'unique_subjects': len(set([subj for subjs in df['subjects'] for subj in subjs])),
                'grade_levels_covered': df['grade_level'].nunique(),
                'curricula_types': df['curriculum'].nunique()
            },
            'subject_distribution': {},
            'grade_distribution': df['grade_level'].value_counts().to_dict(),
            'curriculum_distribution': df['curriculum'].value_counts().to_dict(),
            'content_quality_metrics': {
                'avg_content_length': df['content'].str.len().mean(),
                'items_with_resources': len(df[df['resources'].str.len() > 0]),
                'items_with_multiple_subjects': len(df[df['subjects'].str.len() > 1])
            }
        }
        
        # Subject distribution
        all_subjects = [subj for subjs in df['subjects'] for subj in subjs]
        subject_counts = pd.Series(all_subjects).value_counts()
        report['subject_distribution'] = subject_counts.to_dict()
        
        # Content gaps analysis
        expected_grades = ['form_1', 'form_2', 'form_3', 'form_4']
        expected_subjects = ['mathematics', 'english', 'kiswahili', 'biology', 'chemistry', 'physics']
        
        gaps = []
        for grade in expected_grades:
            grade_data = df[df['grade_level'] == grade]
            grade_subjects = set([subj for subjs in grade_data['subjects'] for subj in subjs])
            missing_subjects = set(expected_subjects) - grade_subjects
            if missing_subjects:
                gaps.append({
                    'grade': grade,
                    'missing_subjects': list(missing_subjects)
                })
        
        report['content_gaps'] = gaps
        
        return report

def main():
    """Main execution function for comprehensive scraping"""
    logger.info("🎓 Starting Comprehensive KEC Educational Content Scraping")
    logger.info("=" * 70)
    
    # Initialize scraper
    scraper = ComprehensiveKECScraper(max_workers=3, delay_between_requests=1)
    
    try:
        # Scrape all content
        scraped_data = scraper.scrape_all_content()
        
        if scraped_data:
            # Save data
            saved_files = scraper.save_data()
            
            # Generate curriculum report
            report = scraper.generate_curriculum_report()
            
            if report:
                report_filename = f"curriculum_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📊 Curriculum Analysis Report:")
                logger.info(f"  - Total Content Items: {report['summary']['total_content_items']}")
                logger.info(f"  - Unique Subjects: {report['summary']['unique_subjects']}")
                logger.info(f"  - Grade Levels: {report['summary']['grade_levels_covered']}")
                logger.info(f"  - Report saved: {report_filename}")
            
            logger.info("🎉 Comprehensive scraping completed successfully!")
            return saved_files
        else:
            logger.error("❌ No data was scraped")
            return None
            
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return None

if __name__ == "__main__":
    main()
