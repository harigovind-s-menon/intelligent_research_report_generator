"""Generate synthetic training data for query classification.

This creates labeled examples for each query type:
- factual: Simple fact lookup
- exploratory: Open-ended research  
- comparative: Compare multiple things
- analytical: Deep analysis required
- current_events: Recent news/events
"""

import json
import random
from pathlib import Path

# Query templates and examples for each type
QUERY_TEMPLATES = {
    "factual": [
        "What is {topic}?",
        "Who is {person}?",
        "When was {event}?",
        "Where is {place}?",
        "How many {thing} are there?",
        "What year did {event} happen?",
        "Who founded {company}?",
        "What is the capital of {country}?",
        "How tall is {structure}?",
        "What does {term} mean?",
        "Define {concept}",
        "Who invented {invention}?",
        "What is the population of {place}?",
        "When did {person} die?",
        "Where was {person} born?",
        "What is the formula for {concept}?",
        "How long is {thing}?",
        "What color is {thing}?",
        "Who wrote {book}?",
        "What language do they speak in {country}?",
    ],
    "exploratory": [
        "What are the implications of {topic}?",
        "How does {topic} work?",
        "What are the different types of {topic}?",
        "Explain {topic} in detail",
        "What should I know about {topic}?",
        "Tell me about {topic}",
        "What are the key aspects of {topic}?",
        "How has {topic} evolved over time?",
        "What are the main concepts in {topic}?",
        "Give me an overview of {topic}",
        "What is the history of {topic}?",
        "How do people use {topic}?",
        "What are the components of {topic}?",
        "Describe {topic}",
        "What makes {topic} important?",
        "What are the fundamentals of {topic}?",
        "How is {topic} structured?",
        "What are the principles behind {topic}?",
        "Explore {topic}",
        "What can you tell me about {topic}?",
    ],
    "comparative": [
        "What is the difference between {thing1} and {thing2}?",
        "{thing1} vs {thing2}",
        "Compare {thing1} and {thing2}",
        "How does {thing1} compare to {thing2}?",
        "{thing1} or {thing2}, which is better?",
        "What are the pros and cons of {thing1} vs {thing2}?",
        "Contrast {thing1} with {thing2}",
        "Should I choose {thing1} or {thing2}?",
        "What are the similarities between {thing1} and {thing2}?",
        "How do {thing1} and {thing2} differ?",
        "{thing1} versus {thing2}",
        "Which is better: {thing1} or {thing2}?",
        "Compare and contrast {thing1} and {thing2}",
        "What distinguishes {thing1} from {thing2}?",
        "Differences between {thing1} and {thing2}",
        "{thing1} compared to {thing2}",
        "Advantages of {thing1} over {thing2}",
        "How is {thing1} different from {thing2}?",
        "{thing1} and {thing2} comparison",
        "Weigh {thing1} against {thing2}",
    ],
    "analytical": [
        "Why did {event} happen?",
        "What caused {event}?",
        "What are the reasons for {topic}?",
        "Analyze {topic}",
        "What factors contribute to {topic}?",
        "What is the impact of {topic}?",
        "How does {topic} affect {thing}?",
        "What are the consequences of {topic}?",
        "Why is {topic} important?",
        "What drives {topic}?",
        "Explain why {topic} occurs",
        "What are the underlying causes of {topic}?",
        "Break down {topic}",
        "What influences {topic}?",
        "Examine {topic}",
        "What are the effects of {topic}?",
        "Why does {topic} matter?",
        "What role does {topic} play in {thing}?",
        "Investigate {topic}",
        "What are the root causes of {topic}?",
    ],
    "current_events": [
        "What is the latest news on {topic}?",
        "What are the recent developments in {topic}?",
        "What happened with {topic} today?",
        "Current state of {topic}",
        "What's new in {topic}?",
        "Recent updates on {topic}",
        "What is happening with {topic}?",
        "Latest {topic} news",
        "What are the newest trends in {topic}?",
        "Current {topic} situation",
        "What's going on with {topic}?",
        "Recent {topic} announcements",
        "Today's {topic} updates",
        "What are the latest findings on {topic}?",
        "Breaking news about {topic}",
        "What just happened in {topic}?",
        "Recent breakthroughs in {topic}",
        "This week in {topic}",
        "What are people saying about {topic} now?",
        "Latest research on {topic}",
    ],
}

# Fill-in values for templates
TOPICS = [
    "artificial intelligence", "machine learning", "climate change", "quantum computing",
    "blockchain", "electric vehicles", "renewable energy", "space exploration",
    "gene therapy", "cryptocurrency", "cybersecurity", "autonomous vehicles",
    "virtual reality", "5G technology", "biotechnology", "nanotechnology",
    "robotics", "cloud computing", "data science", "neural networks",
    "deep learning", "natural language processing", "computer vision", "IoT",
    "edge computing", "sustainable energy", "fusion power", "CRISPR",
    "mRNA vaccines", "telemedicine", "fintech", "agritech",
]

PEOPLE = [
    "Elon Musk", "Jeff Bezos", "Albert Einstein", "Marie Curie",
    "Steve Jobs", "Bill Gates", "Ada Lovelace", "Alan Turing",
    "Nikola Tesla", "Isaac Newton", "Charles Darwin", "Galileo",
    "Leonardo da Vinci", "Aristotle", "Plato", "Socrates",
]

COMPANIES = [
    "Google", "Apple", "Microsoft", "Amazon", "Tesla", "OpenAI",
    "Meta", "Netflix", "SpaceX", "Nvidia", "Intel", "IBM",
]

COUNTRIES = [
    "Japan", "Germany", "France", "Brazil", "India", "China",
    "Australia", "Canada", "Mexico", "Italy", "Spain", "Korea",
]

THINGS_PAIRS = [
    ("Python", "JavaScript"), ("React", "Vue"), ("AWS", "Azure"),
    ("iPhone", "Android"), ("Mac", "Windows"), ("Tesla", "BMW"),
    ("Netflix", "Disney+"), ("Spotify", "Apple Music"),
    ("PostgreSQL", "MySQL"), ("Docker", "Kubernetes"),
    ("TensorFlow", "PyTorch"), ("REST", "GraphQL"),
    ("agile", "waterfall"), ("monolith", "microservices"),
    ("SQL", "NoSQL"), ("Linux", "Windows"), ("vim", "emacs"),
    ("tabs", "spaces"), ("coffee", "tea"), ("cats", "dogs"),
]

EVENTS = [
    "the industrial revolution", "World War II", "the moon landing",
    "the invention of the internet", "the financial crisis",
    "the pandemic", "the AI boom", "the smartphone revolution",
]


def fill_template(template: str, query_type: str) -> str:
    """Fill a template with random values."""
    result = template
    
    if "{topic}" in result:
        result = result.replace("{topic}", random.choice(TOPICS))
    if "{person}" in result:
        result = result.replace("{person}", random.choice(PEOPLE))
    if "{company}" in result:
        result = result.replace("{company}", random.choice(COMPANIES))
    if "{country}" in result:
        result = result.replace("{country}", random.choice(COUNTRIES))
    if "{place}" in result:
        result = result.replace("{place}", random.choice(COUNTRIES + ["New York", "Tokyo", "London", "Paris"]))
    if "{event}" in result:
        result = result.replace("{event}", random.choice(EVENTS))
    if "{thing}" in result:
        result = result.replace("{thing}", random.choice(TOPICS))
    if "{concept}" in result:
        result = result.replace("{concept}", random.choice(TOPICS))
    if "{term}" in result:
        result = result.replace("{term}", random.choice(TOPICS))
    if "{invention}" in result:
        result = result.replace("{invention}", random.choice(["the telephone", "electricity", "the computer", "the airplane"]))
    if "{structure}" in result:
        result = result.replace("{structure}", random.choice(["the Eiffel Tower", "Mount Everest", "the Empire State Building"]))
    if "{book}" in result:
        result = result.replace("{book}", random.choice(["1984", "Pride and Prejudice", "The Great Gatsby"]))
    
    if "{thing1}" in result and "{thing2}" in result:
        pair = random.choice(THINGS_PAIRS)
        result = result.replace("{thing1}", pair[0]).replace("{thing2}", pair[1])
    
    return result


def generate_dataset(samples_per_class: int = 100) -> list[dict]:
    """Generate a labeled dataset."""
    data = []
    
    for query_type, templates in QUERY_TEMPLATES.items():
        for _ in range(samples_per_class):
            template = random.choice(templates)
            query = fill_template(template, query_type)
            data.append({
                "query": query,
                "label": query_type,
            })
    
    random.shuffle(data)
    return data


def main():
    """Generate and save training data."""
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate training data
    train_data = generate_dataset(samples_per_class=200)
    
    # Generate test data with different seed
    random.seed(123)
    test_data = generate_dataset(samples_per_class=50)
    
    # Save to files
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    train_path = data_dir / "query_classification_train.json"
    test_path = data_dir / "query_classification_test.json"
    
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(test_data)} test samples")
    print(f"Saved to {train_path} and {test_path}")
    
    # Print distribution
    print("\nLabel distribution (train):")
    from collections import Counter
    counts = Counter(d["label"] for d in train_data)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
