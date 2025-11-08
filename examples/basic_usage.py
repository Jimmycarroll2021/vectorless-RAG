"""
Basic usage example for Vectorless RAG.
"""

from vectorless_rag import VectorlessRAG


def main():
    # Initialize the RAG system
    print("Initializing Vectorless RAG system...")
    rag = VectorlessRAG()

    # Sample documents
    documents = [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language known for its simplicity and readability. "
                    "It supports multiple programming paradigms including object-oriented and functional programming.",
            "title": "Introduction to Python"
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn "
                    "and improve from experience without being explicitly programmed. It uses statistical techniques.",
            "title": "Machine Learning Basics"
        },
        {
            "id": "doc3",
            "text": "Natural language processing (NLP) enables computers to understand, interpret, and generate "
                    "human language. It combines linguistics with computer science and AI.",
            "title": "What is NLP?"
        },
        {
            "id": "doc4",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. "
                    "It has revolutionized computer vision, speech recognition, and natural language processing.",
            "title": "Deep Learning Overview"
        },
        {
            "id": "doc5",
            "text": "Data structures are fundamental concepts in computer science that organize and store data efficiently. "
                    "Common data structures include arrays, linked lists, trees, and graphs.",
            "title": "Data Structures Guide"
        },
        {
            "id": "doc6",
            "text": "Algorithms are step-by-step procedures for solving problems or performing tasks. "
                    "Algorithm efficiency is measured using time and space complexity analysis.",
            "title": "Understanding Algorithms"
        },
        {
            "id": "doc7",
            "text": "The Python programming language was created by Guido van Rossum and first released in 1991. "
                    "It emphasizes code readability and allows programmers to express concepts in fewer lines of code.",
            "title": "History of Python"
        },
        {
            "id": "doc8",
            "text": "Neural networks are computing systems inspired by biological neural networks. "
                    "They consist of interconnected nodes (neurons) organized in layers that process information.",
            "title": "Neural Networks Explained"
        },
    ]

    # Index documents
    print(f"\nIndexing {len(documents)} documents...")
    rag.index_documents(documents)

    # Example queries
    queries = [
        "What is NLP?",
        "Tell me about Python programming",
        "How does machine learning work?",
        "Explain neural networks",
    ]

    # Run queries with different methods
    methods = ['bm25', 'tfidf', 'hybrid']

    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Retrieval Method: {method.upper()}")
        print('=' * 80)

        for query in queries:
            print(f"\nQuery: \"{query}\"")
            print("-" * 80)

            # Retrieve documents
            results = rag.retrieve(query, top_k=3, method=method)

            # Display results
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result.get('id')}] {result.get('title', 'No title')}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Text: {result['text'][:150]}...")

    # Show system statistics
    print(f"\n{'=' * 80}")
    print("System Statistics")
    print('=' * 80)
    stats = rag.get_statistics()
    print(f"Documents indexed: {stats['num_documents']}")
    print(f"BM25 vocabulary size: {stats['bm25_stats']['vocab_size']}")
    print(f"Average document length: {stats['bm25_stats']['avg_doc_length']:.2f} tokens")


if __name__ == "__main__":
    main()
