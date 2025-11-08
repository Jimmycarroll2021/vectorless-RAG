"""
Advanced usage examples for Vectorless RAG.
"""

from vectorless_rag import VectorlessRAG, Preprocessor


def custom_preprocessor_example():
    """Example using a custom preprocessor."""
    print("=" * 80)
    print("Example 1: Custom Preprocessor")
    print("=" * 80)

    # Create custom preprocessor with stemming enabled
    custom_preprocessor = Preprocessor(
        lowercase=True,
        remove_stopwords=True,
        remove_punctuation=True,
        stemming=True,  # Enable stemming
        lemmatization=False,
        min_token_length=3
    )

    # Initialize RAG with custom preprocessor
    rag = VectorlessRAG(preprocessor=custom_preprocessor)

    # Sample documents
    documents = [
        {
            "id": "tech1",
            "text": "Software engineering involves designing, developing, testing, and maintaining software systems.",
            "title": "Software Engineering"
        },
        {
            "id": "tech2",
            "text": "Web development encompasses front-end development, back-end development, and database management.",
            "title": "Web Development"
        },
        {
            "id": "tech3",
            "text": "Cloud computing provides on-demand access to computing resources over the internet.",
            "title": "Cloud Computing"
        },
    ]

    rag.index_documents(documents)

    query = "developing software systems"
    results = rag.retrieve(query, top_k=3)

    print(f"\nQuery: \"{query}\"")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")


def query_expansion_comparison():
    """Compare results with and without query expansion."""
    print("\n" + "=" * 80)
    print("Example 2: Query Expansion Comparison")
    print("=" * 80)

    rag = VectorlessRAG()

    documents = [
        {"id": "1", "text": "The automobile is a wheeled motor vehicle used for transportation.", "title": "Automobiles"},
        {"id": "2", "text": "Cars have revolutionized modern transportation and mobility.", "title": "Modern Cars"},
        {"id": "3", "text": "Bicycles are human-powered vehicles with two wheels.", "title": "Bicycles"},
        {"id": "4", "text": "Public transportation includes buses, trains, and subways.", "title": "Public Transit"},
    ]

    rag.index_documents(documents)

    query = "car"

    print(f"\nQuery: \"{query}\"")
    print("\nWithout query expansion:")
    results_no_expansion = rag.retrieve(query, top_k=3, expand_query=False)
    for i, result in enumerate(results_no_expansion, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")

    print("\nWith query expansion:")
    results_with_expansion = rag.retrieve(query, top_k=3, expand_query=True)
    for i, result in enumerate(results_with_expansion, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")


def reranking_comparison():
    """Compare results with and without reranking."""
    print("\n" + "=" * 80)
    print("Example 3: Reranking Comparison")
    print("=" * 80)

    rag = VectorlessRAG()

    documents = [
        {
            "id": "1",
            "text": "Machine learning algorithms can classify images.",
            "title": "Image Classification"
        },
        {
            "id": "2",
            "text": "Convolutional neural networks excel at image classification tasks in machine learning.",
            "title": "CNNs for Images"
        },
        {
            "id": "3",
            "text": "Data preprocessing is important for machine learning.",
            "title": "Data Preprocessing"
        },
    ]

    rag.index_documents(documents)

    query = "machine learning image classification"

    print(f"\nQuery: \"{query}\"")
    print("\nWithout reranking:")
    results_no_rerank = rag.retrieve(query, top_k=3, rerank=False)
    for i, result in enumerate(results_no_rerank, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")

    print("\nWith reranking:")
    results_with_rerank = rag.retrieve(query, top_k=3, rerank=True)
    for i, result in enumerate(results_with_rerank, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")


def hybrid_retrieval_example():
    """Example using hybrid retrieval."""
    print("\n" + "=" * 80)
    print("Example 4: Hybrid Retrieval")
    print("=" * 80)

    rag = VectorlessRAG()

    documents = [
        {"id": "1", "text": "Quantum computing uses quantum mechanics principles.", "title": "Quantum Computing"},
        {"id": "2", "text": "Classical computers use bits, quantum computers use qubits.", "title": "Bits vs Qubits"},
        {"id": "3", "text": "Quantum algorithms can solve certain problems faster.", "title": "Quantum Algorithms"},
    ]

    rag.index_documents(documents)

    query = "quantum computer principles"

    print(f"\nQuery: \"{query}\"")
    results = rag.retrieve(query, top_k=3, method='hybrid')
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f}) [Method: {result['method']}]")


def main():
    """Run all advanced examples."""
    custom_preprocessor_example()
    query_expansion_comparison()
    reranking_comparison()
    hybrid_retrieval_example()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
