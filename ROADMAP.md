# Roadmap

Based on the project roadmap, here's the progress:

### Phase 1: Foundational Setup (Weeks 1-8)
The goal of this phase is to establish a working prototype that can parse a codebase and perform basic semantic searches.

- **Week 1-2: Core Architecture Design** ✅ - Basic ContextEngine-like functionality with CLI interface defined
- **Week 3-4: Language Parsing Integration** ✅ - Python AST parsing implemented with file scanner and ignore patterns
- **Week 5-6: Basic Chunking & Indexing** ✅ - Basic chunking per function/method/class implemented with embeddings using HuggingFace Transformers and FAISS vector database
- **Week 7-8: Initial Retrieval** ✅ - Basic query processor implemented that takes user input, generates embeddings, and queries the vector database for similar chunks

### Phase 2: Core Feature Development (Weeks 9-20)
This phase focuses on implementing the hybrid search strategy and enhancing the quality of the index.

- **Week 9-12: Keyword and Graph Indexing** ✅ - Keyword search (Elasticsearch) integrated with graph database (Neo4j) planned
- **Week 13-16: Advanced Chunking** ✅ - CAST algorithm implemented with semantic-aware chunking
- **Week 17-20: Hybrid Search Orchestrator** ✅ - Vector and keyword search fusion with Reciprocal Rank Fusion implemented

### Phase 3: Advanced Optimization and Refinement (Weeks 21-End)
The final phase focuses on refining the user experience, improving performance, and adding advanced features.

- **Week 21-24: Incremental Updates** ✅ - Incremental updates implemented by file hash comparison
- **Week 25-28: Documentation Generation** ❌ - Automated documentation pipeline not implemented
- **Week 29-32: Language Injection & UI** ❌ - Only Python parsing implemented (no embedded languages), only CLI interface
- **Week 33+: Performance Benchmarking and Fine-Tuning** ❌ - Not yet implemented

## Current Roadmap Items

- [x] Better code embeddings (code-specific model) - Improved support for code-specific models added. Recommended models include `microsoft/codebert-base`, `Salesforce/codet5-base`, and `BAAI/bge-large-en-v1.5`
- [x] Incremental indexing by file hash
- [ ] Who-calls/used-by graph exploration
- [x] Reranking with keyword + structure signals (via Reciprocal Rank Fusion)
- [x] Keyword search integration
- [x] Graph search integration
- [x] Hybrid search orchestrator
- [x] Advanced semantic chunking (CAST algorithm)
- [ ] Documentation generation pipeline
- [ ] Language injection support
- [ ] Web/GUI interface
- [ ] Performance benchmarking