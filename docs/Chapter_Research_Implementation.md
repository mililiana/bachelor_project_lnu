Introduction
This chapter provides a comprehensive description of the research implementation, detailing all activities undertaken to collect, process, and structure the data necessary for answering the research questions. The research involved a controlled experiment comparing two retrieval-augmented generation (RAG) techniques for a university information system. This chapter describes the data collection methodology, data processing pipeline, system designs, and the experimental setup used to evaluate the two techniques.
3.1 Data Collection Methodology
3.1.1 Overview
The foundation of this research is a comprehensive knowledge base of university-related documents from the National University "Lviv Polytechnic" (NULP). The data collection process involved systematically gathering documents from various university sources, including official PDF documents, web pages, and structured data files. The primary method employed was web scraping and document extraction from PDF files.
3.1.2 Document Sources
The data collection targeted multiple categories of university information:
1. **Institutional Information**: Documents describing institutes, departments, and administrative structures
2. **Regulatory Documents**: University regulations, codes of conduct, and policy documents
3. **Student Services**: Information about scholarships, student cards(Leocard), dining facilities.
4. **Academic Information**: Educational programs, double degree programs, Erasmus+ opportunities
5. **Infrastructure**: Building addresses, locations of facilities, and campus information
6. **Administrative Procedures**: Guidelines for various student processes
totaly collected 50+ documents included:
- PDF files from the official university website and internal systems
- Excel spreadsheets containing structured data about programs, scholarships, and partnerships
- Text documents with processed information

3.1.3 Web Scraping and Document Extraction Implementation
The data collection was implemented using a Python-based pipeline that processed documents. The implementation consisted of several key components:
#### PDF Text Extraction
A dedicated PDF parser (`pdf_parser.py`) was developed using the `pdfminer` library to extract text content from PDF documents. The extraction process included:
1. **Text Extraction**: Using `pdfminer.high_level.extract_text()` to extract raw text from PDF files
2. **Text Cleaning**: The extracted text underwent cleaning to:
- Remove form feed characters (`\f`) and replace them with spaces
- Normalize whitespace (multiple spaces collapsed to single spaces)
- Remove excessive punctuation spacing
- Preserve the original text structure and formatting

The PDF parser processed files from the input directory (`data_bachelor`) and saved cleaned text files to the output directory. Each processed file was saved with a `_processed.txt` suffix.

### 3.1.4 Data Organization
All collected documents were organized into a structured format with the following metadata fields:
-`doc_id`: Unique identifier for each document
-`source_url`: Original source URL (when available)
-`category`: Document category ("Інститути", "Інструкції", "Інформаційні матеріали", "Адреси", "Події", "Положення")
-`title`: Document title
-`content`: Full text content of the document

## 3.2 Data Processing and Preparation
### 3.2.1 Text Cleaning Pipeline
After initial extraction, documents underwent a comprehensive cleaning process implemented. The cleaning pipeline applied several transformations:
1. **HTML Tag Removal**: Removed any HTML tags that might have been present in extracted text
2. **Line Filtering**: Removed lines that were:
- Entirely uppercase with minimal content (likely headers or formatting artifacts)
- Short lines containing only uppercase letters, numbers, and dashes
- Signature lines or footer text patterns
3. **Whitespace Normalization**: 
- Collapsed multiple spaces and tabs to single spaces
- Normalized multiple newlines to double newlines (paragraph breaks)
4. **Minimum Length Filtering**: Documents shorter than a specified minimum length (5 characters) were excluded
The cleaning function `clean_text_for_miniLM()` was specifically designed to prepare text for embedding models, ensuring optimal quality for semantic search.

### 3.2.2 Document Chunking Strategy
Given that many documents exceeded the optimal length for embedding models, a semantic chunking strategy was implemented. The chunking process used the following approach:

#### Semantic Chunking Algorithm
The chunking algorithm split documents at sentence boundaries rather than arbitrary character positions, preserving semantic coherence:
1. **Sentence Splitting**: Documents were split at sentence boundaries using the pattern `(?<=[.!?])\s+`
2. **Chunk Size Management**: 
- Target chunk size: 128 characters
- Overlap between chunks: 30 characters
3. **Smart Overlap**: The overlap mechanism worked backwards from the end of each chunk, including complete sentences to maintain context continuity
4. **Metadata Preservation**: Each chunk retained:
   - Parent document ID
   - Chunk index and total chunk count
   - Original document metadata (category, title, source URL)
   - Full context information

The chunking process ensured that related information remained together while keeping chunks within optimal size limits for embedding generation.

### 3.2.3 Metadata Structure

Each processed document chunk contained comprehensive metadata:
```json
{
  "doc_id": "doc_001_chunk_001",
  "parent_doc_id": "doc_001",
  "source_url": "",
  "category": "Інститути",
  "title": "Інститут комп'ютерних наук та інформаційних технологій",
  "chunk_index": 1,
  "total_chunks": 3,
  "content": "..."
}
```

This metadata structure enabled:
- Tracking document provenance
- Category-based filtering during retrieval
- Context reconstruction from chunks
- Quality assessment and debugging

### 3.2.4 Embedding Generation
After chunking, document embeddings were generated using multilingual sentence transformer models. The embedding pipeline performed the following steps:

1. **Model Selection**: implementation used `paraphrase-multilingual-mpnet-base-v2` for improved semantic understanding, as this is multilingual model meaning i can embed ukrainian language, but the input of this model is not high(just 128 tokens)
2. **Embedding Generation**: 
- All document chunks were encoded into dense vector representations
- Embeddings were generated with progress tracking for large batches
3. **Vector Database Storage**: 
- Embeddings were stored in ChromaDB, a persistent vector database
- Used cosine similarity as the distance metric
- Metadata was stored alongside embeddings for filtering and retrieval
The embedding process created a searchable knowledge base where semantic similarity could be computed efficiently.

### 3.2.5 Search Strategy Configurations

During the development and experimentation phase, two distinct hybrid search configurations were implemented and evaluated. These configurations represent different approaches to combining semantic similarity search with keyword-based filtering, each optimized for different types of queries and use cases.

#### Configuration A: Similarity First → Keyword Filtering (Narrowing Down Concepts)

**Implementation**: `pipline1/hybrid_search.py`

This configuration prioritizes semantic understanding and then applies keyword-based refinement. The search process follows a two-stage approach:

1. **Stage 1: Semantic Search**
   - Query embedding is generated using the multilingual sentence transformer model
   - Vector similarity search is performed in ChromaDB to retrieve semantically similar documents
   - Retrieves up to a configurable maximum (default: 100 documents)
   - Applies metadata filters (`where`) and content filters (`where_document`) during retrieval
   - Implements progressive fallback strategy if filters are too restrictive:
     - First attempts search with all filters
     - Falls back to metadata-only filtering if no results
     - Finally attempts pure semantic search without filters

2. **Stage 2: Keyword Boosting**
   - All documents retrieved from semantic search are scored for keyword matches
   - Keyword boost calculation:
     - Title matches: +0.5 per keyword found in document title
     - Content matches: +0.1 per keyword found in document content
   - Combined score: `combined_score = semantic_score + keyword_boost`
   - Results sorted by combined score (semantic similarity + keyword boost)

**Use Case**: This configuration is optimal for queries where:
- The user's intent is conceptual or requires semantic understanding
- Keywords serve to refine and prioritize already semantically relevant results
- The goal is to narrow down a broad set of conceptually similar documents

**Example**: A query like "How do students connect to university Wi-Fi?" benefits from semantic search finding documents about network connectivity, then keyword matching ("Wi-Fi", "network") helps prioritize the most relevant sections.

#### Configuration B: Keyword First → Similarity Filtering (Contextualizing Entities)

**Implementation**: `pipline2/hybrid_search2.py`

This configuration prioritizes exact keyword matching and then applies semantic ranking for contextualization. The search process is reversed:

1. **Stage 1: Keyword-Based Retrieval**
   - Keywords are used to filter documents using ChromaDB's `$contains` operator
   - Creates OR conditions for multiple keywords (matches any keyword in content)
   - Retrieves documents that contain the specified keywords
   - Applies metadata filters and keyword filters simultaneously
   - Implements progressive fallback:
     - First attempts with keyword filter + metadata filters
     - Falls back to metadata-only if no keyword matches
     - Finally attempts without any filters
   - Optional parameter `require_keywords`: If enabled, only returns documents with keyword matches

2. **Stage 2: Semantic Ranking**
   - All keyword-matched documents are re-ranked using semantic similarity
   - Query embedding is computed and compared with each document's embedding
   - Cosine similarity is calculated and normalized to [0, 1] range
   - Keyword score and semantic score are combined: `combined_score = keyword_score + semantic_score`
   - Results sorted by combined score

**Use Case**: This configuration is optimal for queries where:
- The user is searching for specific entities, names, or exact terms
- Keywords are highly discriminative (e.g., building numbers, specific institute names)
- The goal is to find documents containing specific entities, then rank them by contextual relevance

**Example**: A query like "Where is building 19?" benefits from keyword-first search finding all documents mentioning "19" or "корпус", then semantic ranking helps identify which mentions are most contextually relevant to location queries.

#### Enhanced Keyword Boosting Implementation

**Implementation**: `pipline_updated1_semantic_keywords/improved_hybrid_search.py`

Building upon the basic keyword boosting mechanisms, an enhanced version was developed with sophisticated keyword matching algorithms:

1. **Word Boundary Matching**: 
   - Uses regex word boundaries (`\b`) to prevent false positives
   - Example: "кафе" (café) will not match "кафедри" (departments)
   - Ensures exact word matches rather than substring matches

2. **Position-Aware Boosting**:
   - **Title matches**: +0.5 per keyword (strong signal of relevance)
   - **Early content matches**: +1.5 if keyword appears in first 200 characters
   - **Regular content matches**: +1.0 for matches elsewhere in content
   - Early appearance indicates higher topical relevance

3. **Partial Matching for Compound Keywords**:
   - For multi-word keywords, partial matches contribute proportionally
   - Example: "дуальна освіта" (dual education) partially matches if both words found separately
   - Partial match score: `0.2 * (matched_parts / total_parts)`

4. **Boost Capping**:
   - Maximum keyword boost capped at 3.0 to prevent over-weighting
   - Ensures semantic score remains the primary ranking signal

5. **Query-Type Adaptive Boosting**:
   - For list queries, keyword boost is multiplied by 1.5
   - Prioritizes exact matches when enumeration is required

This enhanced implementation provides more nuanced keyword matching while maintaining the balance between semantic understanding and exact term matching.

#### Prompt Engineering for Query Analysis

**Implementation**: `pipline_updated1_semantic_keywords/prompt/build_prompt_enhanced.py` and `system_prompt_enhanced_v3.txt`

A general-purpose, dynamically constructed prompt system was developed to guide the LLM in query analysis. Unlike hardcoded prompts, this system:

1. **Dynamic Metadata Integration**:
   - Automatically extracts available categories from the vector database
   - Retrieves all document titles from the knowledge base
   - Injects this information into the prompt template
   - Enables the LLM to make informed decisions about filters using actual database contents

2. **General Query Processing Rules**:
   - **Explicit Category Queries**: When user explicitly mentions a category
   - **Specific Entity Queries**: When user mentions exact document names or entities
   - **Entity in Category**: Combination of category and entity filters
   - **Broad Search**: Default mode for general questions (null filters)

3. **Query Type Classification**:
   - **Single**: Queries expecting one specific answer (addresses, names, definitions)
   - **List**: Queries requiring enumeration of multiple items
   - **Count**: Queries asking for numerical quantities

4. **Keyword Extraction Guidelines**:
   - Emphasizes specificity: excludes common stop words that appear in all documents
   - Includes synonyms and official terminology
   - Special rules for common query types (e.g., adding "Путівник" for general student life queries)
   - Critical rules for specific domains (e.g., "їдальня" and "харчування" for dining queries)

5. **Whitelist-Based Filtering**:
   - LLM can only use categories and titles that actually exist in the database
   - Prevents hallucination of non-existent filters
   - Ensures filter accuracy and system reliability

The general nature of this prompt system makes it adaptable to different knowledge bases without requiring domain-specific modifications, as it dynamically learns the available categories and entities from the database metadata.

## 3.3 System Design and Implementation
### 3.3.1 Architecture Overview
The research compared two RAG (Retrieval-Augmented Generation) system architectures:
1. **Technique 1: Basic Semantic Search RAG**
2. **Technique 2: Enhanced Hybrid Search RAG with Keyword Boosting**

Both systems followed a three-stage pipeline:
1. **Query Analysis**: Understanding and structuring user queries
2. **Document Retrieval**: Finding relevant documents from the knowledge base
3. **Answer Generation**: Synthesizing answers from retrieved documents

### 3.3.2 Technique 1: Basic Semantic Search RAG

#### 3.3.2.1 Query Analysis

The query analysis component (`llm1.py`) employed Google's Gemini Flash model (`models/gemini-flash-latest`) to analyze user queries and extract structured information for the retrieval pipeline.

**LLM Model and Configuration**:
- **Model**: Google Gemini Flash (latest version)
- **API Integration**: Google Generative AI SDK (`google.generativeai`)
- **Temperature**: 0.0 (deterministic outputs for consistency)
- **Response Format**: JSON (`response_mime_type: "application/json"`)
- **System Instruction**: Pre-defined prompt template loaded from `system_prompt_base.txt`

**LLM Interaction Flow**:

1. **Initialization**:
   - System prompt is loaded and configured as a persistent system instruction
   - The prompt contains:
     - Available categories whitelist (dynamically populated from database metadata)
     - Available document titles whitelist (dynamically populated from database metadata)
     - Filter generation rules
     - Keyword extraction guidelines
     - Prohibited patterns (e.g., no conversational verbs in filters)

2. **Query Processing**:
   - User query is sent directly to the LLM via `model.generate_content(query)`
   - The system instruction (prompt) is automatically prepended to the query
   - No additional user messages or conversation history is maintained
   - Single-shot inference (no multi-turn conversation)

3. **Response Parsing**:
   - LLM returns a JSON-formatted response containing:
     ```json
     {
       "filters": {
         "where": {...} or null,
         "where_document": {...} or null
       },
       "keywords": ["keyword1", "keyword2", ...]
     }
     ```
   - Response is parsed using `json.loads()` with error handling
   - Validation ensures required fields exist

**Keyword Generation Process**:

The LLM generates keywords through natural language understanding of the query:

1. **Extraction Strategy**: The LLM analyzes the query to identify:
   - **Entities**: Specific names, places, organizations (e.g., "ІКНІ", "19 корпус")
   - **Key Terms**: Important domain-specific terms (e.g., "стипендія", "дуальна освіта")
   - **Concepts**: Core concepts relevant to the information need

2. **Keyword Selection Guidelines** (from system prompt):
   - Extract entities, terms, and important words from the query
   - Focus on specific, discriminative terms
   - Avoid overly generic words that appear in all documents

3. **Validation and Fallback**:
   - If `keywords` field is missing or not a list, defaults to `[query]` (using entire query as keyword)
   - Ensures at least one keyword is always available for boosting
   - Keywords are passed as-is to the search engine (case-sensitive matching handled in search)

4. **Output Format**:
   - Keywords returned as a Python list of strings
   - No preprocessing or normalization applied (preserves original form from query)
   - Used directly in keyword matching algorithms

**Filter Generation**:

The LLM generates ChromaDB filter specifications based on query analysis:

1. **Filter Types**:
   - `where`: Metadata filters (e.g., `{"category": {"$eq": "Інститути"}}`)
   - `where_document`: Content filters using `$contains` operator (e.g., `{"$contains": "ІКНІ"}`)

2. **Filter Rules** (from system prompt):
   - **General Category**: Use `where` filter when query explicitly mentions a category
   - **Specific Entity**: Use `where_document` when query mentions exact document names from titles whitelist
   - **Entity in Category**: Use both filters when query specifies both category and entity
   - **No Filters**: Use `null` for broad queries where category is uncertain

3. **Validation**:
   - If `filters` field is missing, defaults to `None` (no filtering)
   - Filters are validated against whitelist of available categories and titles
   - Prevents hallucination of non-existent categories or entities

**Error Handling**:
- JSON parsing errors: Logged with raw response for debugging, defaults to safe fallback values
- API errors: Exception handling with logging, returns default structure
- Invalid filter structures: Handled gracefully with fallback to no filters

#### 3.3.2.2 Document Retrieval

The retrieval engine (`vector_search_engine.py`) implemented a straightforward semantic search:

1. **Semantic Search**: 
   - Query embedding generation using the same model as document embeddings
   - Vector similarity search in ChromaDB
   - Retrieval of top-k documents (default: 5, with 3x expansion to 20 for initial retrieval)
2. **Scoring**: 
   - Documents ranked solely by semantic similarity score
   - No additional boosting or re-ranking applied
3. **Result Formatting**: 
   - Results included document metadata (title, category, content)
   - Semantic similarity scores were normalized (1 - distance)

This approach relied entirely on semantic similarity without keyword-based enhancements.

#### 3.3.2.3 Answer Generation

The answer generation component (`complete_rag_system.py`) used Gemini Flash to synthesize answers:

1. **Context Preparation**: All retrieved documents were formatted with titles and content
2. **Prompt Engineering**: A structured prompt instructed the model to:
   - Answer only based on provided contexts
   - Be concise but complete
   - Use exact information from documents
   - Indicate when information is insufficient
3. **Response Generation**: Temperature set to 0 based on stability experiments (see Section 3.4.5.1)

### 3.3.3 Technique 2: Enhanced Hybrid Search RAG

#### 3.3.3.1 Enhanced Query Analysis

The enhanced query analyzer (`llm1_enhanced.py`) extended the basic analyzer with advanced query understanding capabilities, using the same Google Gemini Flash model with an enhanced prompt system.

**LLM Model and Configuration**:
- **Model**: Google Gemini Flash (`models/gemini-flash-latest`)
- **API Integration**: Google Generative AI SDK with safety settings configured
- **Temperature**: 0.0 (deterministic outputs)
- **Response Format**: JSON (`response_mime_type: "application/json"`)
- **System Instruction**: Dynamically constructed from `system_prompt_enhanced_v3.txt` template
- **Safety Settings**: All safety filters disabled (`BLOCK_NONE`) to prevent false positives on educational content

**Dynamic Prompt Construction**:

The system prompt is built dynamically at initialization using `build_prompt_enhanced.py`:

1. **Metadata Extraction**:
   - Reads metadata cache from `vector_db_metadata_cache.json`
   - Extracts available categories (sorted list)
   - Extracts available document titles (sorted list)

2. **Template Population**:
   - Base template (`system_prompt_enhanced_v3.txt`) contains placeholders: `{categories}` and `{titles}`
   - Categories formatted as bullet list: `- {category_name}`
   - Titles formatted as bullet list: `- {title_name}`
   - Final prompt assembled via string formatting

3. **Benefits of Dynamic Prompting**:
   - LLM has access to actual database contents (not hardcoded lists)
   - Prevents hallucination of non-existent categories/titles
   - Adapts automatically to database changes
   - General-purpose design (works with any knowledge base structure)

**LLM Interaction Flow**:

1. **Initialization**:
   - Enhanced system prompt is built dynamically
   - Prompt includes:
     - Available categories whitelist (from database)
     - Available titles whitelist (from database)
     - Query processing rules (general, not domain-specific)
     - Query type classification guidelines
     - Enhanced keyword extraction rules

2. **Query Processing**:
   - User query sent to LLM via `model.generate_content(query)`
   - System instruction automatically prepended
   - Single-shot inference (no conversation context)

3. **Enhanced Response Structure**:
   ```json
   {
     "filters": {
       "where": {...} or null,
       "where_document": {...} or null
     },
     "keywords": ["keyword1", "keyword2", ...],
     "expected_answer_type": "single" | "list" | "count",
     "explanation": "brief reasoning description"
   }
   ```

**Query Type Classification**:

The LLM classifies queries into three types based on expected answer format:

1. **"single"**: Queries expecting one specific answer
   - Examples: addresses, names, definitions, dates
   - LLM identifies through patterns: "Де знаходиться...", "Хто є...", "Що таке..."

2. **"list"**: Queries requiring enumeration
   - Examples: "Які факультети...", "Перелічи кафедри..."
   - LLM identifies through plural forms and enumeration keywords

3. **"count"**: Queries asking for numerical quantities
   - Examples: "Скільки кафедр...", "Яка кількість..."
   - Distinguished from "single" when asking about groups vs. single values

**Enhanced Keyword Generation**:

The enhanced system includes sophisticated keyword extraction rules:

1. **Keyword Selection Strategy**:
   - **Specificity Focus**: Excludes common stop words that appear in all documents
     - Avoids: "університет", "Львівська політехніка", "студент", "територія"
     - Includes: Domain-specific terms, entity names, unique identifiers
   - **Synonym Inclusion**: Adds official terminology that might appear in documents
   - **Critical Rules**: Special handling for common query types:
     - General student life queries: Automatically adds "Путівник" keyword
     - Dining queries: Automatically adds "їдальня" and "харчування"
     - These rules address common information location patterns

2. **Keyword Quality Guidelines** (from prompt):
   - Extract 3-5 key words (optimal number for boosting)
   - Focus on words that uniquely identify the topic
   - Include synonyms and official terms
   - Avoid generic words that match all documents

3. **Validation and Fallback**:
   - If `keywords` missing or invalid: Defaults to `[query]`
   - Ensures keywords list is always non-empty
   - Keywords passed directly to search engine

**Filter Generation (Enhanced)**:

Enhanced filter generation with more sophisticated rules:

1. **Four Filter Modes**:
   - **Explicit Category**: When user explicitly mentions category name
   - **Specific Entity**: When query mentions exact document/title from whitelist
   - **Entity in Category**: Combination of both filters
   - **Broad Search**: Null filters for general queries (prioritized for student life questions)

2. **Whitelist Enforcement**:
   - LLM can only use categories from the provided whitelist
   - LLM can only use titles from the provided whitelist
   - Prevents hallucination of non-existent filters
   - Ensures filter accuracy

3. **Validation**:
   - `expected_answer_type` validated against `["single", "list", "count"]`
   - Invalid types default to "single"
   - Missing fields handled with safe defaults

**Error Handling**:

Comprehensive error handling ensures system robustness:

1. **JSON Parsing Errors**:
   - Catches `JSONDecodeError` exceptions
   - Logs raw response for debugging
   - Returns safe default structure with error explanation

2. **API Errors**:
   - Catches general exceptions
   - Returns default structure with error message
   - System continues operation with fallback values

3. **Field Validation**:
   - All required fields checked and defaulted if missing
   - Type validation for `expected_answer_type`
   - Ensures consistent output structure

**Output Integration**:

The LLM analysis output is directly integrated into the search pipeline:

- **Filters**: Passed to ChromaDB query operations
- **Keywords**: Used in keyword boosting algorithms
- **Query Type**: Used for adaptive context selection and prompt adaptation
- **Explanation**: Logged for debugging and analysis (not used in search)

#### 3.3.3.2 Hybrid Search Engine

The hybrid search engine (`improved_hybrid_search.py`) implemented a sophisticated multi-stage retrieval process:
##### Stage 1: Semantic Search with Progressive Fallback
1. **Initial Semantic Search**: 
   - Query embedding generation
   - Retrieval of up to 300 candidate documents (configurable)
   - Application of metadata and content filters
2. **Progressive Fallback Strategy**: If initial search returned no results:
   - **Fallback 1**: Retry without `where_document` filter (keeping metadata filters)
   - **Fallback 2**: Retry with semantic search only (no filters)
   - This ensured robust retrieval even with overly restrictive filters
3. **Keyword Expansion**: 
   - For each extracted keyword, additional documents were retrieved using `$contains` operator
   - Word boundary matching prevented false positives (e.g., "кафе" not matching "кафедри")
   - Expanded results merged with semantic search results

##### Stage 2: Enhanced Keyword Boosting and Scoring

1. **Keyword Boost Calculation**: 
   - **Title Boost**: Keywords found in document titles received higher weight (0.5 per match)
   - **Content Boost**: Keywords in content received variable weight:
     - 1.5 if found in first 200 characters (early appearance indicates relevance)
     - 1.0 for matches elsewhere
   - **Partial Matching**: For multi-word keywords, partial matches contributed proportionally
   - **Boost Capping**: Maximum boost capped at 3.0 to prevent over-weighting
2. **Combined Scoring**: 
   - `combined_score = semantic_score + keyword_boost`
   - For list queries, keyword boost multiplied by 1.5 to prioritize keyword matches
3. **Relevance Filtering**: 
   - Documents with semantic score below threshold (default: 0.3) were filtered out
   - This removed low-relevance documents early

##### Stage 3: Adaptive Context Selection

1. **Query-Type Adaptive Selection**: 
   - **Single queries**: 5 documents selected
   - **List queries**: 15 documents selected (more context needed for enumeration)
   - **Count queries**: 10 documents selected
2. **Diversity-Aware Selection**: 
   - Preferred documents from different categories
   - Avoided redundant information
   - Diversity threshold: 0.7 for single queries, 0.5 for list queries
3. **Special Handling for List Queries**: 
   - All keyword-matched documents included (up to 20)
   - Ensured comprehensive coverage for enumeration tasks

#### 3.3.3.3 Enhanced Answer Generation

The answer generation (`improved_complete_rag_system.py`) included:

1. **Adaptive Prompting**: 
   - Different instructions for list vs. single queries
   - List queries: Emphasis on completeness and structured presentation
   - Single queries: Emphasis on conciseness and focus
2. **Context Annotation**: 
   - High-relevance documents (keyword_boost > 0.5) marked as "високорелевантний"
   - Provided additional signal to the LLM about document importance
3. **Metadata Integration**: 
   - Query type information included in prompt
   - Number of retrieved documents provided for context

### 3.3.4 Key Differences Between Techniques

| Aspect | Technique 1 (Basic) | Technique 2 (Enhanced) |
|--------|---------------------|----------------------|
| **Query Analysis** | Basic filter/keyword extraction | Query type classification + enhanced filtering |
| **Retrieval** | Pure semantic search | Semantic + keyword expansion + progressive fallback |
| **Scoring** | Semantic similarity only | Semantic + keyword boosting with adaptive weights |
| **Context Selection** | Fixed top-k (5 documents) | Adaptive (5-15 documents) based on query type |
| **Diversity** | Not considered | Category-based diversity filtering |
| **Answer Generation** | Standard prompt | Query-type adaptive prompting |

## 3.4 Experimental Design

### 3.4.1 Controlled Experiment Setup

The research employed a controlled experiment design to compare the two RAG techniques. The experiment was designed to evaluate:

1. **Retrieval Quality**: Accuracy and relevance of retrieved documents
2. **Answer Quality**: Correctness, completeness, and relevance of generated answers
3. **System Robustness**: Performance across different query types

### 3.4.2 Evaluation Dataset

A comprehensive evaluation dataset was created containing 30 test queries covering 6 categories:
1. **Navigation and Infrastructure**
2. **The educational process and academic rules**
3. **Scholarships**
4. **Student Services, Events and Organisations**
5. **Structure and Institutions**: Questions about university regulations and policies

The queries were stored in JSON format with the following structure:

```json
{
  "id": "1",
  "content": "Where is building №19?"
}
```

### 3.4.3 Experimental Procedure
The experimental procedure followed these steps:

1. **System Initialization**: 
   - Both systems initialized with the same knowledge base
   - Same embedding model and vector database
   - Same LLM models for query analysis and answer generation

2. **Query Processing**: 
   - Each query processed independently through both systems
   - Results saved with full metadata for analysis

3. **Data Collection**: 
   - For each query, the following data was collected:
     - Retrieved documents with scores
     - Generated answer
     - Query analysis details (filters, keywords, query type)
     - Retrieval statistics (number retrieved, filtered, selected)
     - Fallback strategy used (if any)

4. **Rate Limiting**: 
   - API calls to Gemini were rate-limited (10-second delays between queries)
   - Error handling for rate limit exceptions with exponential backoff

5. **Result Storage**: 
   - Results saved in JSON format for both systems
   - Separate result files for each technique
   - Compatible format for evaluation tools

### 3.4.4 Evaluation Metrics
The experiment collected data for multiple evaluation dimensions:
1. **Retrieval Metrics**:
   - Number of documents retrieved
   - Number of documents after filtering
   - Number of documents used in context
   - Semantic similarity scores
   - Keyword boost scores
   - Combined relevance scores
2. **Answer Quality Metrics** (collected for later analysis):
   - Answer correctness
   - Answer completeness
   - Answer relevance
   - Factual accuracy
3. **System Behavior Metrics**:
   - Query type classification accuracy
   - Filter application success rate
   - Fallback strategy usage frequency
   - Context selection diversity

### 3.4.5 Implementation Details

#### 3.4.5.1 Temperature Parameter Selection and Stability Testing

A critical aspect of the experimental design was determining the optimal temperature parameter for answer generation. Temperature controls the randomness of LLM outputs, with lower values producing more deterministic and consistent responses. To ensure reliable and reproducible results, a comprehensive stability testing experiment was conducted.

**Experimental Procedure**:
1. **Multiple Temperature Values**: The system was tested with various temperature settings (0.0, 0.1, 0.2, and higher values)
2. **Repeated Runs**: For each temperature value, the same set of queries was executed multiple times (10 runs for non-deterministic temperatures)
3. **RAGAS Metrics Evaluation**: Each run was evaluated using RAGAS (Retrieval-Augmented Generation Assessment) metrics, including:
   - Context precision
   - Context recall
   - Faithfulness
   - Answer relevancy
4. **Stability Analysis**: The variance in RAGAS metrics across multiple runs was analyzed to determine which temperature provided the most stable and consistent results

**Results and Decision**:
The stability experiments demonstrated that temperatures in the range of 0.0 to 0.1 provided the most stable results across multiple runs. Specifically:
- **Temperature 0.0**: Produced completely deterministic outputs (runs only once as it's deterministic)
- **Temperature 0.1**: Showed minimal variance across 10 runs while maintaining reasonable response quality
- **Higher temperatures (>0.1)**: Exhibited increased variance in RAGAS metrics, indicating less stable performance

Based on these findings, **temperature 0.0 was selected** as the final parameter for both techniques. This choice was motivated by:
1. **Maximum Stability**: Temperature 0.0 ensures completely deterministic outputs, eliminating variance between runs
2. **Reproducibility**: Results are fully reproducible, which is essential for controlled experiments
3. **Consistent Evaluation**: Stable outputs enable reliable comparison between the two techniques without confounding factors from response variability

The temperature selection process demonstrates the rigor applied to experimental design, ensuring that observed differences between techniques are attributable to the retrieval and generation strategies rather than random variation in LLM outputs.

#### Configuration Parameters

**Technique 1 (Basic)**:
- `max_semantic_results`: 100
- `top_k`: 5
- `temperature`: 0.0 (selected based on stability experiments)

**Technique 2 (Enhanced)**:
- `max_semantic_results`: 300
- `relevance_threshold`: 0.3 (configurable, default 0.1 in experiments)
- `max_context_docs`: Adaptive (5-15 based on query type)
- `enable_diversity`: True
- `temperature`: 0.0 (selected based on stability experiments)

3.5 Data Structure and Storage

### 3.5.1 Vector Database Structure
The knowledge base was stored in ChromaDB with the following structure:
- **Collection Name**: `hybrid_collection`
- **Distance Metric**: Cosine similarity
- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Metadata Fields**: 
  - `doc_id`: Unique chunk identifier
  - `parent_doc_id`: Original document ID
  - `category`: Document category
  - `title`: Document title
  - `content`: Chunk content
  - `chunk_index`: Position in original document
  - `total_chunks`: Total chunks in document
  - `source_url`: Original source (when available)

### 3.5.2 Result Data Structure

Experimental results were stored in JSON format with the following structure:

```json
{
  "query": "User query text",
  "llm_plan": {
    "filters": {...},
    "keywords": [...],
    "expected_answer_type": "single|list|count"
  },
  "search_results": [
    {
      "title": "...",
      "content": "...",
      "category": "...",
      "semantic_score": 0.85,
      "keyword_boost": 1.5,
      "combined_score": 2.35
    }
  ],
  "generated_answer": "Generated answer text",
  "query_type": "single",
  "num_retrieved": 150,
  "num_filtered": 45,
  "num_context_docs": 5,
  "fallback_used": null
}
```
This structure enabled comprehensive analysis of system behavior and performance.

