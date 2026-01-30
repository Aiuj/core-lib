# Release Notes

## Overview

This document tracks the main evolutions of the core-lib project, a Python utility library providing reusable components for caching, LLM integration, embeddings, tracing, configuration management, and API utilities.

---

## Release process

To create a new release, use this prompt:

```
Can you look at github history, update the version number in pyproject and library setup.py accordingly.
Then update the release notes and eventually the todo docs as implementation done must be removed. 
```

And once we are sure everything is pushed and commited:

```
Then commit, push and tag this new version in github and create a release for this tag in github with proper description in valid markdown format.
```

---

## January 2026

### v0.3.4 - Auth, Reliability & Usage Tracking (Jan 30, 2026)

- **Authentication Enhancements**: Added JWT auth support and FastMCP v2 middleware, improved legacy auth handling and error messaging
  - Updated ApiSettings/StandardSettings authentication mode handling
  - Added support for static API keys and multiple private keys
- **LLM Reliability & Selection**: Added FallbackLLMClient and ProviderRegistry health-aware selection with env var substitution
  - Added latency measurement and usage tracking for LLM providers
  - Updated OpenAI default model to gpt-5.2-preview
- **Embeddings & Reranker Improvements**: Added usage metrics with latency/cost tracking and reranker token usage
  - Added embedding sanitization for NaN/Inf values in Local and Ollama clients
  - Added query/passage prefix settings for asymmetric retrieval models
- **Miscellaneous Updates**: Added custom exceptions for configuration/API errors and updated docs/instructions
  - Added LangChain integration fallback support and improved ExcelManager efficiency
  - Added additional env var support for LLM providers and Gemini schema cleaning
  - Updated default confidentiality level to prospect

## November 2025

### v0.3.3 - Search Usage & Logging Improvements (Nov 27, 2025)

- **Search Usage Logging**: Added search usage logging functionality with updated service type enumeration
  - Changed embedding usage logging from debug to warning level for better visibility
- **OTLP Debug Mode**: Added debug mode for OTLP logging and updated documentation for test configuration
- **Noisy Logger Suppression**: Updated noisy logger suppression list to include additional MCP server components

### v0.3.2 - Infrastructure Utilities (Nov 24, 2025)

- **Uvicorn Runner Utilities**: Added Uvicorn runner utilities for ASGI applications with comprehensive documentation
- **HealthChecker Utility**: New health monitoring utility for service health checks
- **Enhanced Configuration Parsing**: Improved environment variable parsing to handle inline comments in Redis and Valkey configurations
- **Tracing Robustness**: Improved error handling in LangfuseTracingProvider's add_metadata method

### v0.3.1 - AppSettings Refactor (Nov 21, 2025)

- **AppSettings Refactor**: Major refactoring of AppSettings module with improved clarity and usage documentation
- **Code Structure Improvements**: General code structure refactoring for improved readability and maintainability

### v0.3.0 - Library Rename (Nov 20, 2025)

- **Package Rename**: Renamed package from `faciliter-lib`/`faciliter_lib` to `core-lib`/`core_lib`
- **Backward Compatibility**: Added backward compatibility shim under `faciliter_lib` to avoid breaking existing consumers
- **Updated Defaults**: Updated default OTLP service name and logger scope to "core-lib"
- **Documentation Updates**: Updated all documentation, examples, and references to reflect core-lib naming
- **Import Consistency**: All imports in the library now consistently use `core_lib` to avoid circular imports

---

## November 2025 (Early)

### v0.2.13 - API Client & Embedding Enhancements (Nov 18, 2025)

- **Enhanced APIClient**: Added additional HTTP methods (PUT, PATCH, DELETE) and improved error handling
- **FallbackEmbeddingClient**: New high-availability embedding client with multi-provider fallback support
  - Automatic failover between embedding providers
  - Comprehensive test coverage for fallback functionality
- **Server Configuration**: Enhanced server configuration for backward compatibility with SERVER_HOST and SERVER_PORT
- **Noisy Log Suppression**: Suppress noisy OpenTelemetry exporter logs to reduce clutter in application logs
- **Confidentiality Levels**: Added confidentiality level constants and utilities for access control
- **Version Management**: Added version management documentation

### v0.2.12 - Google GenAI & Logging Improvements (Nov 14, 2025)

- **Google GenAI Provider**: Added instrumentation handling and response text extraction improvements
  - JSON parsing utilities and fallback support for better reliability
- **OTLP Compatibility Fix**: Fixed boolean to string conversion for feature flags in OTLP logging
- **Test Refactoring**: Renamed `TestSchema` to `SampleSchema` in tests for clarity
- **Django Integration**: Added Django Logging Integration Guide and example configuration
- **Application Metadata**: Enhanced log records with app name and version attributes
- **Module Refactoring**: Moved `parse_from` function to `tracing` module for better organization

### v0.2.11 - Observability & Authentication (Nov 4, 2025)

- **Service Usage Tracking**: Added service usage tracking with OpenTelemetry/OpenSearch integration
- **Intelligence Level Support**: Added intelligence_level support to tracing, middleware, and tests
- **Implementation Docs**: Added implementation summary docs for Auth, Infinity, OTLP, and Service Usage
- **FROM Field Description**: Centralized FROM_FIELD_DESCRIPTION for API parameter tracing

---

## October 2025

### v0.2.10 - OTLP Logging & Settings (Oct 30, 2025)

- **OTLP Logging Auto-Enable**: Automatic OTLP logging enablement with service name/version detection
- **Contextual Logging**: Added LoggingContext support and OpenSearch dashboard documentation
- **Module Logger**: Use get_module_logger across modules for consistent logging
- **Independent OTLP Log Level**: Support for independent OTLP log level configuration
- **Automatic Logging Setup**: Added optional automatic logging setup to initialize_settings

### v0.2.9 - Logging Infrastructure (Oct 25-29, 2025)

- **OTLP Logging Support**: Full OTLP logging implementation with handler, config, integration, docs, examples and tests
- **LoggerSettings**: Added LoggerSettings with OVH LDP and file logging support
- **ApiSettings Refactor**: Introduced ApiSettings and refactored StandardSettings to inherit from it
- **OpenSearchSettings**: Added OpenSearchSettings config module with env parsing, validation and client config

### v0.2.8 - Gemini Retry Handling (Oct 20, 2025)

- **Retry Logic for Server Errors**: Hardened Gemini retry handling for 503 "model is overloaded" errors
  - Automatic retry with exponential backoff now handles server overload errors gracefully
- **Improved Logging**: Cleaner error logging with retryable errors showing warnings instead of full tracebacks
- **Test Coverage**: Added 503 retry test and documentation clarifications
- **Document Categories**: Added technical, operations and HR document categories to DOC_CATEGORIES

### v0.2.7 - Observability & Tracing (Oct 16, 2025)

- **Observability Docs**: Added comprehensive observability documentation
- **Tracing Hardening**: Hardened tracing add_metadata method for better reliability
- **Index Rename**: Renamed index from 'contextual_chunks' to 'document_chunks' for embedding storage

### v0.2.6 - API Authentication (Oct 12, 2025)

- **Dynamic API Key Authentication**: Implemented dynamic API key management mechanism
- **API Client Authentication**: Added authentication support to API client

### v0.2.5 - Infinity Embeddings (Oct 6-10, 2025)

- **Infinity Server Provider**: Added Infinity server provider for embeddings
- **Embedding Normalization**: Manage embedding normalization based on model characteristics
- **URL Consistency**: Renamed INFINITY_BASE_URL for configuration consistency

### v0.2.4 - Job Queue & Settings (Oct 1, 2025)

- **Job Queue**: Basic job queue implementation using Redis/Valkey
- **Settings Singleton**: Better settings singleton management

---

## September 2025

### v0.2.3 - Cache & Configuration (Sep 22-26, 2025)

- **Standard Settings Class**: Added standard settings class to ease configuration management across applications
- **FastAPI Configuration**: FastAPI configuration implementation with safe nuller
- **Connection Pooling**: Added connection pooling and healthcheck on cache manager (Redis and ValKey)
- **Improved Cache Management**: Multi-tenancy support and cache clearing methods
- **Better Logging**: Enhanced logging settings with local file support

### v0.2.2 - Rate Limits & LLM (Sep 21-25, 2025)

- **Gemini Rate Limits**: LLM implementation of retry and rate limits logic for Gemini
- **Language Detection**: Enhanced language detection with list of detected languages
- **Excel Handling**: Better manage excel file and byte loading with proper file closing
- **Embedding Cache**: Cache embeddings for 24h in memory cache

### v0.2.1 - Valkey Support (Sep 17-18, 2025)

- **Valkey Cache**: Added Valkey as drop-in open source Redis replacement
- **MCP File Utils**: New temporary file creation from MCP file objects
- **Embeddings Module**: Extended embeddings capabilities with new providers (Ollama, Google, OpenAI, local models)
- **LLM Factory**: Simplified LLM usage by adding a factory pattern

### v0.2.0 - Excel Manager (Sep 1-7, 2025)

- **Excel Manager**: Added ExcelManager in tools to convert Excel into Markdown or JSON IR structure
- **Document Categories**: Refactored categories into doc_categories module

---

## August 2025

### v0.1.1 - LLM Foundation (Aug 12-13, 2025)

- **OpenAI Implementation**: Draft OpenAI provider implementation
- **Thinking Mode**: Added thinking mode support for LLM providers
- **Langfuse Tracing**: Added Langfuse tracing for Google GenAI
- **Ollama Structured Output**: Managed Ollama structured output support
- **Native Libraries**: New version of LLM management using native libraries and standard logging

### v0.1.0 - Initial Release (Aug 6-7, 2025)

- **Language Detection**: Added language detection utility
- **Central Logging**: Implemented central logging with app name support
- **JSON Parsing**: Added JSON parsing utilities
- **LLM Instantiation**: Basic LLM client instantiation

---

## July 2025

### v0.0.1 - Project Bootstrap (Jul 27-31, 2025)

- **Initial Commit**: Project initialization with basic structure
- **Redis Cache**: Redis-backed caching implementation with mock support for testing
- **LangFuse Tracing**: Initial LangFuse tracing class implementation
- **CI/CD Setup**: Fixed Python version in CI/CD pipeline
- **Build Dependencies**: Added build dependencies for package distribution
