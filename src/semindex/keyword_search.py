"""
Keyword search module using Elasticsearch for hybrid search functionality.
"""
import os
from typing import List, Tuple, Optional
from elasticsearch import Elasticsearch
from dataclasses import dataclass


@dataclass
class KeywordResult:
    symbol_id: int
    path: str
    name: str
    kind: str
    score: float


class KeywordSearcher:
    """
    Handles keyword search functionality using Elasticsearch.
    """
    
    def __init__(self, index_dir: str, es_host: str = "localhost", es_port: int = 9200):
        """
        Initialize the keyword searcher.
        
        :param index_dir: Directory where index files are stored
        :param es_host: Elasticsearch host
        :param es_port: Elasticsearch port
        """
        self.index_dir = index_dir
        self.es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])
        self.index_name = "semindex_code"
        
    def create_index(self):
        """
        Create the Elasticsearch index with appropriate mappings for code search.
        """
        mapping = {
            "mappings": {
                "properties": {
                    "symbol_id": {"type": "integer"},
                    "path": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "code_analyzer"},
                    "kind": {"type": "keyword"},
                    "signature": {"type": "text", "analyzer": "code_analyzer"},
                    "docstring": {"type": "text", "analyzer": "code_analyzer"},
                    "imports": {"type": "text", "analyzer": "code_analyzer"},
                    "bases": {"type": "text", "analyzer": "code_analyzer"},
                    "content": {"type": "text", "analyzer": "code_analyzer"}
                }
            },
            "settings": {
                "analysis": {
                    "analyzer": {
                        "code_analyzer": {
                            "tokenizer": "whitespace",
                            "filter": [
                                "lowercase",
                                "code_ngram"
                            ]
                        }
                    },
                    "filter": {
                        "code_ngram": {
                            "type": "ngram",
                            "min_gram": 2,
                            "max_gram": 10,
                            "token_chars": ["letter", "digit", "punctuation", "symbol"]
                        }
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)
    
    def index_symbol(self, symbol_data: dict):
        """
        Index a single symbol in Elasticsearch.
        
        :param symbol_data: Dictionary containing symbol information
        """
        doc = {
            "symbol_id": symbol_data["id"],
            "path": symbol_data["path"],
            "name": symbol_data["name"],
            "kind": symbol_data["kind"],
            "signature": symbol_data.get("signature", ""),
            "docstring": symbol_data.get("docstring", ""),
            "imports": symbol_data.get("imports", ""),
            "bases": symbol_data.get("bases", ""),
            "content": symbol_data.get("content", "")
        }
        
        self.es.index(index=self.index_name, body=doc, id=symbol_data["id"])
    
    def bulk_index_symbols(self, symbols_data: List[dict]):
        """
        Bulk index multiple symbols in Elasticsearch.
        
        :param symbols_data: List of dictionaries containing symbol information
        """
        actions = []
        for symbol_data in symbols_data:
            doc = {
                "symbol_id": symbol_data["id"],
                "path": symbol_data["path"],
                "name": symbol_data["name"],
                "kind": symbol_data["kind"],
                "signature": symbol_data.get("signature", ""),
                "docstring": symbol_data.get("docstring", ""),
                "imports": symbol_data.get("imports", ""),
                "bases": symbol_data.get("bases", ""),
                "content": symbol_data.get("content", "")
            }
            
            action = {
                "_index": self.index_name,
                "_id": symbol_data["id"],
                "_source": doc
            }
            actions.append(action)
        
        from elasticsearch.helpers import bulk
        bulk(self.es, actions)
    
    def search(self, query: str, top_k: int = 10) -> List[KeywordResult]:
        """
        Perform keyword search.
        
        :param query: Search query string
        :param top_k: Number of top results to return
        :return: List of keyword search results
        """
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name^3", "signature^2", "docstring^1.5", "content", "imports", "bases"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": top_k
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                KeywordResult(
                    symbol_id=source["symbol_id"],
                    path=source["path"],
                    name=source["name"],
                    kind=source["kind"],
                    score=hit["_score"]
                )
            )
        
        return results
    
    def delete_index(self):
        """
        Delete the Elasticsearch index.
        """
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)