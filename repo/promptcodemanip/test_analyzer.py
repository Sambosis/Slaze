#!/usr/bin/env python3
"""
Standalone test script for CodeAnalyzer.
Assumes sample1.py and sample2.py are in the same directory.
Verifies parsing, graph building, metrics, semantic search, and complexity cache.
Run with: python test_analyzer.py
"""

from analyzer import CodeAnalyzer
import pprint

if __name__ == "__main__":
    analyzer = CodeAnalyzer()
    print('Parsing sample1.py')
    analyzer.parse_file('sample1.py')
    print('Parsing sample2.py')
    analyzer.parse_file('sample2.py')
    print('\nMerged Graph:')
    G = analyzer.get_merged_graph()
    print(f'Nodes: {list(G.nodes)}')
    print(f'Edges: {list(G.edges)}')
    print('\nMetrics:')
    pprint.pprint(analyzer.get_metrics())
    print('\nSemantic Search "hello":')
    print(analyzer.semantic_search('hello'))
    print('\nSearch "work":')
    print(analyzer.semantic_search('work'))
    print('\nComplexity cache:', analyzer.complexity_cache)