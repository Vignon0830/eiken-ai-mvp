import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_app import get_namespace

def test_get_namespace():
    assert get_namespace("5級", "語彙") == "vocab-5"
    assert get_namespace("準2級", "長文") == "passages-pre2"
    assert get_namespace("3級", "リスニング") == "listening-3"
