from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

pypdf=pytest.importorskip('pypdf')
from data import BASE_DIR


def test_documents_contain_executed_results_and_reference_limitation():
    summary=pd.read_csv(BASE_DIR/'results/research/summary.csv')
    path=BASE_DIR/'docs/Previsão de Inadimplência de Crédito com Redes Neurais.pdf'
    reader=pypdf.PdfReader(path)
    text=' '.join(' '.join(page.extract_text() for page in reader.pages).split())
    assert len(reader.pages)>=10
    assert '140' in text and 'cinco sementes' in text
    assert 'já havia sido consultado' in text
    for _,row in summary.iterrows():
        value=f'{row.average_precision_mean:.3f}'.replace('.',',')
        assert value in text
    for page in reader.pages:
        assert len(page.extract_text())>60


def test_synthesis_and_presentation_are_valid_and_nonempty():
    for name,expected in [('proposta_pesquisa_inadimplencia.pdf',4),('apresentacao_resultados.pdf',16)]:
        reader=pypdf.PdfReader(BASE_DIR/'docs'/name)
        assert len(reader.pages)==expected
        assert all(len(page.extract_text())>60 for page in reader.pages)


def test_original_documents_preserved():
    original=BASE_DIR/'docs/originais'
    manifest=json.loads((original/'sha256.json').read_text())
    for filename,digest in manifest.items():
        assert hashlib.sha256((original/filename).read_bytes()).hexdigest()==digest
