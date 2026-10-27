"""Public-style fixtures for date-based archival; no network or private data."""
import csv
from datetime import date
import json

import pytest

from LCD.experiments.data_creation.collect_originals import write_docx
from LCD.experiments.data_creation.filter_decisions_by_date import create_date_plan, decision_date
from LCD.experiments.data_creation.review_decisions import apply_review


def test_published_date_precedes_archive_metadata_and_ignores_case_year():
    result = decision_date('Постановление № 5-1/2024 от 31 декабря 2023 г.',
                           'Дело №5-1/2024\nПОСТАНОВЛЕНИЕ\n1 января 2024 года\nСудья рассмотрел дело.')
    assert result['date'] == '2024-01-01'
    assert result['method'] == 'published_opening'
    assert result['year_disagreement']


def test_anonymized_opening_uses_title_and_does_not_use_offence_date():
    result = decision_date('Постановление от 4 февраля 2024 г.',
                           'ПОСТАНОВЛЕНИЕ\nДД.ММ.ГГГГ\nСудья рассмотрев дело, УСТАНОВИЛ: 2 января 2020 года нарушил закон.')
    assert result['date'] == '2024-02-04'
    assert result['method'] == 'archive_title'
    assert decision_date('Дело № 5-20/2024', 'Судья рассмотрел дело.')['date'] is None


def test_ambiguous_opening_is_not_silently_resolved():
    assert decision_date('Постановление от 1 января 2024 г.',
                          '2 января 2024 года\n3 января 2024 года\nСудья')['date'] is None


def test_date_archive_keeps_cutoff_and_other_pools_and_verifies_hashes(tmp_path):
    root, output = tmp_path/'decisions', tmp_path/'review'
    root.mkdir()
    fields = ['filename', 'source_text', 'source_html', 'title', 'group', 'article']
    rows = []
    for name, day, group in [('old', '31.12.2023', 'main'), ('new', '01.01.2024', 'main'),
                              ('domain', '01.01.2020', 'domain_shift')]:
        folder = root/group/'18.8'
        folder.mkdir(parents=True, exist_ok=True)
        text = f'ПОСТАНОВЛЕНИЕ\n{day}\nСудья рассмотрел дело.'
        write_docx(folder/f'{name}.docx', text.splitlines())
        (folder/f'{name}.txt').write_text(text)
        (folder/f'{name}.html').write_text(text)
        rows.append(dict(filename=f'{group}/18.8/{name}.docx', source_text=f'{group}/18.8/{name}.txt',
                         source_html=f'{group}/18.8/{name}.html', title='Постановление', group=group, article='18.8'))
    with (root/'provenance.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    original_doc = root/'main/18.8/old.doc'
    original_doc.write_bytes(b'published legacy Word document')
    summary = create_date_plan(root, output, date(2024, 1, 1))
    assert summary == dict(reviewed=2, excluded=1, retained=1, pending=[], cutoff='2024-01-01')
    original_manifest = (root/'provenance.csv').read_bytes()
    original_text = (root/rows[0]['source_text']).read_text()
    (root/rows[0]['source_text']).write_text('Changed after planning')
    with pytest.raises(ValueError, match='Source changed'):
        apply_review(output/'plan.json')
    assert (root/'provenance.csv').read_bytes() == original_manifest
    (root/rows[0]['source_text']).write_text(original_text)
    assert apply_review(output/'plan.json')['excluded'] == 1
    assert not (root/rows[0]['filename']).exists()
    assert (output/'excluded'/rows[0]['filename']).is_file()
    assert not original_doc.exists()
    assert (output/'excluded/main/18.8/old.doc').read_bytes() == b'published legacy Word document'
    assert (root/rows[1]['filename']).is_file() and (root/rows[2]['filename']).is_file()
    assert (output/'before_review/provenance.csv').read_bytes() == original_manifest
    assert len(json.loads((output/'plan.json').read_text())['records']) == 2
