"""Preparation resources are maintained curriculum, not fabricated interview evidence."""
from importlib.resources import files
from urllib.parse import urlsplit
import json
import tomllib

from paperleet.catalog import problems


def test_shared_interview_taxonomy_and_preparation_paths_are_publishable():
    resource = files('paperleet').joinpath('interview_catalog.json')
    catalog = json.loads(resource.read_text())
    assert catalog['version'] == 1
    for group in ('companies', 'roles', 'stages', 'outcomes'):
        rows = catalog[group]
        assert rows and len({row['id'] for row in rows}) == len(rows)
        assert all(row['name'] and row['name_en'] for row in rows)
    published = {row['id'] for row in problems()}
    assert len(catalog['paths']) >= 3
    for path in catalog['paths']:
        assert path['name'] and path['name_en'] and path['description'] and path['description_en']
        assert path['modules'] and len(set(path['modules'])) == len(path['modules'])
        assert set(path['modules']) <= published
    companies = {row['id'] for row in catalog['companies']}
    official = {'www.amazon.jobs', 'careers.microsoft.com', 'openai.com'}
    for row in catalog['resources']:
        url = urlsplit(row['url'])
        assert url.scheme == 'https' and url.hostname in official
        assert row['company'] in companies and row['checked']
    assert 'reports' not in catalog and 'frequencies' not in catalog
    from pathlib import Path
    project = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert 'interview_catalog.json' in project['tool']['setuptools']['package-data']['paperleet']
