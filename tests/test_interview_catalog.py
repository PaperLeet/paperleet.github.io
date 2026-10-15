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
        roles = {row['id'] for row in catalog['roles']}
        assert path['roles'] and len(set(path['roles'])) == len(path['roles'])
        assert set(path['roles']) <= roles
    companies = {row['id'] for row in catalog['companies']}
    official = {'www.amazon.jobs', 'careers.microsoft.com', 'openai.com', 'www.google.com',
                'deepmind.google', 'www.nvidia.com', 'www.anthropic.com', 'www.apple.com',
                'jobs.bytedance.com', 'talent.alibaba.com', 'careers.tencent.com', 'talent.baidu.com',
                'zhaopin.meituan.com', 'zhaopin.kuaishou.cn', 'www.deepseek.com', 'x.ai', 'www.tesla.com'}
    assert len({row['id'] for row in catalog['resources']}) == len(catalog['resources'])
    assert len({row['company'] for row in catalog['resources']}) >= 16
    for row in catalog['resources']:
        url = urlsplit(row['url'])
        assert url.scheme == 'https' and url.hostname in official and not url.username and not url.password
        assert row['kind'] in {'guide', 'careers'}
        assert row['company'] in companies and row['checked']
    assert 'reports' not in catalog and 'frequencies' not in catalog
    from pathlib import Path
    project = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert 'interview_catalog.json' in project['tool']['setuptools']['package-data']['paperleet']
