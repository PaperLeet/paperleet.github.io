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
    assert len({path['id'] for path in catalog['paths']}) == len(catalog['paths'])
    assert {module for path in catalog['paths'] for module in path['modules']} == published
    for path in catalog['paths']:
        assert path['name'] and path['name_en'] and path['description'] and path['description_en']
        assert path['modules'] and len(set(path['modules'])) == len(path['modules'])
        assert set(path['modules']) <= published
        roles = {row['id'] for row in catalog['roles']}
        assert path['roles'] and len(set(path['roles'])) == len(path['roles'])
        assert set(path['roles']) <= roles
    companies = {row['id'] for row in catalog['companies']}
    official = {'www.amazon.jobs', 'careers.microsoft.com', 'openai.com', 'www.google.com', 'www.metacareers.com',
                'deepmind.google', 'developers.google.com', 'www.nvidia.com', 'docs.nvidia.com',
                'engineering.fb.com', 'nvidia.github.io', 'www.anthropic.com', 'www.apple.com',
                'jobs.bytedance.com', 'talent.alibaba.com', 'careers.tencent.com', 'talent.baidu.com',
                'zhaopin.meituan.com', 'zhaopin.kuaishou.cn', 'www.deepseek.com', 'x.ai', 'www.tesla.com', 'www.microsoft.com'}
    # Publisher-owned code references outside the official hosts are exact URL/company pairs.
    official_code = {'https://github.com/meta-llama/llama3/blob/32b7ddc9b638f0fd6b04b80e1845536a95f7a5d5/llama/model.py': 'meta',
                     'https://www.tensorflow.org/tutorials/generative/cvae': 'google'}
    assert len({row['id'] for row in catalog['resources']}) == len(catalog['resources'])
    assert len({row['company'] for row in catalog['resources']}) >= 17
    for row in catalog['resources']:
        url = urlsplit(row['url'])
        assert url.scheme == 'https' and (url.hostname in official or official_code.get(row['url']) == row['company']) and not url.username and not url.password
        assert all(isinstance(row.get(field),str) and row[field].strip() for field in ('name','name_en','description','description_en'))
        assert row['kind'] in {'guide', 'careers', 'technical'}
        assert row['company'] in companies and row['checked']
    assert 'reports' not in catalog and 'frequencies' not in catalog
    from pathlib import Path
    project = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert 'interview_catalog.json' in project['tool']['setuptools']['package-data']['paperleet']
