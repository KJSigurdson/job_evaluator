import httpx, json

body = {
    "operationName": "AlgoliaSearchKey",
    "variables": {},
    "query": "query AlgoliaSearchKey {\n  algolia_search_key {\n    api_key\n    app_id\n    index_name\n    index_name_sorted_by_votes\n    index_name_profiles\n    index_name_jobs\n    index_name_jobs_sorted_by_closes_at\n    __typename\n  }\n}",
}
headers = {
    "content-type": "application/json",
    "origin": "https://jobs.probablygood.org",
    "referer": "https://jobs.probablygood.org/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
}
r = httpx.post("https://backend.jobs.probablygood.org/api/graphql/AlgoliaSearchKey", json=body, headers=headers)
print("HTTP", r.status_code)
print(json.dumps(r.json(), indent=2)[:1500])