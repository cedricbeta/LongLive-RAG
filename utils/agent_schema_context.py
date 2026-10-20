"""Opt-in schema context, separate from server-side token constraints.

Appending the schema to actual prompt bytes also gives the existing client a
new fingerprint. It does not change sampling, validation, or cache semantics.
"""
import json


class SchemaContextClient:
    def __init__(self,client):self.client=client
    @property
    def cache_dir(self):return self.client.cache_dir
    def call(self,*,prompt,schema,**kwargs):
        context='\nRequired response JSON schema (field names and allowed values):\n'+json.dumps(schema,ensure_ascii=False,sort_keys=True)
        return self.client.call(prompt=prompt+context,schema=schema,**kwargs)
