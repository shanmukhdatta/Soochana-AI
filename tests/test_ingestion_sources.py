from datetime import datetime,timezone
import pytest
from app.ingestion.adapters.gdelt import GDELTAdapter
from app.ingestion.adapters.bluesky import BlueskyAdapter

NOW=datetime(2026,10,9,16,0,tzinfo=timezone.utc)

class Response:
    status_code=200
    headers={'Content-Type':'application/json'}
    def __init__(self,data):self.data=data
    def raise_for_status(self):pass
    def json(self):return self.data

class Session:
    def __init__(self,result):self.result=result;self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return Response(self.result)


def test_gdelt_uses_headline_discovery_timestamp_not_verified_publication():
    session=Session({'articles':[{'title':'Nvidia launches a chip','url':'https://news.example/item?utm_source=x',
          'seendate':'20261009120000','language':'English','domain':'news.example'},
          {'title':'Nvidia launches a chip','url':'https://news.example/item','language':'English'}]})
    out=GDELTAdapter({'base_url':'https://api.gdeltproject.org/api/v2/doc/doc',
        'queries':['Nvidia'],'timespan':'1h','max_records_per_query':10},session=session).collect(NOW,10)
    assert len(out)==1 and out[0].timestamp_quality=='gdelt_discovery'
    assert out[0].reported_at.hour==12 and out[0].first_seen_at==NOW
    assert session.calls[0][1]['params']['mode']=='artlist'


def test_gdelt_rejects_malformed_shape():
    with pytest.raises(ValueError):
        GDELTAdapter({'base_url':'https://foo','queries':['Nvidia'],'timespan':'1h',
           'max_records_per_query':2},session=Session({'bad':True,'articles':'no'})).collect(NOW,2)


def test_bluesky_search_extracts_public_posts_without_promising_auth():
    post={'uri':'at://did:plc:abcdef/app.bsky.feed.post/xyz','author':{'did':'did:plc:abcdef'},
          'record':{'text':'$NVDA launches major product','createdAt':'2026-10-09T13:00:00Z', 'langs':['en']}}
    session=Session({'posts':[post,post,{'uri':'no','record':{'text':'invalid'}}]})
    adapter=BlueskyAdapter({'base_url':'https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts',
                           'queries':['$NVDA'],'max_records_per_query':10},session=session)
    out=adapter.collect(NOW,20)
    assert len(out)==1 and out[0].source_type=='social_media'
    assert out[0].source_url.startswith('https://bsky.app/profile/did:plc:abcdef/post/xyz')
    assert out[0].timestamp_quality=='self_reported'
    assert session.calls[0][1]['params']['sort']=='latest'


def test_bluesky_filters_non_english_posts():
    p={'uri':'at://did:plc:a/app.bsky.feed.post/1','author':{'did':'did:plc:a'},
       'record':{'text':'$AAPL geht','langs':['de']}}
    adapter=BlueskyAdapter({'base_url':'https://public.api.bsky.app','queries':['x'],
                  'max_records_per_query':10},session=Session({'posts':[p]}))
    assert adapter.collect(NOW,4)==[]
