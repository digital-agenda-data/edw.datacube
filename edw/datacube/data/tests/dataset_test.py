from mock import ANY
from .base import sparql_test, create_cube


@sparql_test
def test_all_datasets_query_returns_the_dataset():
    cube = create_cube()
    res = cube.get_datasets()
    dataset = {
        'uri': ('http://semantic.digital-agenda-data.eu/'
                'dataset/digital-agenda-scoreboard-key-indicators'),
        'title': 'Digital Agenda Scoreboard Dataset',
    }
    assert dataset in res


@sparql_test
def test_dataset_metadata():
    cube = create_cube()
    res = cube.get_dataset_metadata(cube.dataset)
    assert res['title'] == "Digital Agenda Scoreboard Dataset"
    #assert "You can also browse the data" in res['description']
    #assert res['license'].startswith('http://')

@sparql_test
def test_dataset_dimensions_metadata():
    cube = create_cube()
    res = cube.get_dimensions()
    assert {'notation': 'ref-area',
            'label': "Country",
            'comment': ANY} in res['dimension']
    notations = lambda type_label: [d['notation'] for d in res[type_label]]
    assert sorted(res) == ['attribute', 'dimension',
                           'dimension group', 'measure']
    assert notations('dimension') == ['indicator', 'breakdown', 'unit-measure',
                                      'ref-area', 'time-period']
    assert sorted(notations('dimension group')) == ['breakdown-group',
                                            'indicator-group']
    assert sorted(notations('attribute')) == ['flag', 'note']
    assert [d['label'] for d in res['measure']] == ['Observation']

@sparql_test
def test_dataset_dimensions_flat_list():
    cube = create_cube()
    res = cube.get_dimensions(flat=True)
    assert sorted([d['notation'] for d in res]) == [
        'breakdown',
        'breakdown-group',
        'flag',
        'indicator',
        'indicator-group',
        'note',
        'obsValue',
        'ref-area',
        'time-period',
        'unit-measure',
    ]

@sparql_test
def test_get_dataset_details():
    cube = create_cube()
    res = cube.get_dataset_details()
    by_notation = {r.get('notation'): r for r in res}
    i_iusell = by_notation['i_iusell']
    assert "selling online" in i_iusell.get('short_label', '').lower()
    assert "in the last 3 months" in i_iusell['definition']
    assert i_iusell['groupName'] == "eCommerce"
    assert i_iusell['sourcelabel'] == "Eurostat - ICT Households survey"
    #assert "Extraction from HH/Indiv" in i_iusell['sourcenotes']
    assert i_iusell['sourcelink'] == (
            'http://ec.europa.eu/eurostat'
            '/web/information-society/data/comprehensive-database')

@sparql_test
def test_get_dimension_option_metadata_list():
    cube = create_cube()
    uri_list = [
        'http://semantic.digital-agenda-data.eu/codelist/indicator/e_igov',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/mbb_ltecov',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/gdp',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/foa_cit',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/bb_ne',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/mbb_penet',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/e_itsp2'
    ]
    res = cube.get_dimension_option_metadata_list('indicator', uri_list)
    result = filter( lambda item: item['notation'] == 'e_igov', res)[0]
    assert result['groupName'] == 'eGovernment'
    assert result['innerOrder'] == '5'
    assert result['label'] == 'Enterprises interacting online with public authorities'
    assert result['notation'] == 'e_igov'
    assert result['parentOrder'] == '60'
    assert result['short_label'] == 'Use of eGovernment services - enterprises'
    assert result['source_definition'] == 'Eurostat - Community survey on ICT usage and eCommerce in Enterprises'
    assert result['source_label'] == 'Eurostat - ICT Enterprises survey'
    assert result['source_url'] == 'http://ec.europa.eu/eurostat/web/information-society/data/comprehensive-database'
    assert result['uri'] == 'http://semantic.digital-agenda-data.eu/codelist/indicator/e_igov'
    assert result['definition'][0:31] == 'Use of internet for interaction'
