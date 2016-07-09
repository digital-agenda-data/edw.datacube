from mock import ANY
from .base import sparql_test, create_cube


@sparql_test
def test_all_datasets_query_returns_the_dataset():
    cube = create_cube()
    res = cube.get_datasets()
    dataset = {
        'uri': ('http://semantic.digital-agenda-data.eu/'
                'dataset/digital-agenda-scoreboard-key-indicators'),
        'title': ANY
    }
    assert dataset in res


@sparql_test
def test_dataset_metadata():
    cube = create_cube()
    res = cube.get_dataset_metadata(cube.dataset)
    assert "Digital Agenda" in res['title']
    #assert "You can also browse the data" in res['description']
    #assert res['license'].startswith('http://')

@sparql_test
def test_dataset_dimensions_metadata():
    cube = create_cube()
    res = cube.get_dimensions()
    assert {'notation': 'ref-area',
            'label': "Country",
            'comment': ANY,
            'uri': "http://eurostat.linked-statistics.org/dic/geo"} in res['dimension']
    assert {'notation': 'breakdown-group',
            'label': "Breakdown group",
            'comment': ANY,
            'uri': "http://semantic.digital-agenda-data.eu/codelist/breakdown-group"} in res['dimension group']
    assert {'notation': 'flag',
            'label': "Flag",
            'comment': ANY,
            'uri': "http://eurostat.linked-statistics.org/dic/flags"} in res['attribute']
    assert {'notation': 'obsValue',
            'label': "Observation",
            'comment': ANY,
            'uri': None} in res['measure']
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
    assert i_iusell['group_name'][0] == "eCommerce"
    assert i_iusell['source_label'] == "Eurostat - ICT Households survey"
    assert "Extraction" in i_iusell['source_notes']
    assert i_iusell['source_url'] == (
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
        'http://semantic.digital-agenda-data.eu/codelist/indicator/e_itsp2',
        'http://semantic.digital-agenda-data.eu/codelist/indicator/bb_penet'
    ]
    res = cube.get_dimension_option_metadata_list('indicator', uri_list)
    result = filter( lambda item: item['notation'] == 'e_igov', res)[0]
    assert result['group_name'][0] == 'Discontinued indicators'
    assert result['inner_order'][0] == '25'
    assert result['label'] == 'Enterprises interacting online with public authorities'
    assert result['notation'] == 'e_igov'
    assert result['parent_order'][0] == '900'
    assert result['short_label'] == 'Use of eGovernment services - enterprises'
    assert result['source_definition'] == 'Eurostat - Community survey on ICT usage and eCommerce in Enterprises'
    assert result['source_label'] == 'Eurostat - ICT Enterprises survey'
    assert result['source_url'] == 'http://ec.europa.eu/eurostat/web/information-society/data/comprehensive-database'
    assert result['uri'] == 'http://semantic.digital-agenda-data.eu/codelist/indicator/e_igov'
    assert result['definition'][0:31] == 'Use of internet for interaction'

    result = filter( lambda item: item['notation'] == 'bb_penet', res)[0]
    assert result['group_name'][0] == 'Broadband take-up and coverage'
    assert result['inner_order'][0] == '6'
    assert result['label'] == 'Fixed broadband take-up (subscriptions/100 people)'
    assert result['notation'] == 'bb_penet'
    assert result['parent_order'][0] == '20'
    assert result['short_label'] == 'Fixed broadband take-up (penetration rate)'
    assert result['source_definition'][0:43] == 'Electronic communications market indicators'
    assert result['source_label'] == 'Communications Committee survey'
    assert result['source_url'] == 'http://ec.europa.eu/digital-agenda/about-fast-and-ultra-fast-internet-access'
    assert result['uri'] == 'http://semantic.digital-agenda-data.eu/codelist/indicator/bb_penet'
    assert result['definition'][0:39] == 'Number of fixed broadband subscriptions'
