from .base import sparql_test, create_cube


@sparql_test
def test_notations_lookup_finds_values():
    cube = create_cube()
    notation = 'bb_lines'
    bb_lines_uri = ('http://semantic.digital-agenda-data.eu/codelist/indicator/bb_lines')
    assert (cube.metadata.lookup_notation('indicator', notation)['uri'] == bb_lines_uri)
    assert cube.metadata.lookup_metadata('indicator', bb_lines_uri)['notation'] == notation


@sparql_test
def test_notations_lookup_finds_dimension_groups():
    cube = create_cube()
    notation = 'internet-services'
    internet_services_uri = ('http://semantic.digital-agenda-data.eu/codelist/indicator-group/internet-services')
    assert (cube.metadata.lookup_notation('indicator-group', notation)['uri'] == internet_services_uri)
    assert (cube.metadata.lookup_metadata('indicator-group', internet_services_uri)['notation'] == notation)
