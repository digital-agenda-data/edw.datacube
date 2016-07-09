from mock import ANY
from .base import sparql_test, create_cube

@sparql_test
def test_cube_dimensions_cache():
    cube = create_cube()
    metadata = cube.metadata.get()
    dimensions = metadata['dimensions']
    assert 'http://semantic.digital-agenda-data.eu/def/property/indicator-group' in dimensions
    assert 'http://semantic.digital-agenda-data.eu/def/property/indicator' in dimensions
    assert 'http://purl.org/linked-data/sdmx/2009/measure#obsValue' in dimensions
    assert {'notation': 'ref-area',
            'label': "Country",
            'short_label': "Country",
            'comment': ANY,
            'type_label': 'dimension',
            'uri': "http://eurostat.linked-statistics.org/dic/geo",
            'dimension': 'http://semantic.digital-agenda-data.eu/def/property/ref-area',
            'dimension_type': 'http://purl.org/linked-data/cube#dimension',
            'group_dimension': None,
            'group_notation': None,
            } in dimensions.values()
    assert {'notation': 'breakdown-group',
            'label': "Breakdown group",
            'short_label': "Breakdown group",
            'comment': ANY,
            'type_label': 'dimension group',
            'uri': "http://semantic.digital-agenda-data.eu/codelist/breakdown-group",
            'dimension': 'http://semantic.digital-agenda-data.eu/def/property/breakdown-group',
            'dimension_type': None,
            'group_dimension': None,
            'group_notation': None,
            } in dimensions.values()
    assert {'notation': 'breakdown',
            'label': "Breakdown",
            'short_label': "Breakdown",
            'comment': ANY,
            'type_label': 'dimension',
            'uri': "http://semantic.digital-agenda-data.eu/codelist/breakdown",
            'dimension': 'http://semantic.digital-agenda-data.eu/def/property/breakdown',
            'dimension_type': 'http://purl.org/linked-data/cube#dimension',
            'group_dimension': 'http://semantic.digital-agenda-data.eu/def/property/breakdown-group',
            'group_notation': 'breakdown-group',
            } in dimensions.values()
    assert {'notation': 'flag',
            'label': "Flag",
            'short_label': "Flag",
            'comment': ANY,
            'type_label': 'attribute',
            'uri': "http://eurostat.linked-statistics.org/dic/flags",
            'dimension': 'http://semantic.digital-agenda-data.eu/def/property/flag',
            'dimension_type': 'http://purl.org/linked-data/cube#attribute',
            'group_dimension': None,
            'group_notation': None,
            } in dimensions.values()
    assert {'notation': 'obsValue',
            'label': "Observation",
            'short_label': "Observation",
            'comment': ANY,
            'type_label': 'measure',
            'uri': None,
            'dimension': 'http://purl.org/linked-data/sdmx/2009/measure#obsValue',
            'dimension_type': 'http://purl.org/linked-data/cube#measure',
            'group_dimension': None,
            'group_notation': None,
            } in dimensions.values()

    groupers = metadata['groupers']
    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown-group' in groupers
    assert 'http://semantic.digital-agenda-data.eu/def/property/indicator-group' in groupers

    grouped = metadata['grouped_dimensions']
    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown' in grouped
    assert 'http://semantic.digital-agenda-data.eu/def/property/indicator' in grouped

    by_notation = metadata['dimensions_by_notation']
    assert 'breakdown-group' in by_notation
    assert 'breakdown' in by_notation
    assert 'time-period' in by_notation
    assert 'flag' in by_notation
    assert 'obsValue' in by_notation

    assert 'http://purl.org/linked-data/sdmx/2009/measure#obsValue' == metadata['measure_uri']

    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown-group' in cube.metadata.get_all_dimensions_uri()
    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown' in cube.metadata.get_all_dimensions_uri()
    assert 'http://purl.org/linked-data/sdmx/2009/measure#obsValue' not in cube.metadata.get_all_dimensions_uri()

    assert 'breakdown' == cube.metadata.lookup_dimension_code('http://semantic.digital-agenda-data.eu/def/property/breakdown')
    assert 'breakdown-group' == cube.metadata.lookup_dimension_code('http://semantic.digital-agenda-data.eu/def/property/breakdown-group')

    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown' == cube.metadata.lookup_dimension_uri('breakdown')
    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown-group' == cube.metadata.lookup_dimension_uri('breakdown-group')

    assert 'http://semantic.digital-agenda-data.eu/def/property/breakdown' == cube.metadata.lookup_dimension_uri_by_grouper_uri('http://semantic.digital-agenda-data.eu/def/property/breakdown-group')

    assert cube.metadata.is_group_dimension('http://semantic.digital-agenda-data.eu/def/property/breakdown-group')
    assert cube.metadata.is_group_dimension('http://semantic.digital-agenda-data.eu/def/property/indicator-group')
    assert cube.metadata.is_grouped_dimension('http://semantic.digital-agenda-data.eu/def/property/indicator')
    assert cube.metadata.is_grouped_dimension('http://semantic.digital-agenda-data.eu/def/property/breakdown')

