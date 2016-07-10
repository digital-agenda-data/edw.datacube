# -*- coding: utf-8 -*-
from mock import Mock, MagicMock, call
import csv
import simplejson as json
import pytest
from StringIO import StringIO
from edw.datacube.browser.export import ExportCSV


def test_columns_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [{
        'data': [
            {
                'code': 'SK',
                'name': 'Slovakia',
                'y': 1

            }
        ],
        'name': 'series1',
        'notation': None
    }]
    exporter.datapoints(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[0] == 'Data extracted:';
    assert csv_output[1].split(',') == ['series', 'name', 'code', 'y'];
    assert csv_output[2].split(',') == ['series1', 'Slovakia', 'SK', '1']


def test_csv_metadata():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    metadata = {
        'chart-title': 'Number of households',
        'source-dataset': 'Digital-Agenda',
        'chart-url': 'http://url.to/chart#table',
        'filters-applied': [
            ['indicator', 'Internet Usage'],
            ['breakdown', 'by age'],
            ['time-period', '2012']
        ],
    }
    exporter.write_metadata(out, metadata)
    out.seek(0)
    output = out.read().split('\r\n')
    from datetime import datetime
    assert output[0] == 'Chart title:,Number of households'
    assert output[1] == 'Source dataset:,Digital-Agenda'
    assert output[2] == 'Extraction-Date:,' + datetime.now().strftime('%d %b %Y')
    assert output[3] == 'Link to the chart/table:,http://url.to/chart#table'
    assert output[4] == 'Selection of filters applied'
    assert output[5] == 'indicator,Internet Usage'
    assert output[6] == 'breakdown,by age'
    assert output[7] == 'time-period,2012'

def test_csv_annotations():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    metadata = {
        "description": "description text",
        "section_title": "Definition and scopes:",
        "indicators_details_url": "http://url.to/indicators",
        "blocks": [
            {
            "definition": "dèfinitioñ ţext",
            "filter_label": "Indicator",
            "label": "long label text",
            "source_label": "source label text",
            "source_url": "http://url.to/source",
            "source_definition": "source definition text",
            "note": "note text"
            }
        ],
    }
    exporter.write_annotations(out, metadata)
    out.seek(0)
    output = out.read().split('\r\n')
    assert output[0] == 'Definition and scopes:'
    assert output[1] == 'Indicator:,long label text'
    assert output[2] == 'Definition:,dèfinitioñ ţext'
    assert output[3] == 'Notes:,note text'
    assert output[4] == 'Source:,source definition text'
    assert output[5] == 'List of available indicators:,http://url.to/indicators'

def test_columns_csv_export_multiseries():
    exporter = ExportCSV(MagicMock(), MagicMock())
    headers = ['series', 'name', 'code', 'y'];
    out = StringIO()
    series = [
        {
            'data': [
                {
                    'code': 'SK',
                    'name': 'Slovakia',
                    'y': 1

                }
            ],
            'name': 'series1',
            'notation': None
        },
        {
            'data': [
                {
                    'code': 'AB',
                    'name': 'ABCD',
                    'y': 2

                }
            ],
            'name': 'series2',
            'notation': None
        }
    ]
    exporter.datapoints(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[2].split(',') == ['series1', 'Slovakia', 'SK', '1']
    assert csv_output[3].split(',') == ['series2', 'ABCD', 'AB', '2']


def test_lines_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [
        {
            'data': [
                {
                    'code': '2000',
                    'name': 'Year:2000',
                    'y': 1

                }
            ],
            'name': 'series1',
            'notation': None
        },
    ]
    exporter.datapoints(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[2].split(',') == ['series1', 'Year:2000', '2000', '1']


def test_scatter_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [
        [
            {
                'data': [
                    {
                        'name': 'AT',
                        'x': 1,
                        'y': 1

                    }
                ],
                'name': 'Austria'
            },
        ]
    ]
    exporter.datapoints_n(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[0].split(',') == ['series', 'name', 'x', 'y'];
    assert csv_output[1].split(',') == ['Austria', 'AT', '1', '1']


def test_bubbles_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [
        [
            {
                'data': [
                    {
                        'name': 'AT',
                        'x': 1,
                        'y': 1,
                        'z': 1

                    }
                ],
                'name': 'Austria'
            },
        ]
    ]
    exporter.datapoints_n(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[0].split(',') == ['series', 'name', 'x', 'y', 'z'];
    assert csv_output[1].split(',') == ['Austria', 'AT', '1', '1', '1']


def test_bar_profile_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [
        {
            'data': [
                {
                    'attributes': {
                        'time-period': {
                            'notation': '2000'
                        }
                    },
                    'code': 'notation',
                    'name': 'long_label',
                    'eu': 0.5,
                    'original': 0.9,
                    'y': 2,

                }
            ],
            'name': 'Austria'
        },
    ]
    exporter.datapoints_profile(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[0].split(',') == [
            'period', 'name', 'eu', 'original']
    assert csv_output[1].split(',') == [
            '2000', 'long_label', '0.5', '0.9']


def test_table_profile_csv_export():
    exporter = ExportCSV(MagicMock(), MagicMock())
    out = StringIO()
    series = [
        {
            'data': {
                'latest': 2010,
                'ref-area': {
                    'label': 'Austria'
                },
                'table':{
                    'some-key': {
                        '2009': 0.8,
                        '2010': 0.9,
                        'eu': 52,
                        'breakdown': 'All individuals',
                        'indicator': 'Telephoning or video calls',
                        'unit-measure': '% of internet users (last 3 months)',
                        'rank': 15,
                        'inner_order': 5,
                        'unit': 'pc_ent'
                    },
                    'some-other-key': {
                        '2009': 0.8,
                        'eu': 52,
                        'breakdown': 'ex breakdown',
                        'indicator': 'ex indicator',
                        'unit-measure': 'ex unit',
                        'rank': 15
                    }
                }
            }
        }
    ]
    exporter.datapoints_profile_table(out, series)
    out.seek(0)
    csv_output = out.read().split('\r\n')
    assert csv_output[0].split(',') == [
            'country', 'indicator', 'breakdown', 'unit', '2007', '2008', '2009', '2010', 'EU28 value 2010', 'rank']
    assert csv_output[1].split(',') == [
            'Austria', 'Telephoning or video calls', 'All individuals', '% of internet users (last 3 months)', '-', '-', '0.8', '0.9', '52', '15']
    assert csv_output[2].split(',') == [
            'Austria', 'ex indicator', 'ex breakdown', 'ex unit', '-', '-', '0.8', '-', '52', '15']


def test_formatter_decision():
    exporter = ExportCSV(MagicMock(), MagicMock())
    series = [
        {
            'data': [
                {
                    'code': '2000',
                    'name': 'year:2000',
                    'y': 1

                }
            ],
            'name': 'series1',
            'notation': None
        },
    ]
    exporter.datapoints = Mock()
    exporter.datapoints_n = Mock()

    exporter.request.form = {
        'metadata': json.dumps({}),
        'chart_data': json.dumps(series),
        'chart_type': 'columns',
        'annotations': json.dumps({})
    }
    exporter.export()
    assert(exporter.datapoints.call_count == 1)

    exporter.request.form = {
        'metadata': json.dumps({}),
        'chart_data': json.dumps(series),
        'chart_type': 'bubbles',
        'annotations': json.dumps({})
    }
    exporter.export()
    assert(exporter.datapoints_n.call_count == 1)

    exporter.request.form = {
        'metadata': json.dumps({}),
        'chart_data': json.dumps(series),
        'chart_type': 'scatter',
        'annotations': json.dumps({})
    }
    exporter.export()
    assert(exporter.datapoints_n.call_count == 2)
