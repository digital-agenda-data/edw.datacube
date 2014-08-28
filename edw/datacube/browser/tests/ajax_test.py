# -*- coding: utf-8 -*-
from mock import Mock, MagicMock, call, patch
import simplejson as json
import pytest
from edw.datacube.browser.query import AjaxDataView


def ajax(cube, name, form):
    from edw.datacube.browser.query import AjaxDataView
    datasource = Mock(get_cube=Mock(return_value=cube))
    view = AjaxDataView(datasource, Mock(form=form))
    return json.loads(getattr(view, name)())


@pytest.fixture()
def mock_cube(request):
    from edw.datacube.data.cube import Cube
    return MagicMock(spec=Cube)


def test_dump_csv_fields_order(mock_cube):
    dump = ('header\n'
           'breakdown1,indicator1,ref-area1,time-period1,unit-measure1,value1\n')
    mock_cube.dump.return_value = dump

    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    datasource.getId.return_value = 'testcube'
    view = AjaxDataView(datasource, Mock(form={}))
    import csv
    stream = Mock()
    res = view.dump_csv(stream, csv.excel)
    header_write_call = res.write.mock_calls[0]
    row_write_call = res.write.mock_calls[1]
    assert header_write_call == call('indicator,breakdown,unit_measure,' \
                                     'time_period,ref_area,value\r\n')
    assert row_write_call == call(
           'indicator1,breakdown1,unit-measure1,time-period1,ref-area1,value1\r\n')


def test_dump_csv_response_content_type(mock_cube):
    dump = [ { 'indicator': 'i',
               'breakdown': u'b',
               'unit_measure': 'u-m',
               'time_period': 't',
               'ref_area': 'r',
               'value': 0.5 } ]
    mock_cube.dump.return_value = iter(dump)

    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    datasource.getId.return_value = 'testcube'
    view = AjaxDataView(datasource, Mock(form={}))
    res = view.download_csv()
    setHeader_call = res.setHeader.mock_calls[0]
    assert setHeader_call == call('Content-type', 'text/csv; charset=utf-8')


def test_all_datasets(mock_cube):
    datasets = [
        {'uri': 'dataset-one', 'title': 'One'},
        {'uri': 'dataset-two', 'title': 'Two'},
    ]
    mock_cube.get_datasets.return_value = datasets
    res = ajax(mock_cube, 'datasets', {})
    assert res['datasets'] == datasets


def test_dataset_metadata(mock_cube):
    metadata = {
        'title': "Teh Title",
        'description': "Teh Description",
        'license': "http://example.com/license",
    }
    mock_cube.get_dataset_metadata.return_value = metadata
    res = ajax(mock_cube, 'dataset_metadata',
               {'dataset': 'http://the-dataset'})
    assert res == metadata
    cube_call = mock_cube.get_dataset_metadata.mock_calls[0]
    assert cube_call == call('http://the-dataset')


def test_dimension_all_indicator_options(mock_cube):
    mock_cube.get_dimension_options.return_value = [
        {'label': 'indicator one', 'notation': 'one'},
        {'label': 'indicator two', 'notation': 'two'},
    ]
    res = ajax(mock_cube, 'dimension_options', {'dimension': 'indicator'})
    assert {'label': 'indicator one', 'notation': 'one'} in res['options']
    assert {'label': 'indicator two', 'notation': 'two'} in res['options']


def test_dimension_single_filter_passed_on_to_query(mock_cube):
    mock_cube.get_dimension_options.return_value = {}
    ajax(mock_cube, 'dimension_options', {
        'dimension': 'ref-area',
        'time-period': '2002',
        'rev': '123',
    })
    cube_call = mock_cube.get_dimension_options.mock_calls[0]
    assert cube_call == call('ref-area', [('time-period', '2002')])


def test_dimension_labels_passed_on_to_query(mock_cube):
    mock_cube.get_dimension_labels.return_value = [
        {'label': 'indicator one', 'short_label': 'ind one'},
    ]
    ajax(mock_cube, 'dimension_labels', {
        'dimension': 'unit-measure',
        'value': 'pc_ind',
        'rev': '123',
    })
    cube_call = mock_cube.get_dimension_labels.mock_calls[0]
    assert cube_call == call('unit-measure', 'pc_ind')


def test_dimension_filters_passed_on_to_query(mock_cube):
    mock_cube.get_dimension_options.return_value = {}
    ajax(mock_cube, 'dimension_options', {
        'dimension': 'ref-area',
        'time-period': '2002',
        'indicator': 'h_iacc',
        'rev': '123',
    })
    cube_call = mock_cube.get_dimension_options.mock_calls[0]
    assert cube_call == call('ref-area', [('indicator', 'h_iacc'),
                                          ('time-period', '2002')])


def test_dimension_xy_filters_passed_on_to_query(mock_cube):
    mock_cube.get_dimension_options_xy.return_value = ['something']
    res = ajax(mock_cube, 'dimension_options_xy', {
        'dimension': 'ref-area',
        'time-period': '2002',
        'breakdown': 'blahblah',
        'x-indicator': 'i_iuse',
        'y-indicator': 'i_iu3g',
        'x-__dataset': 'dataset1',
        'y-__dataset': 'dataset2',
        'rev': '123',
    })
    assert mock_cube.get_dimension_options_xy.mock_calls[0] == call(
        'ref-area',
        [('breakdown', 'blahblah'), ('time-period', '2002')],
        [('indicator', 'i_iuse')],
        [('indicator', 'i_iu3g')], 'dataset1', 'dataset2')
    assert res == {'options': ['something']}


def test_data_query_sends_filters(mock_cube):
    ajax(mock_cube, 'datapoints', {
        'indicator': 'i_bfeu',
        'breakdown': 'IND_TOTAL',
        'unit-measure': 'pc_ind',
        'rev': '123',
    })
    cube_call = mock_cube.get_observations.mock_calls[0]
    assert cube_call == call(filters=[('breakdown', 'IND_TOTAL'),
                                      ('indicator', 'i_bfeu'),
                                      ('unit-measure', 'pc_ind')])


def test_data_query_returns_rows(mock_cube):
    rows = [{'time-period': '2011', 'ref-area': 'IE', 'value': 0.2222},
            {'time-period': '2010', 'ref-area': 'PT', 'value': 0.0609}]
    mock_cube.get_observations.return_value = iter(rows)
    res = ajax(mock_cube, 'datapoints', {
        'indicator': 'i_bfeu',
        'breakdown': 'IND_TOTAL',
        'unit-measure': 'pc_ind',
    })
    cube_call = mock_cube.get_observations.mock_calls[0]
    assert cube_call == call(filters=[('breakdown', 'IND_TOTAL'),
                                      ('indicator', 'i_bfeu'),
                                      ('unit-measure', 'pc_ind')])
    assert res == {'datapoints': rows}


def test_data_xy_query_sends_filters(mock_cube):
    mock_cube.get_data_xy.return_value = ['something']
    res = ajax(mock_cube, 'datapoints_xy', {
        'x-indicator': 'i_iuse',
        'y-indicator': 'i_iu3g',
        'unit-measure': 'pc_ind',
        'breakdown': 'IND_TOTAL',
        'join_by': 'ref-area',
        'rev': '123',
    })
    cube_call = mock_cube.get_data_xy.mock_calls[0]
    assert cube_call == call(filters=[('breakdown', 'IND_TOTAL'),
                                      ('unit-measure', 'pc_ind')],
                             join_by='ref-area',
                             x_filters=[('indicator', 'i_iuse')],
                             y_filters=[('indicator', 'i_iu3g')])
    assert res == {'datapoints': ['something']}


from edw.datacube.browser import query
@patch.object(query, 'queryMultiAdapter')
def test_datapoints_cpt_blacklist_filtering(mock_adapter, mock_cube):
    datapoints = [
        {
            "breakdown": {
                "short-label": "All enterprises",
                "notation": "ent_all_xfin",
                "label": "All enterprises"
            },
            "indicator": {
                "short-label": "Use of eGovernment services - enterprises",
                "notation": "e_igov",
                "label": "Enterprises interacting online with public authorities",
                "inner_order": 1
            },
            "time-period": {
                "short-label": "2004",
                "notation": "2004",
                "label": "Year:2004"
            },
            "value": 0.8388,
            "note": None,
            "flag": None,
            "unit-measure": {
                "short-label": "% of enterprises",
                "notation": "pc_ent",
                "label": "Percentage of enterprises"
            },
            "ref-area": {
                "short-label": None,
                "notation": "EE",
                "label": "Estonia"
            }
        }
    ]

    form = {
        'indicator-group': 'egovernment',
        'ref-area': 'EE',
        'time-period': '2012',
        'indicator': ["e_igov","e_igov2pr","e_igovrt","i_igov12rt","i_iugov12"]
    }

    from edw.datacube.browser.query import AjaxDataView
    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    view = AjaxDataView(datasource, Mock(form=form))
    mock_cube.get_observations_cp = Mock(return_value=datapoints)
    mock_adapter.return_value = Mock(
        whitelist=[
            {'indicator-group': 'egovernment',
             'breakdown': 'ent_all_xfin',
             'indicator': 'e_igov',
             'unit-measure': 'pc_ent'}],
        eu = {'EE': 'Estonia'}
    )

    res = json.loads(view.datapoints_cpt())
    assert 'e_igov,ent_all_xfin,pc_ent' in res['datapoints']['table'].keys()

    #whitelist filtering is now done in sparql
    #datapoints[0]['breakdown']['notation'] = 'blacklisted'
    #view.datapoints = Mock(return_value=json.dumps(datapoints))
    #res = json.loads(view.datapoints_cpt())
    #assert 'e_igov,blacklisted,pc_ent' not in res['datapoints']['table'].keys()


from edw.datacube.browser import query
@patch.object(query, 'queryMultiAdapter')
def test_datapoints_cpc_latest(mock_adapter, mock_cube):
    datapoints = [
        {
            "breakdown": {
                "short-label": "All enterprises",
                "notation": "ent_all_xfin",
                "label": "All enterprises"
            },
            "indicator": {
                "short-label": "Use of eGovernment services - enterprises",
                "notation": "e_igov",
                "label": "Enterprises interacting online with public authorities",
                "inner_order": 1
            },
            "time-period": {
                "short-label": "2013-Q4",
                "notation": "2013-Q4",
                "label": "Quarter:2013-Q4"
            },
            "value": 0.8388,
            "note": None,
            "flag": None,
            "unit-measure": {
                "short-label": "% of enterprises",
                "notation": "pc_ent",
                "label": "Percentage of enterprises"
            },
            "ref-area": {
                "short-label": None,
                "notation": "EE",
                "label": "Estonia"
            }
        },{
            "breakdown": {
                "short-label": "All enterprises",
                "notation": "ent_all_xfin",
                "label": "All enterprises"
            },
            "indicator": {
                "short-label": "Use of eGovernment services - enterprises",
                "notation": "e_igov",
                "label": "Enterprises interacting online with public authorities",
                "inner_order": 1
            },
            "time-period": {
                "short-label": "2013-Q2",
                "notation": "2013-Q2",
                "label": "Quarter:2013-Q2"
            },
            "value": 0.8388,
            "note": None,
            "flag": None,
            "unit-measure": {
                "short-label": "% of enterprises",
                "notation": "pc_ent",
                "label": "Percentage of enterprises"
            },
            "ref-area": {
                "short-label": None,
                "notation": "EE",
                "label": "Estonia"
            }
        }
    ]

    form = {
        'indicator-group': 'egovernment',
        'ref-area': 'EE',
        'time-period': '2012',
        'indicator': ["e_igov","e_igov2pr","e_igovrt","i_igov12rt","i_iugov12"]
    }

    from edw.datacube.browser.query import AjaxDataView
    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    view = AjaxDataView(datasource, Mock(form=form))
    mock_cube.get_observations_cp = Mock(return_value=datapoints)
    mock_adapter.return_value = Mock(
        whitelist=[
            {'indicator-group': 'egovernment',
             'breakdown': 'ent_all_xfin',
             'indicator': 'e_igov',
             'unit-measure': 'pc_ent'}],
        eu = {'EE': 'Estonia'}
    )
    res = json.loads(view.datapoints_cpc())
    assert len(res['datapoints']) == 1
    assert res['datapoints'][0]['time-period']['notation'] == "2013-Q4"
