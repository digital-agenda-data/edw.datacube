# -*- coding: utf-8 -*-
from mock import Mock, MagicMock, call, patch
import simplejson as json
import pytest
import copy
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

def get_datapoints_cp():
    obs = {
            "breakdown": {
                "short-label": "All enterprises",
                "notation": "ent_all_xfin",
                "label": "All enterprises"
            },
            "indicator": {
                "short-label": "Enterprises with a fixed broadband connection",
                "notation": "e_broad",
                "label": "Enterprises with a fixed broadband connection",
                "inner_order": 1
            },
            "time-period": {
                "short-label": "2015",
                "notation": "2015",
                "label": "Year:2015"
            },
            "value": 0.926255,
            "note": None,
            "flag": None,
            "unit-measure": {
                "short-label": "% of enterprises",
                "notation": "pc_ent",
                "label": "Percentage of enterprises"
            },
            "ref-area": {
                "short-label": None,
                "notation": "EU27",
                "label": "EU27"
            }
        }
    obs2 = copy.deepcopy(obs)
    obs2['ref-area']['notation'] = 'EU28'
    obs2['value'] = 0.925897

    obs3 = copy.deepcopy(obs)
    obs3['ref-area']['notation'] = 'RO'
    obs3['value'] = 0.55

    return [obs, obs2, obs3]

from edw.datacube.browser import query
from edw.datacube.browser.query import AjaxDataView

@patch.object(query, 'queryMultiAdapter')
def test_datapoints_cpc_EU28(mock_adapter, mock_cube):

    datapoints = get_datapoints_cp()
    form = {
        'indicator-group': 'broadband',
        'ref-area': 'RO',
        'time-period': '2015',
        'indicator': ["e_broad"],
        'subtype': 'chart'
    }

    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    view = AjaxDataView(datasource, Mock(form=form))
    mock_cube.get_observations_cp = Mock(return_value=datapoints)
    mock_adapter.return_value = Mock(
        whitelist=[
            {'indicator-group': 'broadband',
             'breakdown': 'ent_all_xfin',
             'indicator': 'e_broad',
             'unit-measure': 'pc_ent'}],
        eu = {'RO': 'Romania'}
    )
    res = json.loads(view.datapoints_cpc())
    assert len(res['datapoints']) == 3
    assert res['datapoints'][0]['eu'] == 0.925897

@patch.object(query, 'queryMultiAdapter')
def test_datapoints_cpt_EU28(mock_adapter, mock_cube):

    datapoints = get_datapoints_cp()
    form = {
        'indicator-group': 'broadband',
        'ref-area': 'RO',
        'time-period': '2015',
        'indicator': ["e_broad"],
        'subtype': 'table'
    }

    datasource = Mock(get_cube=Mock(return_value=mock_cube))
    view = AjaxDataView(datasource, Mock(form=form))
    mock_cube.get_observations_cp = Mock(return_value=datapoints)
    mock_adapter.return_value = Mock(
        whitelist=[
            {'indicator-group': 'broadband',
             'breakdown': 'ent_all_xfin',
             'indicator': 'e_broad',
             'unit-measure': 'pc_ent'}],
        eu = {'RO': 'Romania'}
    )
    res = json.loads(view.datapoints_cpt())
    assert len(res['datapoints']) == 4
    assert len(res['datapoints']['table']) == 1
    assert res['datapoints']['latest'] == 2015
    assert res['datapoints']['ref-area']['notation'] == 'RO'
    assert res['datapoints']['table']['e_broad,ent_all_xfin,pc_ent']['eu'] == 0.925897
