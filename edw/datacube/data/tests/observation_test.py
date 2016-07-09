from .base import sparql_test, create_cube


@sparql_test
def test_get_same_observation_in_two_dimensions():
    filters = [
        ('time-period', '2011'),
        ('indicator', 'i_bfeu'),
        ('breakdown', 'IND_TOTAL'),
        ('unit-measure', 'pc_ind'),
        ('ref-area', 'IE'),
    ]
    cube = create_cube()
    points = list(cube.get_data_xy('ref-area', filters, [], []))
    assert len(points) == 1
    assert points[0]['value'] == {'x': 0.222202, 'y': 0.222202}

@sparql_test
def test_get_observations_with_labels_xy():
    filters = [
        ('time-period', '2011'),
        ('indicator', 'i_bfeu'),
        ('breakdown', 'IND_TOTAL'),
        ('unit-measure', 'pc_ind'),
    ]
    cube = create_cube()
    points = list(cube.get_data_xy('ref-area', filters, [], []))
    assert points[0]['indicator']['label'].startswith(
        'Individuals ordering goods')


@sparql_test
def test_get_observations_with_notes_single_dimension():
    filters = [
        ('indicator-group', 'broadband'),
        ('indicator', 'h_bbfix'),
        ('breakdown-group', 'total'),
        ('breakdown', 'HH_total',),
        ('unit-measure', 'pc_hh'),
        ('time-period', '2010'),
    ]
    cube = create_cube()
    points = list(cube.get_observations(filters))
    assert filter(
      lambda item: item['ref-area']['notation'] == 'LT',
      points)[0]['note'] == 'Sum of households in quartiles less than total.348 households refused to answer question about household income.'


@sparql_test
def test_get_observations_with_notes_multidimension():
    filters = [
        ('indicator-group', 'ecommerce'),
        ('indicator', 'i_bgoodo'),
        ('breakdown-group', 'total'),
        ('breakdown', 'IND_TOTAL',),
        ('unit-measure', 'pc_ind'),
        ('time-period', '2006'),
    ]
    cube = create_cube()
    points = list(cube.get_data_xyz('ref-area', filters, [], [], []))
    note='Estimation - based on results of 2007 HH survey';
    assert filter(
      lambda item: item['ref-area']['notation'] == 'CZ',
      points)[0]['note'] == {'x': note, 'y':note, 'z':note}


@sparql_test
def test_get_xy_observations_with_all_breakdowns():
    filters = [
        ('time-period', '2011'),
        ('indicator', 'i_bfeu'),
        ('unit-measure', 'pc_ind'),
    ]
    x_filters = [
        ('breakdown', 'IND_TOTAL'),
    ]
    y_filters = [
        ('breakdown', 'IND_TOTAL'),
    ]
    cube = create_cube()
    points = list(cube.get_data_xy('ref-area', filters, x_filters, y_filters))
    assert points[0]['indicator']['label'].startswith(
        'Individuals ordering goods')


@sparql_test
def test_get_same_observation_in_xyz_dimensions():
    filters = [
        ('time-period', '2011'),
        ('indicator', 'i_bfeu'),
        ('breakdown', 'IND_TOTAL'),
        ('unit-measure', 'pc_ind'),
        ('ref-area', 'IE'),
    ]
    cube = create_cube()
    points = list(cube.get_data_xyz('ref-area', filters, [], [], []))
    assert len(points) == 1
    assert points[0]['value'] == {'x': 0.222202, 'y': 0.222202, 'z': 0.222202}


@sparql_test
def test_get_xy_observations_for_2_countries_all_years():
    filters = [
        ('indicator', 'i_bfeu'),
        ('breakdown', 'IND_TOTAL'),
        ('unit-measure', 'pc_ind'),
    ]
    x_filters = [('ref-area', 'IE')]
    y_filters = [('ref-area', 'DK')]
    cube = create_cube()
    pts = list(cube.get_data_xy('time-period', filters, x_filters, y_filters))
    assert len(pts) >= 6
    assert filter(
               lambda item: item['time-period']['notation'] == '2011',
               pts)[0]['value'] == {'x': 0.222202, 'y': 0.279457}
    assert filter(
               lambda item: item['time-period']['notation'] == '2012',
               pts)[0]['value'] == {'x': 0.281144, 'y': 0.289237}


@sparql_test
def test_get_xyz_observations_for_3_countries_all_years():
    filters = [
        ('indicator', 'i_bfeu'),
        ('breakdown', 'IND_TOTAL'),
        ('unit-measure', 'pc_ind'),
    ]
    x_filters = [('ref-area', 'IE')]
    y_filters = [('ref-area', 'DK')]
    z_filters = [('ref-area', 'AT')]
    cube = create_cube()
    pts = list(cube.get_data_xyz('time-period', filters, x_filters, y_filters, z_filters))
    assert len(pts) >= 5
    assert filter(
               lambda item: item['time-period']['notation'] == '2008',
               pts)[0]['value'] == {'x': 0.170733, 'y': 0.197577, 'z': 0.244686}


@sparql_test
def test_get_observations_with_all_attributes():
    cube = create_cube()
    filters = [ ('breakdown', 'TOTAL_FBB'),
                ('indicator', 'bb_dsl'),
                ('indicator-group', 'broadband'),
                ('ref-area', 'EU28'),
                ('unit-measure', 'pc_lines')]
    result = list(cube.get_observations(filters))
    assert len(result) >= 18
    assert result[0]['value'] is not None
    assert result[0]['indicator']['label'].startswith('DSL subscriptions share in fixed broadband')
    assert result[0]['indicator']['short-label'].startswith('DSL subscr')

@sparql_test
def test_get_observations_cp():
    cube = create_cube()
    filters = [ ('indicator-group', 'internet-usage'),
                ('ref-area', 'CY'),
                ('time-period', '2013')
              ]
    whitelist=[
        {'indicator-group': 'internet-usage', 'indicator': 'h_iacc', 'breakdown': 'HH_total', 'unit-measure': 'pc_hh'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_ia12ave', 'breakdown': 'Y16_24', 'unit-measure': 'ia12ave'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_ia12ave', 'breakdown': 'Y25_54', 'unit-measure': 'ia12ave'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_ia12ave', 'breakdown': 'Y55_74', 'unit-measure': 'ia12ave'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_iday', 'breakdown': 'IND_TOTAL', 'unit-measure': 'pc_ind'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_iumc', 'breakdown': 'IND_TOTAL', 'unit-measure': 'pc_ind'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_iuse', 'breakdown': 'IND_TOTAL', 'unit-measure': 'pc_ind'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_iuse', 'breakdown': 'rf_ge1', 'unit-measure': 'pc_ind'},
        {'indicator-group': 'internet-usage', 'indicator': 'i_iux', 'breakdown': 'IND_TOTAL', 'unit-measure': 'pc_ind'}
    ]
    result = list(cube.get_observations_cp(filters, whitelist))
    assert len(result) == 9
    h_iacc = filter(lambda item: item['indicator']['notation'] == 'h_iacc', result)[0]
    assert h_iacc['value'] == 0.647121
    assert h_iacc['indicator']['label'].startswith('Households with access to the Internet at home')
    assert h_iacc['indicator']['short-label'].startswith('Households with access to the Internet at home')
    assert len(filter(lambda item: item['indicator']['notation'] == 'i_ia12ave', result)) == 3

@sparql_test
def test_get_observations_cp2():
    # Should not raise IndexError: list index out of range becuase of lowercase IND_TOTAL
    cube = create_cube()
    filters = [ ('indicator-group', 'eHealth'),
                ('ref-area', 'BE'),
                ('time-period', '2014')
              ]
    whitelist=[
        {'indicator-group': 'ehealth', 'indicator': 'I_IUMAPP', 'breakdown': 'ind_total', 'unit-measure': 'pc_ind_iu3'},
        {'indicator-group': 'ehealth', 'indicator': 'i_ihif', 'breakdown': 'ind_total', 'unit-measure': 'pc_ind_iu3'}
    ]
    result = list(cube.get_observations_cp(filters, whitelist))
    assert len(result) == 1
    i_ihif = filter(lambda item: item['indicator']['notation'] == 'I_IUMAPP', result)[0]
    assert i_ihif['value'] == 0.218317
    assert i_ihif['indicator']['label'].startswith('making an appointment with a practitioner via a website')
    assert i_ihif['indicator']['short-label'].startswith('Appointment with a practitioner via a website')
