from .base import sparql_test, create_cube
from edw.datacube.data.cube import Cube


@sparql_test
def test_get_group_dimensions():
    cube = create_cube()
    res = cube.get_group_dimensions()
    assert res == ['breakdown-group', 'indicator-group']


@sparql_test
def test_unit_measure_labels_query():
    cube = create_cube()
    [res] = cube.get_dimension_labels(dimension='unit-measure', value='pc_ind')
    expected = {'short_label': '% of individuals', 'label': 'Percentage of individuals', 'notation': 'pc_ind'}
    assert res['short_label'] == expected['short_label']
    assert res['label'].startswith(expected['label'])
    assert res['notation'] == expected['notation']


@sparql_test
def test_indicator_labels_query():
    cube = create_cube()
    [res] = cube.get_dimension_labels(dimension='indicator', value='i_iusnet')
    expected = {
        'short_label': "Participating in social networks",
        'label': "participating in social networks, over the internet"}
    assert res['short_label'] == expected['short_label']
    assert res['label'].startswith(expected['label'])


@sparql_test
def test_period_labels_query():
    cube = create_cube()
    [res] = cube.get_dimension_labels(dimension='time-period', value='2006')
    expected = {
        'short_label': '2006',
        'label': 'Year:2006'}
    assert expected['short_label'] == res['short_label']
    assert expected['label'] == res['label']


@sparql_test
def test_get_all_country_options():
    cube = create_cube()
    items = cube.get_dimension_options('ref-area')
    codes = [y['notation'] for y in items]
    assert len(codes) >= 50
    assert 'DE' in codes
    assert 'ES' in codes
    assert 'BG' in codes
    assert 'EU27' in codes
    assert 'EU28' in codes


@sparql_test
def test_get_available_country_options_for_year():
    cube = create_cube()
    items = cube.get_dimension_options('ref-area', [
        ('time-period', '2006'),
        ('indicator', 'e_igovrt'),
    ])
    codes = [y['notation'] for y in items]
    print codes
    assert len(codes) == 30
    assert 'EL' in codes
    assert 'EU27' in codes
    assert 'HR' not in codes


@sparql_test
def test_group_notations(monkeypatch):
    cube = create_cube()
    items = cube.get_dimension_options('indicator', [
        ('indicator-group', 'ict-skills'),
    ])
    assert items[0]['group_notation'] == 'ict-skills'


@sparql_test
def test_get_available_country_options_for_year_and_indicator():
    cube = create_cube()
    items = cube.get_dimension_options('ref-area', [
        ('time-period', '2002'),
        ('indicator', 'h_iacc'),
    ])
    codes = [y['notation'] for y in items]
    assert len(codes) == 19
    assert 'DE' in codes
    assert 'ES' not in codes
    assert 'BG' not in codes
    assert 'EU27' not in codes


@sparql_test
def test_get_available_indicator_group_options():
    cube = create_cube()
    items = cube.get_dimension_options('indicator-group')
    codes = [y['notation'] for y in items]
    assert len(codes) >= 15
    assert 'internet-usage' in codes
    assert 'ebusiness' in codes


@sparql_test
def test_get_available_indicator_group_options_for_year_and_country():
    cube = create_cube()
    items = cube.get_dimension_options('indicator-group', [
        ('time-period', '2002'),
        ('ref-area', 'DK'),
    ])
    codes = [y['notation'] for y in items]
    assert len(codes) == 6
    assert 'internet-usage' in codes
    assert 'ebusiness' not in codes


@sparql_test
def test_get_available_year_options_for_indicator_group():
    cube = create_cube()
    items = cube.get_dimension_options('time-period', [
        ('indicator-group', 'mobile'),
    ])
    years = [y['notation'] for y in items]
    assert len(years) >= 8
    assert '2012' in years
    assert '2002' not in years


@sparql_test
def test_get_indicators_in_group():
    cube = create_cube()
    items = cube.get_dimension_options('indicator', [
        ('indicator-group', 'ict-skills'),
    ])
    indicators = [i['notation'] for i in items]
    assert len(indicators) >= 15
    assert 'i_skedu' in indicators
    assert 'P_IUSE' in indicators
    assert 'e_broad' not in indicators


@sparql_test
def test_get_altlabel_for_group_dimension():
    cube = create_cube()
    items = cube.get_dimension_options('breakdown-group')
    label = [it['short_label'] for it in items
                if it['notation'] == 'byage3classes'
            ][0]
    assert u'Age (3 groups)' == label


@sparql_test
def test_get_altlabel_for_not_group_dimension():
    cube = create_cube()
    items = cube.get_dimension_options('unit-measure')
    label = [it['short_label'] for it in items
                if it['notation'] == 'pc_ind'
            ][0]
    assert u'% of individuals' == label


@sparql_test
def test_get_years_for_xy_indicators():
    cube = create_cube()
    items = cube.get_dimension_options_xy('time-period',
        [('ref-area', 'ES')],
        [('indicator', 'i_iuse')],
        [('indicator', 'i_iu3g')],
    )
    years = [i['notation'] for i in items]
    assert len(years) == 9
    assert '2015' in years
    assert '2010' in years
    assert '2012' in years
    assert '2006' not in years
    assert '2004' not in years


@sparql_test
def test_get_years_for_xyz_indicators():
    cube = create_cube()
    items = cube.get_dimension_options_xyz('time-period',
        [('ref-area', 'ES')],
        [('indicator', 'bb_dsl')],
        [('indicator', 'bb_penet')],
        [('indicator', 'bb_ne')]
    )
    years = [i['notation'] for i in items]
    assert len(years) >= 18
    assert '2007-12' in years
    assert '2012-06' in years
    assert '2012' not in years


@sparql_test
def test_get_indicator_metadata():
    cube = create_cube()
    res = cube.get_dimension_option_metadata('indicator', 'i_iuse')
    assert res['label'].startswith("Individuals who are regular")
    assert res['short_label'] == "Regular internet users"
    assert res['definition'] == ("Individuals using the internet at least "
                                 "once a week in the last 3 months.")


@sparql_test
def test_get_breakdown_metadata():
    cube = create_cube()
    res = cube.get_dimension_option_metadata('breakdown', 'ent_sm_xfin')
    assert res['label'] == "SMEs (10-249 persons employed), without financial sector"
    assert res['short_label'] == "SMEs (10-249 persons employed)"
    assert res['definition'] == "Since 2010: SM_C10_S951_XK, 2009: SM_C10_N82_XK, until 2008: sm_dfghiko"
    assert 'note' not in res
    assert 'source_label' not in res


@sparql_test
def test_get_indicator_source_metadata():
    cube = create_cube()
    res = cube.get_dimension_option_metadata('indicator', 'i_iuse')
    assert res['label'] == "Individuals who are regular internet users (at least once a week)"
    assert res['source_label'] == "Eurostat - ICT Households survey"
    assert res['source_definition'] == (
        "Eurostat - Community survey on ICT usage in Households and by Individuals")
    #assert res['source_notes'] == (
    #    u"Extraction from HH/Indiv comprehensive database (ACCESS) version\xa0April 2014")
    assert res['source_url'] == (
        "http://ec.europa.eu/eurostat/web/information-society/data/comprehensive-database")

#@sparql_test
#def test_dump_has_output():
#    cube = create_cube()
#    res = cube.dump()
#    assert(res.next())
#
#@sparql_test
#def test_dump_row_fields():
#    cube = create_cube()
#    res = cube.dump()
#    expected = set([
#        'unit_measure',
#        'indicator',
#        'time_period',
#        'value',
#        'ref_area',
#        'breakdown'])
#    assert expected.difference(set(res.next().keys())) == set([])


@sparql_test
def test_get_labels():
    import sparql
    cube = create_cube()
    data = [('http://reference.data.gov.uk/id/gregorian-year/2007', 'time-period'),
                ('http://reference.data.gov.uk/id/gregorian-year/2009', 'time-period')]
    res = cube.get_labels(data)
    assert sorted(res.keys()) == map(lambda item: item[0], data)
    assert res[data[1][0]]['notation'] == '2009'
    assert res[data[1][0]]['short_label'] == '2009'


@sparql_test
def test_indicator_groups_are_sorted():
    cube = create_cube()
    res = cube.get_dimension_options(dimension='indicator-group')
    codes = [y['notation'] for y in res]
    assert codes == ['telecom', 'broadband', 'bbquality', 'mobile',
        'internet-usage', 'audiovisual', 'internet-services', 'egovernment', 'ecommerce',
        'ebusiness', 'ict-skills', 'ict-specialist', 'ict-edu', 'eHealth',
        'security-privacy', 'ict-sector', 'research-and-development',
        'back', 'discontinued']

@sparql_test
def test_dimension_options_sort_bug():
    #hardcoded endpoint and uri
    cube = Cube('http://test-virtuoso.digital-agenda-data.eu/sparql',
        'http://semantic.digital-agenda-data.eu/dataset/CNECT_Dashboard')
    res = cube.get_dimension_options('breakdown', [
        ('breakdown-group', 'byimplementation+status'),
        ('indicator', 'DAE_MS_Actions'),
        ('time-period', '2013'),
        ('unit-measure', 'nbr_actions')
    ])
    #should not fail with AttributeError: 'dict' object has no attribute 'sort'
    assert res == {}


@sparql_test
def test_dimension_options_in_multiple_groups_filtered():
    #hardcoded endpoint and uri
    cube = create_cube()
    res = cube.get_dimension_options('breakdown', [
        ('breakdown-group', 'byage3classes'),
        ('indicator', 'i_iueduif'),
        ('time-period', '2009'),
        ('unit-measure', 'pc_ind')
    ])
    #should return three unique items filtered by breakdown-group
    codes = [y['notation'] for y in res]
    assert len(codes) == len(set(codes))
    groups = set([y['group_notation'] for y in res])
    assert len(groups) == 1
    assert 'byage3classes' == groups.pop()

@sparql_test
def test_dimension_options_in_multiple_groups_unfiltered():
    #hardcoded endpoint and uri
    cube = create_cube()
    res = cube.get_dimension_options('breakdown', [
        ('indicator', 'i_iueduif')
    ])
    #should return Y16_24 twice, in different breakdown groups
    breakdowns = [y for y in res if y['notation'] == 'Y16_24']
    assert len(breakdowns) == 2
    groups = set([y['group_notation'] for y in breakdowns])
    assert groups == set(['byage3classes', 'byage6classes'])

@sparql_test
def test_dimension_options_xy_none_common():
    cube = create_cube()

    res = cube.get_dimension_options_xy('time-period', [],
         [('indicator', 'mbb_3gcov'),
          ('breakdown', 'TOTAL_POP'),
          ('unit-measure', 'pc_pop')
         ],
         [
          ('indicator', 'TOTAL_POPHH'),
          ('breakdown', 'mbb_ltecov'),
          ('unit-measure', 'pc_hh_all')
         ]
    );
    # should return no result
    assert res == {}

@sparql_test
def test_dimension_options_xy_time_periods():
    cube = create_cube()

    res = cube.get_dimension_options_xy('time-period', [],
         [('indicator', 'bb_hhi'),
          ('breakdown', 'TOTAL_FBB'),
          ('unit-measure', 'hhi')
         ],
         [
          ('indicator', 'bb_scov'),
          ('breakdown', 'TOTAL_POPHH'),
          ('unit-measure', 'pc_hh_all')
         ]
    );
    # should return only 2013
    # the intersection of [2013-06,2014-06] and [2011,2012,2013,2014]
    codes = [y['notation'] for y in res]
    assert '2013' in codes
    assert '2014' in codes
    assert '2012' not in codes
    assert '2013-06' not in codes

@sparql_test
def test_dimension_options_xy_time_periods2():
    cube = create_cube()

    res = cube.get_dimension_options_xy('time-period', [],
         [('indicator', 'bb_hhi'),
          ('breakdown', 'TOTAL_FBB'),
          ('unit-measure', 'hhi')
         ],
         [
          ('indicator', 'AAAA_cov'),
          ('breakdown', '1M_websites'),
          ('unit-measure', 'pc_websites')
         ]
    );
    # should return only 2013
    # the intersection of [2013-06,2014-06] and [2012-Q2, 2012-Q3, 2013-Q1..Q4, 2014-Q4]
    codes = [y['notation'] for y in res]
    assert '2013' in codes
    assert '2014' in codes
    assert '2012' not in codes
    assert '2013-Q1' not in codes

@sparql_test
def test_dimension_codelist():
    cube = create_cube()
    res = cube.get_dimension_codelist('indicator')
    codes = [ (y['notation'], y['uri']) for y in res]
    assert ('i_iuse', 'http://semantic.digital-agenda-data.eu/codelist/indicator/i_iuse') in codes

@sparql_test
def test_patch_codelist():
    cube = create_cube()
    res = cube.get_dimension_metadata()
    codes = [ (y['notation'], y.get('label'), y['uri']) for y in res['breakdown-group']]
    assert ('byage6classes', 'Age (6 groups)', 'http://semantic.digital-agenda-data.eu/codelist/breakdown-group/byage6classes') in codes

    codes = [ (y['notation'], y.get('label'), y['uri']) for y in res['time-period']]
    assert ('2009-Q3', 'Quarter:2009-Q3', 'http://reference.data.gov.uk/id/gregorian-quarter/2009-Q3') in codes

    codes = [ (y['notation'], y.get('label'), y['uri']) for y in res['ref-area']]
    assert ('RO', 'Romania', 'http://eurostat.linked-statistics.org/dic/geo#RO') in codes
    assert ('ASS', 'ASS', 'http://eurostat.linked-statistics.org/dic/geo#ASS') in codes

@sparql_test
def test_get_dimension_options_raw():
    cube = create_cube()
    res = cube.get_dimension_options_raw('ref-area')
    assert 'http://eurostat.linked-statistics.org/dic/geo#RO' in res
    assert 'http://eurostat.linked-statistics.org/dic/geo#ASS' in res
    res = cube.get_dimension_options_raw('time-period')
    assert 'http://reference.data.gov.uk/id/gregorian-quarter/2009-Q3' in res

