""" Query
"""
import csv
import json
import logging
from StringIO import StringIO
from datetime import datetime
from zope.component import queryMultiAdapter
from Products.Five.browser import BrowserView
from eea.cache import cache as eeacache
from DateTime import DateTime

logger = logging.getLogger('edw.datacube')

def json_response(func):
    def _decorator(self, *args, **kwargs):
        header = self.request.RESPONSE.setHeader
        header("Content-Type", "application/json")
        header("Expires", "Sun, 17-Jan-2038 19:14:07 GMT")
        return func(self, *args, **kwargs)
    return _decorator

def jsonify(request, data, cache=True):
    header = request.RESPONSE.setHeader
    header("Content-Type", "application/json")
    if cache:
        header("Expires", "Sun, 17-Jan-2038 19:14:07 GMT")
    return json.dumps(data, indent=2, sort_keys=True)

def cacheKey(method, self, *args, **kwargs):
    """ Generate unique cache id
    """
    return (self.context.absolute_url(1), dict(self.request.form))

def cacheKey_cp(method, self, *args, **kwargs):
    """ Generate unique cache id for country profiles
    """
    view = queryMultiAdapter((self.context, self.request),
                             name=u'whitelist.json')
    timestamp = view.contextualWlTimestamp() if view else DateTime.ISO()

    return (self.context.absolute_url(1), dict(self.request.form), timestamp)

class AjaxDataView(BrowserView):

    def __init__(self, context, request):
        super(AjaxDataView, self).__init__(context, request)
        self.endpoint = self.request.get('endpoint', '')
        self.cube = context.get_cube(self.endpoint)

    def jsonify(self, *args, **kwargs):
        return jsonify(self.request, *args, **kwargs)

    def datasets(self):
        return self.jsonify({'datasets': self.cube.get_datasets()},
                            cache=False)

    def dataset_metadata(self):
        dataset = self.request.form['dataset']
        return self.jsonify(self.cube.get_dataset_metadata(dataset))

    def dataset_details(self):
        return self.jsonify(self.cube.get_dataset_details())

    def get_dimensions(self):
        flat = bool(self.request.form.get('flat'))
        dimensions = self.cube.get_dimensions(flat=flat)
        return self.jsonify(dimensions)

    def revision(self):
        form = dict(self.request.form)
        revision = self.cube.get_revision()
        return self.jsonify(revision)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_labels(self):
        form = dict(self.request.form)
        dimension = form.pop('dimension')
        value = form.pop('value')
        [labels] = self.cube.get_dimension_labels(dimension, value)
        return self.jsonify(labels)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_codelist(self):
        form = dict(self.request.form)
        dimension = form.pop('dimension')
        res = self.cube.get_dimension_codelist(dimension)
        return self.jsonify(res)

    def notations(self):
        form = dict(self.request.form)
        res = self.cube.notations.get()
        return self.jsonify(res)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        filters = sorted(form.items())
        options = self.cube.get_dimension_options(dimension, filters)
        filtered_options = filter(lambda it: it['notation'] != "", options)
        return self.jsonify({'options': filtered_options})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options_xy(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        x_dataset = form.pop('x-__dataset', '')
        y_dataset = form.pop('y-__dataset', '')

        (filters, x_filters, y_filters) = ([], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            else:
                filters.append((k, v))

        options = self.cube.get_dimension_options_xy(dimension, filters,
                                                     x_filters, y_filters,
                                                     x_dataset, y_dataset)
        return self.jsonify({'options': options})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options_xyz(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        (filters, x_filters, y_filters, z_filters) = ([], [], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            elif k.startswith('z-'):
                z_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        options = self.cube.get_dimension_options_xyz(dimension, filters,
                                                      x_filters, y_filters,
                                                      z_filters)
        return self.jsonify({'options': options})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options_cp(self):
        subtype = self.request.form.pop('subtype', 'table')
        options = self.dimension_options()

        if self.request.form.get('dimension', '') != 'ref-area':
            return options

        if subtype == u'table':
            return options

        # Get EU countries
        view = queryMultiAdapter((self.context, self.request),
                                 name=u'european-union.json')
        eu = view.eu if view else {}

        options = json.loads(options)

        rows = []
        for option in options.get('options', []):
            if subtype == u'polar' and option.get('notation') == 'EU27':
                rows.append(option)
                continue

            if option.get('notation', '') not in eu and option.get('notation', '') not in ['NO', 'IS']:
                continue
            rows.append(option)
        return self.jsonify({'options': rows})


    @json_response
    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_value_metadata(self):
        dimension = self.request.form['dimension']
        value = self.request.form['value']
        res = self.cube.get_dimension_option_metadata(dimension, value)
        return self.jsonify(res)

    def country_value(self, uid, datapoints):
        """ Get value for given uid
        """
        for point in datapoints:
            key = (
                point['indicator']['notation'],
                point['breakdown']['notation'],
                point['unit-measure']['notation'],
                point['time-period']['notation']
            )
            if uid == key:
                res = point['value']
                try:
                    res = float(res)
                except Exception, err:
                    logger.exception(err)
                    return 0
                else:
                    return res
        return 0

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        filters = sorted(form.items())
        rows = list(self.cube.get_observations(filters=filters))
        return self.jsonify({'datapoints': rows})

    @eeacache(cacheKey_cp, dependencies=['edw.datacube'])
    def datapoints_cp(self):
        """ Datapoints for country profile chart
        """
        subtype = self.request.form.pop('subtype', 'table')
        if subtype == 'table':
            return self.datapoints_cpt()
        else:
            return self.datapoints_cpc()

    @eeacache(cacheKey_cp, dependencies=['edw.datacube'])
    def datapoints_cpc(self):
        # Get whitelisted items
        view = queryMultiAdapter((self.context, self.request),
                                 name=u'whitelist.json')

        whitelist = view.contextualWhitelist() if view else []
        # Get datapoints
        countryName = self.request.form.pop('ref-area', '')
        filters = [('indicator-group', self.request.form['indicator-group'])]
        if countryName:
            filters.append(('ref-area', countryName))
        datapoint_rows = list(self.cube.get_observations_cp(filters, whitelist))

        # Get EU countries
        view = queryMultiAdapter((self.context, self.request),
                                 name=u'european-union.json')
        eu = view.eu if view else {}

        # Get all datapoints
        all_datapoint_rows = list(self.cube.get_observations_cp([
            ('time-period', self.request.form['time-period']),
            ('indicator-group', self.request.form['indicator-group'])],
            whitelist))
        all_datapoint_rows.sort(key=lambda k: k['time-period']['notation'])

        # Compute new values
        mapping = {}
        latest = {}
        for point in all_datapoint_rows:
            key = (
                point['indicator']['notation'],
                point['breakdown']['notation'],
                point['unit-measure']['notation'],
                point['time-period']['notation']
            )

            # latest is a mapping for
            # indicator,breakdown,unit-measure to latest time-period
            latest[key[:-1]] = key[-1]

            try:
                point['value'] = float(point['value'])
            except Exception, err:
                logger.exception(err)
                continue

            countryValue = self.country_value(key, datapoint_rows)
            if not mapping.get(key):
                mapping[key] = {
                    'min': {'value': None, 'ref-area': countryName},
                    'max': {'value': None, 'ref-area': countryName},
                    'med': {'value': None, 'ref-area': 'EU27'}
                }

            # Update med
            if point['ref-area']['notation'] == 'EU27':
                mapping[key]['med']['value'] = point['value']

            # Update min / max only for EU27 countries
            if point['ref-area']['notation'] not in eu:
                continue

            # Update min
            oldValue = mapping[key]['min']['value']
            if oldValue is None:
                newValue = point['value']
            else:
                newValue = min(point['value'], oldValue)
            if oldValue is None or newValue != oldValue:
                mapping[key]['min']['value'] = newValue
                mapping[key]['min']['ref-area'] = point['ref-area']['notation']

            # Update max
            oldValue = mapping[key]['max']['value']
            newValue = max(point['value'], oldValue)
            if newValue != oldValue:
                mapping[key]['max']['value'] = newValue
                mapping[key]['max']['ref-area'] = point['ref-area']['notation'];

        # Update points
        rows = []
        for point in datapoint_rows:
            key =  (
                point['indicator']['notation'],
                point['breakdown']['notation'],
                point['unit-measure']['notation'],
                point['time-period']['notation']
            )

            # if not latest: continue
            if latest.get(key[:-1]) != key[-1]:
                continue

            try:
                point['value'] = float(point['value'])
            except Exception, err:
                logger.exception(err)
                continue

            val = point['value']
            minVal = mapping[key]['min']['value']
            maxVal = mapping[key]['max']['value']
            medVal = mapping[key]['med']['value'] or (minVal+maxVal)/2
            #rank = mapping[key]['rank']['value']

            point['original'] = val
            point['eu'] = medVal
            #point['rank'] = rank
            if val <= medVal:
                if minVal == medVal:
                    point['value'] = 0
                else:
                    point['value'] = (val - minVal) / (medVal - minVal)
            else:
                if maxVal == medVal:
                    point['value'] = 2
                else:
                    point['value'] = 1 + (val - medVal) / (maxVal - medVal)
            rows.append(point)

        return self.jsonify({'datapoints': rows})

    @eeacache(cacheKey_cp, dependencies=['edw.datacube'])
    def datapoints_cpt(self):
        """ Datapoints for country profile table

        return_json = {
          'datapoints' {
            'latest': 2012,
            'has-rank': true,
            'ref-area': {
              'label': 'Romania',
              'short-label': null,
              'notation': 'RO'
            },
            'table': {
             'indicator-1,breakdown-1,unit-measure-1': {
               'indicator': 'Indicator-1-short-label',
               'breakdown': 'breakdown-1-label',
               'unit-measure': 'unit-measure-1-short-label',
               'unit': 'unit-measure-1-notation',
               'inner_order': 'order of the indicator in the group'
               'rank': 15,
               'eu': 0.67,
               '2012': 0.55,
               '2011': 0.46,
               '2010': 0.33,
               '2009': 0.09,
               '2008': 0.01
              }
            }
          }
        }
        """

        view = queryMultiAdapter((self.context, self.request),
                                 name=u'whitelist.json')
        whitelist = view.contextualWhitelist() if view else []

        latestYear = self.request.form.pop('time-period',
                                           datetime.now().year - 1)
        try:
            latestYear = int(latestYear)
        except Exception, err:
            latestYear = datetime.now().year - 1

        # Get datapoints
        countryName = self.request.form.pop('ref-area', '')
        filters = [('indicator-group', self.request.form['indicator-group'])]
        if countryName:
            filters.append(('ref-area', countryName))
        datapoint_rows = list(self.cube.get_observations_cp(filters, whitelist))
        # sort by time-period
        datapoint_rows.sort(key=lambda k: k['time-period']['notation'])

        mapping = {
            'latest': latestYear,
            'has-rank': False,
            'ref-area': {
                'notation': countryName,
                'label': countryName,
                'short-label': None
            },
            'table': {}
        }
        table = mapping['table']
        for point in datapoint_rows:
            # Selected country
            mapping['ref-area'].update(point['ref-area'])

            # Indicator unique identifier
            key = (
                point['indicator']['notation'],
                point['breakdown']['notation'],
                point['unit-measure']['notation']
            )

            try:
                point['value'] = float(point['value'])
            except Exception, err:
                logger.exception(err)
                continue

            table.setdefault(key, {})
            if not table[key].get('indicator'):
                table[key]['indicator'] = (
                    point['indicator']['short-label']
                    or point['indicator']['label']
                    or point['indicator']['notation'])
            if not table[key].get('breakdown'):
                table[key]['breakdown'] = (
                    point['breakdown']['short-label']
                    or point['breakdown']['label']
                    or point['breakdown']['notation'])
            if not table[key].get('unit-measure'):
                table[key]['unit-measure'] = (
                    point['unit-measure']['short-label']
                    or point['unit-measure']['label']
                    or point['unit-measure']['notation'])

            if not table[key].get('inner_order'):
                table[key]['inner_order'] = point['indicator']['inner_order']

            if not table[key].get('unit'):
                table[key]['unit'] = point['unit-measure']['notation']

            year = point['time-period']['notation'].split('-')[0]

            try:
                year = int(year)
            except Exception, err:
                logger.exception(err)
                continue

            table[key][year] = point['value']
            table[key].setdefault('rank', 0)


        # Get EU countries
        view = queryMultiAdapter((self.context, self.request),
                                 name=u'european-union.json')
        eu = view.eu if view else {}

        # Get all datapoints
        all_datapoint_rows = list(self.cube.get_observations_cp([
            ('time-period', unicode(latestYear)),
            ('indicator-group', self.request.form['indicator-group'])],
            whitelist))
        # sort by time-period
        all_datapoint_rows.sort(key=lambda k: k['time-period']['notation'])
        # collapse multiple time periods within the same year
        datapoint_rows = {}
        for point in all_datapoint_rows:
            key = (
                point['indicator']['notation'],
                point['breakdown']['notation'],
                point['unit-measure']['notation'],
                point['ref-area']['notation']
            )
            try:
                datapoint_rows[key] = float(point['value'])
            except Exception, err:
                logger.exception(err)
                continue

        # Compute rank amoung EU27 countries
        for key, value in datapoint_rows.items():
            country = key[-1]
            key = key[:-1]
            if country == 'EU27':
                if table.get(key):
                    table[key]['eu'] = value

            # Skip non-EU countries from rank computation
            if country not in eu or countryName not in eu:
                continue
            myValue = table.get(key, {}).get(latestYear, None)
            if myValue is not None:
                mapping['has-rank'] = True
                if not table[key].get('rank'):
                    table[key]['rank'] = 1
                if value > myValue:
                    table[key]['rank'] += 1

        # convert table.keys from object to string

        table_new = {}
        for key in table.keys():
            key_new = u','.join((key[0], key[1], key[2]))
            table_new[key_new] = table[key]
        mapping['table'] = table_new
        return self.jsonify({'datapoints': mapping})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints_xy(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        join_by = form.pop('join_by')
        (filters, x_filters, y_filters) = ([], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        rows = list(self.cube.get_data_xy(join_by=join_by,
                                          filters=filters,
                                          x_filters=x_filters,
                                          y_filters=y_filters))
        return self.jsonify({'datapoints': rows})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints_xyz(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        join_by = form.pop('join_by')
        (filters, x_filters, y_filters, z_filters) = ([], [], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            elif k.startswith('z-'):
                z_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        rows = list(self.cube.get_data_xyz(join_by=join_by,
                                          filters=filters,
                                          x_filters=x_filters,
                                          y_filters=y_filters,
                                          z_filters=z_filters))
        return self.jsonify({'datapoints': rows})

    def dump_csv(self, response, dialect=csv.excel):
        in_headers = [
            'breakdown',
            'indicator',
            'ref_area',
            'time_period',
            'unit_measure',
            'value']
        out_headers = [
            'indicator',
            'breakdown',
            'unit_measure',
            'time_period',
            'ref_area',
            'value']
        writer = csv.DictWriter(response, out_headers, dialect=dialect, restval='')
        writer.writeheader()
        data = StringIO(self.cube.dump(data_format="text/csv"))
        data.readline() #skip header
        reader = csv.DictReader(data, in_headers, restval='')
        for row in reader:
            encoded_row = {}
            for k,v in row.iteritems():
                encoded_row[k] = unicode(v).encode('utf-8')
            writer.writerow(encoded_row)
        return response

    def download_csv(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/csv; charset=utf-8')
        filename = self.context.getId() + '.csv'
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        return self.dump_csv(response)

    def download_tsv(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/tab-separated-values; charset=utf-8')
        filename = self.context.getId() + '.tsv'
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        return self.dump_csv(response, dialect=csv.excel_tab)

    def dump_rdf(self):
        response = self.request.response
        response.setHeader('Content-type', 'application/rdf+xml; charset=utf-8')
        filename = self.context.getId() + '.rdf'
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        response.write('')
        data = self.cube.dump(data_format='application/rdf+xml')
        response.write(data)
        return response

    def download_codelists(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/rdf+xml; charset=utf-8')
        filename = '%s-codelists.rdf' % self.context.getId()
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        response.write('')
        data = self.cube.dump_constructs(template='construct_codelists.sparql')
        response.write(data)
        return response

    def download_structure(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/rdf+xml; charset=utf-8')
        filename = '%s-structure.rdf' % self.context.getId()
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        response.write('')
        data = self.cube.dump_constructs(template='construct_structure.sparql')
        response.write(data)
        return response

    def download_dataset_metadata(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/rdf+xml; charset=utf-8')
        filename = '%s-metadata.rdf' % self.context.getId()
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        response.write('')
        data = self.cube.dump_constructs(template='construct_dataset_metadata.sparql')
        response.write(data)
        return response
