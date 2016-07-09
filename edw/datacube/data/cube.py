import time
import urllib
import urllib2
import os
import logging
from collections import OrderedDict, defaultdict
import threading
import jinja2
import sparql
import datetime
import re

from decimal import Decimal
from eea.cache import cache as eeacache

SPARQL_DEBUG = bool(os.environ.get('SPARQL_DEBUG') == 'on')

logger = logging.getLogger(__name__)

sparql_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__))
sparql_env.filters.update({
    'literal_n3': lambda value: sparql.Literal(value).n3(),
    'uri_n3': lambda value: sparql.IRI(value).n3(),
})


class QueryError(Exception):
    pass


class DataCache(object):

    def __init__(self):
        self.data = {}
        self.timestamp = None
        self.lock = threading.Lock()

    def ping(self, timestamp):
        with self.lock:
            if timestamp != self.timestamp:
                self.data.clear()
                self.timestamp = timestamp

    def get(self, key, update):
        with self.lock:
            if key not in self.data:
                self.data[key] = update()
            return self.data[key]


data_cache = DataCache()

def cacheKey(method, self, *args, **kwargs):
    """ Generate unique cache id when self has property self.cube
    """
    return (self.cube.endpoint, self.cube.dataset)

def cacheKeyCube(method, self, *args, **kwargs):
    """ Generate unique cache id when self is cube
    """
    return (self.endpoint, self.dataset, args, kwargs)

class CubeMetadata(object):

    def __init__(self, cube):
        self.cube = cube

    def get_dimension_groups(self):
        groups = {}
        query = sparql_env.get_template('group_dimensions.sparql').render(**{
            'dataset': self.cube.dataset,
        })
        return dict([(row['dimension'], row['group_dimension'])
            for row in self.cube._execute(query)])

    def fix_notations(self, row, uri_key='uri'):
        if not row.get('notation'):
            row['notation'] = re.split('[#/]', row[uri_key])[-1]
        if not row.get('label'):
            row['label'] = row['notation']
        if not row.get('short_label'):
            row['short_label'] = row['label']

    def load_dimensions(self):
        # Get info for group dimensions
        groups = self.get_dimension_groups()
        
        group_dimensions = {}
        if groups:
            query = sparql_env.get_template('dimension_info.sparql').render(**{
                'uri_list': groups.values()
            })
            for row in self.cube._execute(query):
                row['type_label'] = 'dimension group'
                row['group_notation'] = None
                row['group_dimension'] = None
                self.fix_notations(row, 'dimension')
                group_dimensions[row['dimension']] = row

        # fix missing notations add group notations
        query = sparql_env.get_template('dimensions.sparql').render(**{
            'dataset': self.cube.dataset,
        })
        result = list(self.cube._execute(query))
        dimensions = []
        for row in result:
            self.fix_notations(row, 'dimension')
            if row['dimension'] in groups:
                group_dimension = group_dimensions[groups[row['dimension']]]
                row['group_notation'] = group_dimension['notation']
                row['group_dimension'] = group_dimension['dimension']
                # add group dimension info
                dimensions.append(group_dimension)
            else:
                row['group_notation'] = None
                row['group_dimension'] = None

            if 'type_label' not in row:
                types = dict([
                    ('http://purl.org/linked-data/cube#dimension', 'dimension'),
                    ('http://purl.org/linked-data/cube#attribute', 'attribute'),
                    ('http://purl.org/linked-data/cube#measure', 'measure'),
                ])
                row['type_label'] = types.get(row['dimension_type'])
            dimensions.append(row)
        return OrderedDict([(row['dimension'], row) for row in dimensions])

    def load_notations(self):
        dimensions = [uri for uri, row in self.dimensions.items()
            if row['type_label'] not in ['measure']]

        result = {}
        for dimension_uri in dimensions:
            dimension_code = self.dimensions[dimension_uri]['notation']
            # Replace dimension_uri with real dimension if grouper
            is_group_dimension = dimension_uri in self.groupers
            if is_group_dimension:
                dimension_uri = self.groupers[dimension_uri]

            # cannot use dimension_options.sparql because metadata is being loaded (lock)
            query = sparql_env.get_template('dimension_options_raw.sparql').render(**{
                'dataset': self.cube.dataset,
                'dimension_uri': dimension_uri,
                'is_group_dimension': is_group_dimension,
            })
            uri_list = [row['uri'] for row in self.cube._execute(query)]

            if not uri_list:
                # can be a dimension with literal values
                continue
            query = sparql_env.get_template('dimension_option_metadata.sparql').render(**{
                'uri_list': uri_list,
            })
            result2 = {}
            # Merge results (UNION is used in SPARQL)
            for row in self.cube._execute(query):
                uri = row['uri']
                if not uri in result2:
                    result2[uri] = {}

                for prop in row:
                    if row[prop] is not None:
                        result2[uri][prop] = row[prop]

            # Patch missing notations and labels
            for row in result2.values():
                self.fix_notations(row, 'uri')
            result[dimension_code] = result2.values()

        return result

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def update(self):
        t0 = time.time()
        logger.info('loading cube metadata')
        self.dimensions = self.load_dimensions()
        logger.info('cube metadata loaded, %.2f seconds', time.time() - t0)
        # prepare some redundant dictionaries for convenient use
        self.grouped_dimensions = dict([(k, v['group_dimension']) for k,v in self.dimensions.items()
            if v['group_dimension']])
        self.groupers = dict((v,k) for k,v in self.grouped_dimensions.items())
        self.dimensions_by_notation = dict([(v['notation'], k) for k,v in self.dimensions.items()])
        self.measure_uri = next( (k for k,v in self.dimensions.items()
            if v['type_label'] == 'measure'), None)
        t0 = time.time()
        logger.info('loading notations')
        self.notations = self.load_notations()
        logger.info('notations loaded, %.2f seconds', time.time() - t0)
        return {
            'dimensions': self.dimensions,
            'notations': self.notations,
            'grouped_dimensions': self.grouped_dimensions,
            'groupers': self.groupers,
            'dimensions_by_notation': self.dimensions_by_notation,
            'measure_uri': self.measure_uri,
            'notations_by_code': {},
            'notations_by_uri': {},
        }

    def get(self):
        cache_key = (self.cube.endpoint, self.cube.dataset, self.__class__.__name__)
        return data_cache.get(cache_key, self.update)

    def get_all_dimensions_uri(self):
        data = self.get()
        return [uri for uri, row in data['dimensions'].items()
            if row['type_label'] in ['dimension', 'dimension group']]

    def get_real_dimensions_and_attributes(self):
        data = self.get()
        return [uri for uri, row in data['dimensions'].items()
            if row['type_label'] in ['dimension', 'attribute']]

    def get_all_dimensions_and_attributes(self):
        data = self.get()
        return [uri for uri, row in data['dimensions'].items()
            if row['type_label'] not in ['measure']]

    def lookup_dimension_code(self, dimension_uri):
        data = self.get()
        return data['dimensions'][dimension_uri]['notation']

    def lookup_dimension_uri(self, dimension_code):
        data = self.get()
        return data['dimensions_by_notation'][dimension_code]

    def lookup_dimension_uri_by_grouper_uri(self, grouper_dimension):
        data = self.get()
        return data['groupers'][grouper_dimension]

    def lookup_measure_uri(self):
        data = self.get()
        return data['measure_uri']

    def lookup_notation(self, dimension_code, notation):
        data = self.get()
        return filter( lambda item: item['notation'] == notation, 
                       data['notations'][dimension_code])[0]

    def lookup_metadata(self, dimension_code, uri):
        data = self.get()
        return filter( lambda item: item['uri'] == uri, 
                       data['notations'][dimension_code])[0]

    def lookup_dimension_info(self, dimension_uri):
        data = self.get()
        return data['dimensions'][dimension_uri]

    def is_group_dimension(self, dimension_uri):
        data = self.get()
        return dimension_uri in data['groupers']

    def is_grouped_dimension(self, dimension_uri):
        data = self.get()
        return dimension_uri in data['grouped_dimensions']

    def lookup_grouper_dimension(self, dimension_uri):
        data = self.get()
        return data['grouped_dimensions'][dimension_uri]

class Cube(object):

    def __init__(self, endpoint, dataset):
        self.endpoint = endpoint
        self.dataset = dataset
        if dataset:
            self.metadata = CubeMetadata(self)

    def _execute(self, query, as_dict=True):
        t0 = time.time()
        if SPARQL_DEBUG:
            logger.info('Running query: \n%s', query)
        try:
            query_object = sparql.Service(self.endpoint).createQuery()
            query_object.method = 'POST'
            res = query_object.query(query)
        except urllib2.HTTPError, e:
            if SPARQL_DEBUG:
                logger.info('Query failed')
            if 400 <= e.code < 600:
                raise QueryError(e.fp.read())
            else:
                raise
        if SPARQL_DEBUG:
            logger.info('Query took %.2f seconds.', time.time() - t0)
        rv = (sparql.unpack_row(r) for r in res)
        if as_dict:
            return (dict(zip(res.variables, r)) for r in rv)
        else:
            return rv

    def get_datasets(self):
        query = sparql_env.get_template('datasets.sparql').render()
        return list(self._execute(query))

    def get_dataset_metadata(self, dataset):
        query = sparql_env.get_template('dataset_metadata.sparql').render(**{
            'dataset': dataset,
        })
        return list(self._execute(query))[0]

    def get_dimension_metadata(self):
        return self.metadata.get()['notations']

    def get_dataset_details(self):
        sparql_template = 'indicator_time_coverage.sparql'
        query = sparql_env.get_template(sparql_template).render(**{
            'dataset': self.dataset,
        })

        # indicator, maxYear, minYear
        res = list(self._execute(query))
        res_by_uri = {row['indicator']: row for row in res}

        meta_list = self.get_dimension_option_metadata_list(
            'indicator', list(res_by_uri)
        )

        for meta in meta_list:
            uri = meta.get('uri', None)
            #meta['sourcelabel'] = meta.get('source_label', None)
            #meta['sourcelink'] = meta.get('source_url', None)
            #meta['sourcenotes'] = meta.get('source_notes', None)
            #meta['notes'] = meta.get('note', None)
            #meta['longlabel'] = meta.get('label', None)
            res_by_uri[uri].update(meta)


        def _sort_key(item):
            try:
                parent = int(item.get('parent_order', None))
            except (ValueError, TypeError):
                parent = 9999
            try:
                inner = int(item.get('inner_order', None))
            except (ValueError, TypeError):
                inner = 9999
            return (parent, item.get('group_name', None), inner)

        return sorted(res, key=_sort_key)

    @eeacache(cacheKeyCube, dependencies=['edw.datacube'])
    def get_dimensions(self, flat=False):
        dimensions = self.metadata.get()['dimensions'].values()
        if flat:
            return dimensions
        else:
            rv = defaultdict(list)
            for row in dimensions:
                rv[row['type_label']].append({
                    'label': row['label'],
                    'notation': row['notation'],
                    'comment': row['comment'] or row['dimension'],
                    'uri': row['uri'],
                })
            return dict(rv)

    @eeacache(cacheKeyCube, dependencies=['edw.datacube'])
    def get_dimension_options(self, dimension, filters=[]):
        # fake an n-dimensional query, with a single dimension, that has no specific filters
        n_filters = [[]]
        return self.get_dimension_options_n(dimension, filters, n_filters)

    def get_dimension_options_xy(self, dimension,
                                 filters, x_filters, y_filters,
                                 x_dataset='', y_dataset=''):
        n_filters = [x_filters, y_filters]
        n_datasets = [x_dataset, y_dataset] if x_dataset and y_dataset else []
        return self.get_dimension_options_n(dimension, filters, n_filters, n_datasets)

    def get_dimension_options_xyz(self, dimension,
                                  filters, x_filters, y_filters, z_filters):
        n_filters = [x_filters, y_filters, z_filters]
        return self.get_dimension_options_n(dimension, filters, n_filters)

    def get_dimension_options_n(self, dimension_code, filters, n_filters, n_datasets=[]):
        common_uris = None
        result_sets = []
        intervals = []
        merged_intervals = []
        uri_list = None
        distinct_types = False
        comparator = None
        dimension_uri = self.metadata.lookup_dimension_uri(dimension_code)

        for idx, extra_filters in enumerate(n_filters):
            if n_datasets:
                dataset = n_datasets[idx]
            else:
                dataset = self.dataset
            query = sparql_env.get_template('dimension_options.sparql').render(**{
                'dataset': dataset,
                'dimension_uri': dimension_uri,
                'filters': filters + extra_filters,
                'metadata': self.metadata,
            })
            result_sets.append(list(self._execute(query)))

        def options(res):
            return set(item['uri'] for item in res)

        def get_interval_type(item):
            uri = item.get('uri')
            return uri.split('/')[-2]

        # Make an uri list containing all uris and a common uri list containing
        # uris common for all result sets
        for res in result_sets:
            if not comparator:
                if res:
                    comparator = get_interval_type(res[0])
            else:
                for elem in res:
                    if comparator != get_interval_type(elem):
                        distinct_types = True
            res = options(res)
            if uri_list is None:
                uri_list = res
                common_uris = res
            else:
                uri_list = uri_list | res
                common_uris = common_uris & res
        if dimension_code == 'time-period' and distinct_types:
            # Query the intervals
            query_intervals = sparql_env.get_template('dimension_options_intervals.sparql').render(**{
                'uri_list': uri_list,
            })
            intervals.append(list(self._execute(query_intervals)))

            merged_intervals = [item for interval in intervals
                                for item in interval]

            # re-calculate common_uris based on parent interval
            common_uris = None
            for res in result_sets:
                # get years from merged_intervals
                years = []
                for option in options(res):
                    # search in merged_intervals
                    interval = filter(lambda interval:interval['uri'] == option, merged_intervals)
                    if ( interval == [] ):
                        # this was a year, not present in merged_intervals
                        years.append(option)
                    else:
                        years.append(interval[0]['parent_year'])
                if common_uris is None:
                    common_uris = set(years)
                else:
                    common_uris = common_uris & set(years)
        data = []
        for uri in common_uris:
            data.append((uri, dimension_code))

        # duplicates - e.g. when a breakdown is member of several breakdown groups
        labels1 = self.get_labels_with_duplicates(data)
        if labels1:
            labels1.sort(key=lambda item: int(item.get('order') or '0'))
            # filter labels1 by group_notation if present in filters
            if self.metadata.is_grouped_dimension(dimension_uri):
                grouper_dimension = self.metadata.lookup_grouper_dimension(dimension_uri)
                grouper_notation = self.metadata.lookup_dimension_code(grouper_dimension)
                filtered_group = next((value for dimension, value in filters if dimension == grouper_notation), None)
                if filtered_group:
                    labels1 = [x for x in labels1 if x['group_notation'] == filtered_group]
        return labels1

    def get_dimension_codelist(self, dimension_code):
        # list of {?uri ?notation ?label}
        # TODO possibly refactor
        return self.get_dimension_metadata()[dimension_code]

    def get_labels_with_duplicates(self, data):
        if len(data) < 1:
            return {}
        tmpl = sparql_env.get_template('labels.sparql')
        uri_list = []
        for item in data:
            # item = (uri, dimension)
            uri_list.append(item[0])
        query = tmpl.render(**{
            'uri_list': uri_list,
        })
        return list(self._execute(query))

    def get_labels(self, data):
        result = self.get_labels_with_duplicates(data)
        labels = {row['uri']:row for row in result}
        return labels

    def get_dimension_option_metadata_list(self, dimension_code, uri_list):
        return [self.metadata.lookup_metadata(dimension_code, uri) for uri in uri_list]

    def get_dimension_option_metadata(self, dimension_code, option):
        return self.metadata.lookup_notation(dimension_code, option)

    def get_columns(self):
        columns_map = {}
        for dimension in self.metadata.get_real_dimensions_and_attributes():
            item = self.metadata.lookup_dimension_info(dimension)
            name = item['notation']
            if name not in columns_map:
                columns_map[name] = {
                    "uri": item['dimension'],
                    "optional": True,
                    "notation": item['notation'],
                    "name": name,
                }
            if item['type_label'] == 'dimension':
                columns_map[name]['optional'] = False
        return columns_map.values()

    def get_observations(self, filters):
        # returns a generator, see _format_observations_result
        columns = self.get_columns()
        query = sparql_env.get_template('data_and_attributes.sparql').render(**{
            'dataset': self.dataset,
            'columns': columns,
            'filters': filters,
            'metadata': self.metadata,
        })
        result = list(self._execute(query, as_dict=False))
        # Keep only URI's in data, to lookup additional metadata attributes
        def reducer(memo, item):
            def uri_filter(x):
                return True if x and x.startswith('http://') else False
            # last element is supposed to be the value, always a literal
            x = [uri for uri in item[:-1]]
            return memo.union(set(filter(uri_filter, x)))

        data = reduce(reducer, result, set())
        column_names = [item['notation'] for item in columns] + ['value']
        return self._format_observations_result(result, column_names, data)

    def _format_observations_result(self, result, columns, data):
        # Append extra metadata e.g. notation, label etc.
        # data is set of (code_uri, dimension_code)
        for row in result:
            result_row = []
            value = row.pop(-1)
            for idx, item in enumerate(row):
                if item not in data:
                    # this is not an URI
                    result_row.append(item)
                else:
                    # this is an URI, add as dict
                    meta = self.metadata.lookup_metadata(columns[idx], item)
                    result_row.append(
                        {'notation': meta['notation'],
                         'inner_order': meta.get('inner_order', 0),
                         'label': meta['label'],
                         'short-label': meta['short_label'],}
                    )
            if type(value) == type(Decimal()):
                value = float(value)
            result_row.append(value)
            yield dict(zip(columns, result_row))

    def get_observations_cp(self, filters, whitelist_items):
        columns = self.get_columns()

        indicator_group = dict(filters)['indicator-group']
        filters = [(key, item) for key, item in filters if key != 'indicator-group']
        whitelist = []
        for item in whitelist_items:
            mapped_item = {}
            if item['indicator-group'].lower() == indicator_group.lower():
                for n, col in enumerate(columns, 1):
                    name = col['notation']
                    if name in ['indicator', 'breakdown', 'unit-measure']:
                        mapped_item[n] = self.metadata.lookup_notation(
                                            name, item[name])['uri']
                whitelist.append(mapped_item)
        if not whitelist:
            return []
        query = sparql_env.get_template('data_and_attributes_cp.sparql').render(**{
            'dataset': self.dataset,
            'columns': columns,
            'filters': filters,
            'metadata': self.metadata,
            'whitelist': whitelist,
        })
        result = list(self._execute(query, as_dict=False))
        def reducer(memo, item):
            def uri_filter(x):
                # last element is supposed to be the value, always a literal
                return True if x and x.startswith('http://') else False
            x = [uri for uri in item[:-1]]
            return memo.union(set(filter(uri_filter, x)))

        data = reduce(reducer, result, set())
        column_names = [item['notation'] for item in columns] + ['value']
        return self._format_observations_result(result, column_names, data)

    def get_data_xy(self, join_by, filters, x_filters, y_filters):
        n_filters = [x_filters, y_filters]
        return self.get_data_n(join_by, filters, n_filters)

    def get_data_xyz(self, join_by, filters, x_filters, y_filters, z_filters):
        n_filters = [x_filters, y_filters, z_filters]
        return self.get_data_n(join_by, filters, n_filters)

    def get_data_n(self, join_by, filters, n_filters):
        # GET COLUMNS AND COLUMNS NAMES
        columns = self.get_columns()
        columns_names = [item['notation'] for item in columns] + ['value']

        # GET DATA AND ATTRIBUTES
        raw_data = []
        idx = 0
        for extra_filters in n_filters:
            query = sparql_env.get_template('data_and_attributes.sparql').render(**{
                'dataset': self.dataset,
                'columns': columns,
                'filters': filters + list(extra_filters),
                'metadata': self.metadata,
            })
            container = {}
            data = self._execute(query, as_dict=False)
            dict_data = []
            for item in data:
                if type(item[-1]) == type(Decimal()):
                    item[-1] = float(item[-1])
                dict_data.append(
                        dict(zip(columns_names, item)))

            # only keep the entry with largest time_period value
            time_periods = {}
            for row in dict_data:
                join_vaule = row[join_by]
                time_periods[join_vaule] = max([time_periods.get(join_vaule),
                                                row['time-period']])
            dict_data = [row for row in dict_data
                         if time_periods[row[join_by]] == row['time-period']]
            raw_data.append(dict_data)

        # JOIN DATA
        def find_common(memo, item):
            join_set = [it[join_by] for it in item]
            temp_common = set(memo).intersection(set(join_set))
            return temp_common
        common = reduce(find_common, raw_data, [it[join_by] for it in raw_data[0]])

        # EXTRACT UNIQUE URIS FROM DATA
        by_category = defaultdict(list)
        data = set()
        for obs_set in raw_data:
            for obs in obs_set:
                if obs[join_by] in common:
                    by_category[obs[join_by]].append(obs)
                    for key, value in obs.items():
                        if isinstance(value, basestring) and value.startswith('http://'):
                            data.add((value, key))

        # GET LABELS FOR URIS
        labels = self.get_labels(data)

        filtered_data = []
        # EXTRACT COMMON ROWS
        dimensions = ['x', 'y', 'z']
        single_keys = [f[0] for f in filters] + [join_by]
        for obs_list in by_category.values():
            idx = 0
            out = defaultdict(dict)
            for obs in obs_list:
                for key in columns_names:
                    if key not in single_keys:
                        out[key][dimensions[idx]] = obs[key]
                    else:
                        out[key] = obs[key]
                for k, v in out.items():
                    if k not in single_keys:
                        uri_labels = labels.get(v[dimensions[idx]], v[dimensions[idx]])
                        if uri_labels:
                            out[k][dimensions[idx]] = uri_labels
                    else:
                        uri_labels = labels.get(v, None)
                        if uri_labels:
                            out[k] = uri_labels
                idx+=1
            filtered_data.append(out)
        return filtered_data

    def get_revision(self):
        query = sparql_env.get_template('last_modified.sparql').render()
        try:
            timestamp = unicode(next(self._execute(query)).get('modified'))
        except StopIteration:
            timestamp = unicode(datetime.date.today().strftime("%Y-%m-%d %H:%M:%S"))
        data_cache.ping(timestamp)
        return timestamp

    def dump_constructs(self, format='application/rdf+xml',
                        template='construct_codelists.sparql'):
        """
        Sends queries directly to the endpoint.
        Returns the Virtuoso response.
        """

        query = sparql_env.get_template(template).render(**{
            'dataset': self.dataset,
            'dimensions': self.metadata.get_all_dimensions_and_attributes()
        })
        data = urllib.urlencode({
            'query': query,
            'format': format
        })
        return urllib2.urlopen(self.endpoint, data=data).read()
